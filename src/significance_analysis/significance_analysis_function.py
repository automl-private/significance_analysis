import os
import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from pymer4.models import Lmer

# import matplotlib.backends.backend_tkagg as tkagg
# from matplotlib.backends.backend_agg import FigureCanvasAgg
# from matplotlib.figure import Figure
# import tkinter as tk


def checkSignificance(
    data: pd.DataFrame,
    metric: str,
    system_id: str,
    input_id: str,
    bin_id: str = None,
    bin_labels: list[str] = None,
    bin_dividers: list[float] = None,
    subset: typing.Tuple[str, typing.Union[str, list[str]]] = None,
    show_plots: typing.Union[list[bool], bool] = True,
):
    if subset is not None and isinstance(subset[1], str):
        if subset[1] == "all" or subset[1] == "a":
            for subset_item in list(data[subset[0]].unique()):
                checkSignificance(
                    data.loc[data[subset[0]] == subset_item],
                    metric,
                    system_id,
                    input_id,
                    bin_id,
                    bin_labels,
                    bin_dividers,
                    show_plots=show_plots,
                )
        elif subset[1] in data[subset[0]].unique():
            checkSignificance(
                data.loc[data[subset[0]] == subset[1]],
                metric,
                system_id,
                input_id,
                bin_id,
                bin_labels,
                bin_dividers,
                show_plots=show_plots,
            )
        else:
            raise SystemExit(
                'Benchmark-Name not in Dataset. Use "all" or "a" to analyse all benchmarks individually or use valid benchmark name/list of valid benchmark names.'
            )
    elif subset is not None and isinstance(subset[1], list):
        print(subset)
        print(type(subset))
        print(subset[1])
        if isinstance(subset[1], str):
            for subset_item in subset[1]:
                if subset_item not in data[subset[0]].unique():
                    raise SystemExit("Subset-Name not in Dataset.")
                checkSignificance(
                    data.loc[data[subset[0]] == subset_item],
                    metric,
                    system_id,
                    input_id,
                    bin_id,
                    bin_labels,
                    bin_dividers,
                    show_plots=show_plots,
                )
        else:
            raise SystemExit("Subset need to be list of strings.")
    else:

        if isinstance(show_plots, bool):
            show_plots = [show_plots, show_plots]

        def GLRT(mod1, mod2):
            chi_square = 2 * abs(mod1.logLike - mod2.logLike)
            delta_params = abs(len(mod1.coefs) - len(mod2.coefs))
            return {
                "chi_square": chi_square,
                "df": delta_params,
                "p": 1 - stats.chi2.cdf(chi_square, df=delta_params),
            }

        pd.options.mode.chained_assignment = None
        pd.set_option("display.max_rows", 5000)
        pd.set_option("display.max_columns", 5000)
        pd.set_option("display.width", 10000)

        if bin_id is not None and bin_labels is not None and bin_dividers is not None:
            if not 0 in bin_dividers:
                bin_dividers.append(0)
            if not 1 in bin_dividers:
                bin_dividers.append(1)
            bin_dividers.sort()
            if len(bin_labels) != (len(bin_dividers) - 1):
                raise SystemExit("Dividiers do not fit divider-labels")

        if show_plots[0]:
            _, ax = plt.subplots()
            ax.boxplot(
                [group[metric] for _, group in data.groupby(system_id)],
                labels=data[system_id].unique(),
            )
            plt.yscale("log")
            plt.show()

        # System-identifier: system_id
        # Input-Identifier: input_id
        # Two models, "different"-Model assumes significant difference between performance of groups, divided by system-identifier
        # Formula has form: "metric ~ system_id + (1 | input_id)"
        differentMeans_model = Lmer(
            formula=metric + " ~ " + system_id + " + (1 | " + input_id + ")", data=data
        )

        # factors specifies names of system_identifier, i.e. Baseline, or Algorithm1
        differentMeans_model.fit(
            factors={system_id: list(data[system_id].unique())},
            REML=False,
            summarize=False,
        )

        # "Common"-Model assumes no significant difference, which is why the system-identifier is not included
        commonMean_model = Lmer(formula=metric + " ~ (1 | " + input_id + ")", data=data)
        commonMean_model.fit(REML=False, summarize=False)

        # Signficant p-value shows, that different-Model fits data sign. better, i.e.
        # There is signficant difference in system-identifier
        result_GLRT_dM_cM = GLRT(differentMeans_model, commonMean_model)
        p_value = result_GLRT_dM_cM["p"]
        print(f"P-value: {p_value}")
        if result_GLRT_dM_cM["p"] < 0.05:
            print(
                f"\nAs the p-value {p_value} is smaller than 0.05, we can reject the Null-Hypothesis that the model "
                f"that does not consider the {system_id} describes the data as well as the one that does. Therefore "
                f"there is significant difference within {system_id}.\n"
            )
        else:
            print(
                f"\nAs the p-value {p_value} is not smaller than 0.05, we cannot reject the Null-Hypothesis that the model "
                f"that does not consider the {system_id} describes the data as well as the one that does. Therefore "
                f"there is no significant difference within {system_id}\n."
            )

        # Post hoc divides the "different"-Model into its three systems
        post_hoc_results = differentMeans_model.post_hoc(marginal_vars=[system_id])
        contrasts = post_hoc_results[1]
        for pair in contrasts["Contrast"]:
            contrasts.loc[contrasts["Contrast"] == pair, system_id + "_1"] = pair.split(
                " - "
            )[0]
            contrasts.loc[contrasts["Contrast"] == pair, system_id + "_2"] = pair.split(
                " - "
            )[1]
        contrasts = contrasts.drop("Contrast", axis=1)
        column = contrasts.pop(system_id + "_2")
        contrasts.insert(0, system_id + "_2", column)
        column = contrasts.pop(system_id + "_1")
        contrasts.insert(0, system_id + "_1", column)
        # [0] shows group-means, i.e. performance of the single system-groups
        print(post_hoc_results[0])  # cell (group) means
        # [1] shows the pairwise comparisons, i.e. improvements over each other, with p-value
        print(contrasts)  # contrasts (group differences)
        best_system_id = post_hoc_results[0].loc[
            post_hoc_results[0]["Estimate"].idxmin()
        ][system_id]
        contenders = []
        for _, row in contrasts.iterrows():
            if row[system_id + "_1"] == best_system_id and not row["Sig"] in [
                "*",
                "**",
                "***",
            ]:
                contenders.append(row[system_id + "_2"])
            if row[system_id + "_2"] == best_system_id and not row["Sig"] in [
                "*",
                "**",
                "***",
            ]:
                contenders.append(row[system_id + "_1"])

        if contenders:
            print(
                f"The best performing {system_id} is {best_system_id}, but {contenders} are only insignificantly worse.\n"
            )
        else:
            print(
                f"The best performing {system_id} is {best_system_id}, all other perform significantly worse.\n"
            )
        # import Orange
        # Generate the critical difference diagram
        # cd = Orange.evaluation.scoring.compute_CD(post_hoc_results[1], alpha=0.05, test="nemenyi")

        # Plot the critical difference diagram
        # Orange.evaluation.graph_ranks(cd, labels=post_hoc_results[1].domain.attributes[0].values)
        # plt.show()

        if not (
            bin_id is not None and bin_labels is not None and bin_dividers is not None
        ):
            return result_GLRT_dM_cM, post_hoc_results

        bins = []
        for div in bin_dividers:
            bins.append(
                np.min(data[bin_id])
                + int((np.max(data[bin_id]) - np.min(data[bin_id])) * div)
            )
        # Bin the data into classes according to bin_dividers
        data = data.assign(
            bin_class=lambda x: pd.cut(
                x[bin_id], bins=bins, labels=bin_labels, include_lowest=True
            )
        )
        # New model "expanded": Divides into system AND bin-classes (Term system:bin_class allows for Cartesian Product, i.e. different Mean for each system and bin-class)
        model_expanded = Lmer(
            (
                metric
                + " ~ "
                + system_id
                + " + bin_class + "
                + system_id
                + ":bin_class + (1 | "
                + input_id
                + ")"
            ),
            data=data,
        )
        model_expanded.fit(
            factors={
                system_id: list(data[system_id].unique()),
                "bin_class": list(data["bin_class"].unique()),
            },
            REML=False,
            summarize=False,
        )
        # Second model "nointeraction" lacks system:src-Term to hypothesise no interaction, i.e. no difference when changing bin-class
        model_nointeraction = Lmer(
            metric + " ~ " + system_id + " + bin_class + (1 | " + input_id + ")",
            data=data,
        )
        model_nointeraction.fit(
            factors={
                system_id: list(data[system_id].unique()),
                "bin_class": list(data["bin_class"].unique()),
            },
            REML=False,
            summarize=False,
        )

        # If it's significant, look at if different systems perform better at different bin-classes
        result_GLRT_ex_ni = GLRT(model_expanded, model_nointeraction)
        p_value = result_GLRT_ex_ni["p"]
        print(f"P-value: {p_value}")
        if p_value < 0.05:
            print(
                f"\nAs the p-value {p_value} is smaller than 0.05, we can reject the Null-Hypothesis that the model "
                f"that does not consider the {system_id} and the {bin_id} describes the data as well as the one that does. Therefore "
                f"there is significant difference within {system_id} and the {bin_id}.\n"
            )
        else:
            print(
                f"\nAs the p-value {p_value} is not smaller than 0.05, we cannot reject the Null-Hypothesis that the model "
                f"that does not consider the {system_id} and the {bin_id} describes the data as well as the one that does. Therefore "
                f"there is no significant difference within {system_id} and {bin_id}\n."
            )

        post_hoc_results2 = model_expanded.post_hoc(
            marginal_vars=system_id, grouping_vars="bin_class"
        )
        # Means of each combination
        print(post_hoc_results2[0])
        # Comparisons for each combination
        for bin_class in bin_labels:
            contrasts = post_hoc_results2[1].query("bin_class == '" + bin_class + "'")

            for pair in contrasts["Contrast"]:
                contrasts.loc[
                    contrasts["Contrast"] == pair, system_id + "_1"
                ] = pair.split(" - ")[0]
                contrasts.loc[
                    contrasts["Contrast"] == pair, system_id + "_2"
                ] = pair.split(" - ")[1]
            contrasts = contrasts.drop("Contrast", axis=1)
            column = contrasts.pop(system_id + "_2")
            contrasts.insert(0, system_id + "_2", column)
            column = contrasts.pop(system_id + "_1")
            contrasts.insert(0, system_id + "_1", column)
            print(
                contrasts[
                    (contrasts["Sig"] == "***")
                    | (contrasts["Sig"] == "**")
                    | (contrasts["Sig"] == "*")
                ]
            )
            best_system_id = (
                post_hoc_results2[0]
                .query("bin_class == '" + bin_class + "'")
                .loc[
                    post_hoc_results2[0]
                    .query("bin_class == '" + bin_class + "'")["Estimate"]
                    .idxmin()
                ][system_id]
            )
            contenders = []
            for _, row in contrasts.iterrows():
                if row[system_id + "_1"] == best_system_id and not row["Sig"] in [
                    "*",
                    "**",
                    "***",
                ]:
                    contenders.append(row[system_id + "_2"])
                if row[system_id + "_2"] == best_system_id and not row["Sig"] in [
                    "*",
                    "**",
                    "***",
                ]:
                    contenders.append(row[system_id + "_1"])
            if contenders:
                print(
                    f"The best performing {system_id} in bin-class {bin_class} is {best_system_id}, but {contenders} are only insignificantly worse.\n"
                )
            else:
                print(
                    f"The best performing {system_id} in bin-class {bin_class} is {best_system_id}, all other perform significantly worse.\n"
                )

        if show_plots[1]:
            _, ax = plt.subplots(figsize=(10, 6))
            for sys_id, group in post_hoc_results2[0].groupby(system_id):
                ax.errorbar(
                    group["bin_class"],
                    group["Estimate"],
                    yerr=group["SE"],
                    fmt="o-",
                    capsize=1,
                    label=sys_id,
                    lolims=group["2.5_ci"],
                    uplims=group["97.5_ci"],
                )
            ax.set_xlabel(bin_id)
            ax.set_ylabel("Estimate")
            ax.set_title(f"Estimates by {system_id} and {bin_id}")
            ax.legend()
            plt.show()

        return result_GLRT_dM_cM, post_hoc_results, result_GLRT_ex_ni, post_hoc_results2


###TODO: Edit Main!
if __name__ == "__main__":
    dfList = []
    folders = ["./dataset_secondRun", "./dataset_q_rs_run", "./dataset_SR_KG_run"]
    for folder in folders:
        filesList = os.listdir(folder)
        for file in filesList:
            df = pd.read_pickle(folder + "/" + file)
            # print(df)
            df["mean"] = df["mean"].cummin()
            if df["aquisition"].unique()[0][0] == "q":
                df["acqu_class"] = "MonteCarlo"
            elif df["aquisition"].unique()[0] == "randomSearch":
                df["acqu_class"] = "RandomSearch"
            else:
                df["acqu_class"] = "Analytical"
            dfList.append(df)
    data = pd.concat(dfList)
    data = data.reset_index()
    data = (
        data.drop("n", axis=1)
        .drop("frac_nonnull", axis=1)
        .drop("metric_name", axis=1)
        .drop("index", axis=1)
    )
    data = data.rename(columns={"aquisition": "acquisition"}).sort_values(
        ["acquisition", "benchmark", "seed", "budget"]
    )
    data = data.loc[data["budget"] >= 9]
    # data=data.drop('frac_nonnull#,errors='ignore')
    # data = pd.read_pickle("./sign_analysis_example/example_dataset.pkl")
    metric = "mean"
    system_id = "acqu_class"
    input_id = "benchmark"
    bin_id = "budget"
    bin_labels = ["16-36%", "37-56%", "57-76%", "77-96%", "97-100%"]
    bin_dividers = [0.24, 0.48, 0.72, 0.96]
    print(data)
    checkSignificance(
        data,
        metric,
        system_id,
        input_id,
        bin_id,
        bin_labels,
        # ["49","50"],
        # [0.5],
        bin_dividers,
        show_plots=[True, True],
    )
    print("Done")


# pip install signficance-analysis

# bin_labels = ["16-36%", "37-56%", "57-76%", "77-96%", "97-100%"]
# bin_dividers = [0.24, 0.48, 0.72, 0.96]


# checkSignificance(data,"mean","acqu_class","benchmark","budget")
