import os
import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
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

        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", None)
        pd.options.display.max_rows = 500
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

        if bin_id is not None and bin_labels is not None and bin_dividers is not None:
            if not 0 in bin_dividers:
                bin_dividers.append(0)
            if not 1 in bin_dividers:
                bin_dividers.append(1)
            bin_dividers.sort()
            if len(bin_labels) != (len(bin_dividers) - 1):
                raise SystemExit("Dividiers do not fit divider-labels")

        if show_plots[0]:

            sns.set(style="darkgrid")
            g = sns.FacetGrid(data, col=system_id, col_wrap=3, height=4)
            g.map(
                sns.regplot,
                bin_id,
                metric,
                lowess=True,
                scatter_kws={"s": 10, "alpha": 0.5},
            )
            # Generate contour lines for entire dataset
            # sns.kdeplot(x=data[bin_id], y=data[metric], levels=5, alpha=0.5, ax=g.fig.gca())

            plt.show()

        # System-identifier: system_id
        # Input-Identifier: input_id
        # Two models, "different"-Model assumes significant difference between performance of groups, divided by system-identifier
        # Formula has form: "metric ~ system_id + (1 | input_id)"
        differentMeans_model = Lmer(
            formula=metric + " ~ " + system_id + " + (1 | " + input_id + ")", data=data
        )

        print(data)
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
        print(result_GLRT_dM_cM)

        # Post hoc divides the "different"-Model into its three systems
        post_hoc_results = differentMeans_model.post_hoc(marginal_vars=[system_id])
        # [0] shows group-means, i.e. performance of the single system-groups
        print(post_hoc_results[0])  # cell (group) means
        # [1] shows the pairwise comparisons, i.e. improvements over each other, with p-value
        print(post_hoc_results[1])  # contrasts (group differences)

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
        print(result_GLRT_ex_ni)  # test interaction

        post_hoc_results2 = model_expanded.post_hoc(
            marginal_vars=system_id, grouping_vars="bin_class"
        )

        if show_plots[1]:

            sns.catplot(
                x="bin_class",
                y="Estimate",
                hue=system_id,
                kind="point",
                data=post_hoc_results2[0],
                capsize=0.1,
                errorbar="sd",
                height=6,
                aspect=1.5,
            )

            # Set the axis labels and title
            plt.xlabel("Bin Class")
            plt.ylabel("Estimated Mean")
            plt.title("Interaction Plot of Estimated Means")

            # Show the plot
            plt.show()

        # Means of each combination
        print(post_hoc_results2[0])
        # Comparisons for each combination
        for bin_class in bin_labels:
            print(
                post_hoc_results2[1].query("bin_class == '" + bin_class + "'")[
                    (
                        post_hoc_results2[1].query("bin_class == '" + bin_class + "'")[
                            "Sig"
                        ]
                        == "***"
                    )
                    | (
                        post_hoc_results2[1].query("bin_class == '" + bin_class + "'")[
                            "Sig"
                        ]
                        == "**"
                    )
                    | (
                        post_hoc_results2[1].query("bin_class == '" + bin_class + "'")[
                            "Sig"
                        ]
                        == "*"
                    )
                ]
            )

        return result_GLRT_dM_cM, post_hoc_results, result_GLRT_ex_ni, post_hoc_results2


###TODO: Edit Main!
if __name__ == "__main__":
    dfList = []
    folders = ["./dataset_secondRun", "./dataset_qrun"]
    for folder in folders:
        filesList = os.listdir(folder)
        for file in filesList:
            df = pd.read_pickle(folder + "/" + file)
            # print(df)
            df["mean"] = df["mean"].cummin()
            # print(df)
            dfList.append(df)
    data = pd.concat(dfList)
    data = data.reset_index()
    data = (
        data.drop("n", axis=1)
        .drop("frac_nonnull", axis=1)
        .drop("metric_name", axis=1)
        .drop("index", axis=1)
    )
    # data=data.loc[data["budget"] <9]
    # data=data.drop('frac_nonnull#,errors='ignore')
    # data = pd.read_pickle("./sign_analysis_example/example_dataset.pkl")
    metric = "mean"
    system_id = "aquisition"
    input_id = "benchmark"
    bin_id = "budget"
    bin_labels = ["0-16% (Sobol)", "16-37%", "37-58%", "58-79%", "79-100%"]
    bin_dividers = [0.16, 0.37, 0.58, 0.79, 1]
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
