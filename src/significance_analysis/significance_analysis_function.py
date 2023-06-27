import typing

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from pymer4.models import Lmer


def conduct_analysis(
    data: pd.DataFrame,
    metric: str,
    system_id: str,
    input_id: str,
    bin_id: str = None,
    bins: typing.Union[list[list[str]], list[float]] = None,
    bin_labels: list[str] = None,
    subset: typing.Tuple[str, typing.Union[str, list[str], list[list[str]]]] = None,
    show_plots: bool = True,
    summarize: bool = True,
) -> typing.Union[
    typing.Tuple[dict[str, any], typing.Tuple[any, pd.DataFrame]],
    typing.Dict[str, typing.Tuple[dict[str, any], typing.Tuple[any, pd.DataFrame]]],
]:
    """LMER-Based Performance Analysis

    Args:
        data (pd.DataFrame): Dataset
        metric (str): Column name of metric (e.g. Mean) in dataset
        system_id (str): Column name of system (e.g. Algorithm) in dataset
        input_id (str): Column name of input (e.g. Benchmark) in dataset
        bin_id (str, optional): Column name of bin (e.g. Budget). Defaults to None.
        bins (typing.Union[list[list[str]], list[float]], optional): Specified bins: If None, bins for every unique value are used. If list of float, numeric variable gets binned into intervals according to list. If list of list of str, variable gets binned into bins according to sublists. Defaults to None.
        bin_labels (list[str], optional): Labels for bins. If None, bins are named after content/interval borders. If list of str, bins are named according to list. Defaults to None.
        subset (typing.Tuple[str, typing.Union[str, list[str], list[list[str]]]], optional): Subset of dataset that should be used for analysis. First entry of tuple is name of variable that defines subset. Second entry is either list of entries that should be analysed iteratively (single entries or groups), or "all"/"a" for all entries. Defaults to None.
        show_plots (bool, optional): Show plots. First entry is boxplot comparing systems, second entry is graph showing systems in all bins. Defaults to True.
        summarize (bool, optional): Print results while analysing. Defaults to True.

    Raises:
        SystemExit: If Subset-Value is not in Dataset.
        SystemExit: If the number of Labels does not fit the number of numeric Bins.
        SystemExit: If the number of Labels does not fit the number of categorical Bins.

    Returns:
        typing.Tuple[dict[str,any],typing.Tuple[any,pd.DataFrame]]: First the result-dictionary of the GLRT and second the post_hoc-analysis of the LMEM.
    """

    if subset is not None:
        result_dict = {}
        if isinstance(subset[1], str):
            if subset[1] in ["all", "a", "All", "A"]:
                subset_list = list(data[subset[0]].unique())
            else:
                subset_list = [subset[1]]
        else:
            subset_list = subset[1]
        for subset_item in subset_list:
            print(f"Analysis for {subset_item}")
            if isinstance(subset_item, str):
                subset_item = [subset_item]
            if any(item not in data[subset[0]].unique() for item in subset_item):
                raise SystemExit(
                    f"A Subset-Value of {subset_item} is not in Dataset. Choose from {data[subset[0]].unique()}"
                )
            result_dict[subset_item[0]] = conduct_analysis(
                data.loc[data[subset[0]].isin(subset_item)],
                metric,
                system_id,
                input_id,
                bin_id,
                bins=bins,
                bin_labels=bin_labels,
                show_plots=show_plots,
                summarize=summarize,
            )
        return result_dict
    else:

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

        if bin_id is None:
            if len(data[input_id].unique()) == 1:
                data.loc[data.sample(1).index, input_id] = (
                    data[input_id].unique()[0] + "_d"
                )

            if show_plots:
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
                formula=f"{metric}~{system_id}+(1|{input_id})", data=data
            )

            # factors specifies names of system_identifier, i.e. Baseline, or Algorithm1
            differentMeans_model.fit(
                factors={system_id: list(data[system_id].unique())},
                REML=False,
                summarize=False,
            )

            # "Common"-Model assumes no significant difference, which is why the system-identifier is not included
            commonMean_model = Lmer(formula=f"{metric}~ (1 | {input_id})", data=data)
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

            if summarize:
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

            return result_GLRT_dM_cM, post_hoc_results

        else:
            if bins is None:
                data[f"{bin_id}_bins"] = data[bin_id]
            else:
                if isinstance(bins, list) and all(
                    isinstance(bin, (float, int)) for bin in bins
                ):
                    bins_set = set(bins)
                    bins_set.add(data[bin_id].min())
                    bins_set.add(data[bin_id].max())
                    bins = sorted(list(bins_set))
                    if bin_labels is None:
                        bin_labels = [
                            f"{bins[i]}_{bins[i+1]}" for i in range(len(bins) - 1)
                        ]
                    else:
                        if len(bin_labels) != len(bins) + 1:
                            raise SystemExit(
                                f"Too many or too few labels ({len(bin_labels)} labels and {len(bins)} bins)"
                            )
                    data[f"{bin_id}_bins"] = pd.cut(
                        data[bin_id], bins=bins, labels=bin_labels, include_lowest=True
                    )
                else:
                    if bin_labels is not None:
                        if len(bin_labels) != len(bins):
                            raise SystemExit(
                                f"Too many or too few labels ({len(bin_labels)} labels and {len(bins)} bins)"
                            )
                        data[f"{bin_id}_bins"] = data[bin_id].apply(
                            lambda x: bin_labels[
                                bins.index([s for s in bins if x in s][0])
                            ]
                        )
                    else:
                        data[f"{bin_id}_bins"] = data[bin_id].apply(
                            lambda x: "_".join([s for s in bins if x in s][0])
                        )

            # New model "expanded": Divides into system AND bin-classes (Term system:bin_id allows for Cartesian Product, i.e. different Mean for each system and bin-class)
            model_expanded = Lmer(
                f"{metric} ~  {system_id} + {bin_id}_bins + {system_id}:{bin_id}_bins + (1 | {input_id})",
                data=data,
            )
            model_expanded.fit(
                factors={
                    system_id: list(data[system_id].unique()),
                    f"{bin_id}_bins": list(data[f"{bin_id}_bins"].unique()),
                },
                REML=False,
                summarize=False,
            )
            # Second model "nointeraction" lacks system:src-Term to hypothesise no interaction, i.e. no difference when changing bin-class
            model_nointeraction = Lmer(
                f"{metric} ~ {system_id} + {bin_id}_bins + (1 | {input_id})",
                data=data,
            )
            model_nointeraction.fit(
                factors={
                    system_id: list(data[system_id].unique()),
                    f"{bin_id}_bins": list(data[f"{bin_id}_bins"].unique()),
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

            post_hoc_results = model_expanded.post_hoc(
                marginal_vars=system_id, grouping_vars=f"{bin_id}_bins"
            )
            if summarize:
                # Means of each combination
                print(post_hoc_results[0])
            # Comparisons for each combination
            for group in data[f"{bin_id}_bins"].unique():
                contrasts = post_hoc_results[1].query(f"{bin_id}_bins == '{group}'")

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
                if summarize:
                    print(contrasts[contrasts["Sig"] != ""])
                best_system_id = (
                    post_hoc_results[0]
                    .query(f"{bin_id}_bins == '{group}'")
                    .loc[
                        post_hoc_results[0]
                        .query(f"{bin_id}_bins == '{group}'")["Estimate"]
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
                        f"The best performing {system_id} in {bin_id}-class {group} is {best_system_id}, but {contenders} are only insignificantly worse.\n"
                    )
                else:
                    print(
                        f"The best performing {system_id} in {bin_id}-class {group} is {best_system_id}, all other perform significantly worse.\n"
                    )

            if show_plots:
                _, ax = plt.subplots(figsize=(10, 6))
                for sys_id, group in post_hoc_results[0].groupby(system_id):
                    ax.errorbar(
                        group[f"{bin_id}_bins"],
                        group["Estimate"],
                        yerr=group["SE"],
                        fmt="o-",
                        capsize=3,
                        label=sys_id,
                        lolims=group["2.5_ci"],
                        uplims=group["97.5_ci"],
                    )
                ax.set_xlabel(bin_id)
                ax.set_ylabel("Estimate")
                ax.set_title(f"Estimates by {system_id} and {bin_id}")
                ax.legend()
                plt.show()

            return result_GLRT_ex_ni, post_hoc_results
