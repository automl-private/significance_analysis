# import os
import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from pymer4.models import Lmer


def checkSignificance(
    data: pd.DataFrame,
    metric: str,
    system_id: str,
    input_id: str,
    bin_id: str = None,
    bin_labels: list[str] = None,
    bin_dividers: list[float] = None,
    specificBenchmark: typing.Union[str, list[str]] = None,
):
    if specificBenchmark is not None and isinstance(specificBenchmark, str):
        if specificBenchmark == "all" or specificBenchmark == "a":
            for benchmark in list(data[input_id].unique()):
                checkSignificance(
                    data.loc[data[input_id] == benchmark],
                    metric,
                    system_id,
                    input_id,
                    bin_id,
                    bin_labels,
                    bin_dividers,
                )
        elif specificBenchmark in data[input_id].unique():
            checkSignificance(
                data.loc[data[input_id] == specificBenchmark],
                metric,
                system_id,
                input_id,
                bin_id,
                bin_labels,
                bin_dividers,
            )
        else:
            raise SystemExit(
                'Benchmark-Name not in Dataset. Use "all" or "a" to analyse all benchmarks individually or use valid benchmark name/list of valid benchmark names.'
            )
    elif specificBenchmark is not None and isinstance(specificBenchmark, list):
        if isinstance(specificBenchmark[0], str):
            for benchmark in specificBenchmark:
                if benchmark not in data[input_id].unique():
                    raise SystemExit("Benchmark-Name not in Dataset.")
                checkSignificance(
                    data.loc[data[input_id] == benchmark],
                    metric,
                    system_id,
                    input_id,
                    bin_id,
                    bin_labels,
                    bin_dividers,
                )
        else:
            raise SystemExit("Benchmarks need to be list of strings.")
    else:

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

        sns.set(style="darkgrid")

        g = sns.FacetGrid(data, col="algorithm", col_wrap=3, height=4)
        g.map(sns.regplot, "budget", "mean", lowess=True, scatter_kws={"s": 10})
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

        sns.catplot(
            x="bin_class",
            y="Estimate",
            hue="algorithm",
            kind="point",
            data=post_hoc_results2[0],
            capsize=0.1,
            errorbar=("ci", 80),
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
            print(post_hoc_results2[1].query("bin_class == '" + bin_class + "'"))

        return result_GLRT_dM_cM, post_hoc_results, result_GLRT_ex_ni, post_hoc_results2


###TODO: Edit Main!
if __name__ == "__main__":
    # dfList = []
    # filesList = os.listdir("./experimentPlatform/results")
    # for file in filesList:
    #    dfList.append(pd.read_pickle("./experimentPlatform/results/" + file))
    # data = pd.concat(dfList)
    data = pd.read_pickle("./sign_analysis_example/example_dataset.pkl")
    print(data)
    metric = "mean"
    system_id = "algorithm"
    input_id = "benchmark"
    bin_id = "budget"
    bin_labels = ["short", "long"]
    bin_dividers = [0.4, 1]
    checkSignificance(data, metric, system_id, input_id, bin_id, bin_labels, bin_dividers)
    print("Done")
