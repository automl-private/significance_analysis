"""import os
import pickle
import sys

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tabulate
from autorank import autorank, create_report, plot_stats
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator"""
import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymer4.models import Lmer
from scipy import stats
from scipy.stats import rankdata

pd.set_option("chained_assignment", None)
pd.set_option("display.max_rows", 5000)
pd.set_option("display.max_columns", 5000)
pd.set_option("display.width", 10000)


def load_priorband_data():
    df = pd.read_parquet("datasets/priorband_single.parquet")
    df = df.reset_index()
    df_collection = []
    for seed_nr in range(50):
        partial_df = df[["benchmark", "prior", "algorithm", "used_fidelity"]]
        partial_df["value"] = df[f"seed-{seed_nr}"]
        partial_df["seed"] = seed_nr
        df_collection.append(partial_df)
        print(f"⚙️ Seed {seed_nr+1}/50        ", end="\r", flush=True)
    complete_df = pd.concat(df_collection, ignore_index=True)
    print("✅ Loading data done     ")
    return complete_df


def combine_bench_prior(row):
    return f"{row['benchmark']}_{row['prior']}"


def add_rel_ranks(row, data: pd.DataFrame, benchmark: str, time: str):
    values = data.loc[
        (data[benchmark] == row[benchmark])
        & (data["seed"] == row["seed"])
        & (data[time] == row[time])
    ]["value"].values
    ranked_data = rankdata(values)
    return ranked_data[values.tolist().index(row["value"])].astype(float)


def add_regret(df: pd.DataFrame, normalize: False):
    best = {}
    ranges = {}
    print("⚙️ Preparing regret", end="\r", flush=True)
    for benchmark in df["bench_prior"].unique():
        best[benchmark] = min(df.loc[df["bench_prior"] == benchmark]["value"])
        ranges[benchmark] = (
            max(df.loc[df["bench_prior"] == benchmark]["value"]) - best[benchmark]
        )

    def calculate_simple_regret(row, normalize: bool = False):
        if normalize:
            return (
                abs(best[row["bench_prior"]] - row["value"]) / ranges[row["bench_prior"]]
            )
        return abs(best[row["bench_prior"]] - row["value"])

    if normalize:
        print("⚙️ Adding regret       ", end="\r", flush=True)
        df["regret"] = df.apply(calculate_simple_regret, axis=1, normalize=True)
        print("✅ Adding regret done                      ")
    else:
        print("⚙️ Adding normalized regret       ", end="\r", flush=True)
        df["norm_regret"] = df.apply(calculate_simple_regret, axis=1, normalize=False)
        print("✅ Adding normalized regret done                      ")
    return df


def rename_algos(row, algo_dict: dict):
    return algo_dict[row["algorithm"]]


def create_incumbent(data, fs, f_space, benchmarks, algos, benchmark, algorithm):
    dataset = pd.DataFrame()
    for n_f, max_f in enumerate(f_space):
        for b_n, bench in enumerate(benchmarks):
            df_at_point = data.loc[
                (data["used_fidelity"] <= max_f)
                & (data[benchmark] == bench)
                & (data[algorithm].isin(algos))
            ]
            for seed in df_at_point["seed"].unique():
                print(
                    f"⚙️ Fidelity {n_f+1}/{len(f_space)}, Benchmark {b_n+1}/{len(benchmarks)}          ",
                    end="\r",
                    flush=True,
                )
                for algo in algos:
                    if (
                        len(
                            df_at_point.loc[
                                (df_at_point["seed"] == seed)
                                & (df_at_point["algorithm"] == algo)
                            ]
                        )
                        > 0
                    ):
                        df_criteria = (
                            df_at_point.loc[
                                (df_at_point["seed"] == seed)
                                & (df_at_point["algorithm"] == algo)
                            ]
                            .iloc[-1]
                            .to_frame()
                            .T
                        )
                        df_criteria["used_fidelity"] = max_f
                        dataset = pd.concat([dataset, df_criteria], ignore_index=True)
        dataset[["value", "used_fidelity"]] = dataset[["value", "used_fidelity"]].astype(
            float
        )
        dataset["seed"] = dataset["seed"].astype(int)
    datasets = {}
    for max_f in fs:
        datasets[max_f] = dataset.loc[dataset["used_fidelity"] <= max_f]
    return datasets


def create_priorband_benchPrior_relRanks_f24():
    algorithm = "algorithm"
    benchmark = "bench_prior"
    time = "used_fidelity"
    algos = [
        "pb_mutation_dynamic_geometric-default-at-target",
        "random_search",
        "hyperband",
    ]
    fs = [24]
    f_space = np.linspace(1, max(fs), max(fs)).tolist()

    benchmarks = [
        "jahs_cifar10",
        "jahs_colorectal_histology",
        "jahs_fashion_mnist",
        "lcbench-126026",
        "lcbench-167190",
        "lcbench-168330",
        "lcbench-168910",
        "lcbench-189906",
        "cifar100_wideresnet_2048",
        "imagenet_resnet_512",
        "lm1b_transformer_2048",
        "translatewmt_xformer_64",
    ]
    label_dict = {
        "random_search": "RS",
        "hyperband": "HB",
        "pb_mutation_dynamic_geometric-default-at-target": "PB",
        "jahs_cifar10": "JAHS-C10",
        "jahs_colorectal_histology": "JAHS-CH",
        "jahs_fashion_mnist": "JAHS-FM",
        "lcbench-126026": "LC-126026",
        "lcbench-167190": "LC-167190",
        "lcbench-168330": "LC-168330",
        "lcbench-168910": "LC-168910",
        "lcbench-189906": "LC-189906",
        "cifar100_wideresnet_2048": "PD1-Cifar100",
        "imagenet_resnet_512": "PD1-ImageNet",
        "lm1b_transformer_2048": "PD1-LM1B",
        "translatewmt_xformer_64": "PD1-WMT",
        "random_search_prior": "RS+Prior",
        "bo": "BO",
        "pibo-no-default": "PiBO",
        "bohb": "BOHB",
        "priorband_bo": "PriorBand+BO",
    }

    data = load_priorband_data()
    data = data.loc[
        (data[algorithm].isin(algos))
        & (data["benchmark"].isin(benchmarks))
        & (data["prior"].isin(["at25", "bad"]))
    ]
    data["bench_prior"] = data.apply(combine_bench_prior, axis=1)
    data.drop(columns=["benchmark", "prior"], inplace=True)
    benchmarks = data[benchmark].unique()
    max_f = max(fs)
    data = create_incumbent(data, fs, f_space, benchmarks, algos, benchmark, algorithm)[
        max_f
    ]
    print(f"⚙️ F {max_f}: Adding relative ranks             ", end="\r", flush=True)
    data["rel_rank"] = data.apply(
        add_rel_ranks, data=data, benchmark=benchmark, time=time, axis=1
    )
    print(f"⚙️ F {max_f}: Renaming algorithms             ", end="\r", flush=True)
    data[algorithm] = data.apply(rename_algos, algo_dict=label_dict, axis=1)
    print("✅ Dataset loaded                   ", end="\r", flush=True)
    return data


def create_piBo_benchPrior_relRanks_f24():
    algorithm = "algorithm"
    benchmark = "bench_prior"
    time = "used_fidelity"
    algos = ["random_search_prior", "priorband_bo", "pibo-no-default", "bo", "bohb"]
    fs = [24]
    f_space = np.linspace(1, max(fs), max(fs)).tolist()

    benchmarks = [
        "jahs_cifar10",
        "jahs_colorectal_histology",
        "jahs_fashion_mnist",
        "lcbench-126026",
        "lcbench-167190",
        "lcbench-168330",
        "lcbench-168910",
        "lcbench-189906",
        "cifar100_wideresnet_2048",
        "imagenet_resnet_512",
        "lm1b_transformer_2048",
        "translatewmt_xformer_64",
    ]
    label_dict = {
        "random_search": "RS",
        "hyperband": "HB",
        "pb_mutation_dynamic_geometric-default-at-target": "PB",
        "jahs_cifar10": "JAHS-C10",
        "jahs_colorectal_histology": "JAHS-CH",
        "jahs_fashion_mnist": "JAHS-FM",
        "lcbench-126026": "LC-126026",
        "lcbench-167190": "LC-167190",
        "lcbench-168330": "LC-168330",
        "lcbench-168910": "LC-168910",
        "lcbench-189906": "LC-189906",
        "cifar100_wideresnet_2048": "PD1-Cifar100",
        "imagenet_resnet_512": "PD1-ImageNet",
        "lm1b_transformer_2048": "PD1-LM1B",
        "translatewmt_xformer_64": "PD1-WMT",
        "random_search_prior": "RS+Prior",
        "bo": "BO",
        "pibo-no-default": "PiBO",
        "bohb": "BOHB",
        "priorband_bo": "PriorBand+BO",
    }

    data = load_priorband_data()
    data = data.loc[
        (data[algorithm].isin(algos))
        & (data["benchmark"].isin(benchmarks))
        & (data["prior"].isin(["at25", "bad"]))
    ]
    data["bench_prior"] = data.apply(combine_bench_prior, axis=1)
    data.drop(columns=["benchmark", "prior"], inplace=True)
    benchmarks = data[benchmark].unique()
    max_f = max(fs)
    data = create_incumbent(data, fs, f_space, benchmarks, algos, benchmark, algorithm)[
        max_f
    ]
    print(f"⚙️ F {max_f}: Adding relative ranks             ", end="\r", flush=True)
    data["rel_rank"] = data.apply(
        add_rel_ranks, data=data, benchmark=benchmark, time=time, axis=1
    )
    print(f"⚙️ F {max_f}: Renaming algorithms             ", end="\r", flush=True)
    data[algorithm] = data.apply(rename_algos, algo_dict=label_dict, axis=1)
    print("✅ Dataset loaded                   ", end="\r", flush=True)
    data.to_parquet("pibo_benchPrior_relRanks_f24_meta.parquet")
    return data


def add_benchmark_metafeatures(data: pd.DataFrame):
    meta_feature_df = pd.read_csv("benchmark_metafeatures.csv")

    def add_meta_features(row):
        return meta_feature_df.loc[
            meta_feature_df["code_name"] == row["bench_prior"].rsplit("_", 1)[0]
        ][
            ["# Vars", "# cont. Vars", "# cond. Vars", "# cat. Vars", "log int", "int"]
        ].values[
            0
        ]

    data[
        ["n_Vars", "n_cont_Vars", "n_cond_Vars", "n_cat_Vars", "log_int", "int"]
    ] = data.apply(add_meta_features, axis=1).to_list()
    return data


def glrt(
    mod1, mod2, names: list[str] = None, returns: bool = False
) -> dict[str, typing.Any]:
    """Generalized Likelihood Ratio Test on two Liner Mixed Effect Models from R

    Args:
        mod1 (Lmer): First, simple model, Null-Hypothesis assumes that this model contains not significantly less information as the second model
        mod2 (Lmer): Second model, Alternative Hypothesis assumes that this model contains significant new information

    Returns:
        dict[str,typing.Any]: Result dictionary with Chi-Square-Score, Degrees of Freedom and p-value of the test
    """
    assert (
        mod1.logLike
        and mod2.logLike
        and mod1.coefs is not None
        and mod2.coefs is not None
    )
    chi_square = 2 * abs(mod1.logLike - mod2.logLike)
    delta_params = abs(len(mod1.coefs) - len(mod2.coefs))
    p = 1 - stats.chi2.cdf(chi_square, df=delta_params)
    if names:
        print(
            f"{names[0]} ({round(mod1.logLike,2)}) {'==' if p>0.05 else '>>' if mod1.logLike>mod2.logLike else '<<' if mod1.logLike<mod2.logLike else '=='} {names[1]} ({round(mod2.logLike,2)})"
        )
        print(f"Chi-Square: {chi_square}, P-Value: {p}")
    if returns:
        return {
            "p": p,
            "chi_square": chi_square,
            "df": delta_params,
        }


def model(
    formula: str,
    data: pd.DataFrame,
    system_id: str = "algorithm",
    factor: str = None,
    factor_list: list[str] = None,
):
    if not "|" in formula:
        data["dummy"] = "0"
        data.at[data.index[0], "dummy"] = "1"
        formula += "+(1|dummy)"
    model = Lmer(
        formula=formula,
        data=data,
    )
    factors = {system_id: list(data[system_id].unique())}
    if factor:
        factors[factor] = list(data[factor].unique())
    if factor_list:
        for factor in factor_list:
            factors[factor] = list(data[factor].unique())
    model.fit(
        factors=factors,
        REML=False,
        summarize=False,
        verbose=False,
    )

    # model = Lm(
    #     formula=formula,
    #     data=data,
    # )
    # model.fit(verbose=False, summarize=False)
    return model


def create_cd_cluster(
    result_cluster, show: bool = True
):  #:list[list[(pd.DataFrame,pd.DataFrame)]],x_axis:list[str],y_axis:list[str]):
    x_axis = list(result_cluster.keys())

    color_dict = {
        "random_search": "red",
        "hyperband": "green",
        "pb_mutation_dynamic_geometric-default-at-target": "blue",
        "priorband": "blue",
        "RS": "red",
        "HB": "green",
        "PB": "blue",
        "RS+Prior": "pink",
        "BO": "gold",
        "PiBO": "brown",
        "BOHB": "green",
        "PriorBand+BO": "purple",
    }

    fig, axes = plt.subplots(
        len(result_cluster.values()),
        len(list(result_cluster.values())[0].values()),
        figsize=(
            10 * len(list(result_cluster.values())[0].values()) / 3,
            10 * len(result_cluster.values()) / 3,
        ),
    )

    for list_n, (list_k, list_e) in enumerate(result_cluster.items()):
        y_axis = list(result_cluster[list_k].keys())
        for cell_n, cell in enumerate(list_e.values()):
            scoreframe = cell[0].sort_values(by=["Estimate"])[["algorithm", "Estimate"]]
            contrasts = cell[1]
            for pair in contrasts["Contrast"]:
                contrasts.loc[contrasts["Contrast"] == pair, "algorithm_1"] = pair.split(
                    " - "
                )[0]
                contrasts.loc[contrasts["Contrast"] == pair, "algorithm_2"] = pair.split(
                    " - "
                )[1]
            contrasts = contrasts.drop("Contrast", axis=1)
            column = contrasts.pop("algorithm_2")
            contrasts.insert(0, "algorithm_2", column)
            column = contrasts.pop("algorithm_1")
            contrasts.insert(0, "algorithm_1", column)
            contrastframe = contrasts[["Sig", "algorithm_1", "algorithm_2"]]

            min_score = scoreframe["Estimate"][0]
            max_score = scoreframe["Estimate"][len(scoreframe["Estimate"]) - 1]

            significance_lines = []
            for n_best_algo, best_algo in enumerate(scoreframe["algorithm"]):
                for n_worst_algo, worst_algo in reversed(
                    list(enumerate(scoreframe["algorithm"][n_best_algo + 1 :]))
                ):
                    if "+" in worst_algo and not "(" in worst_algo:
                        worst_algo = f"({worst_algo})"
                    if "+" in best_algo and not "(" in best_algo:
                        best_algo = f"({best_algo})"
                    significance = contrastframe.loc[
                        (
                            (contrastframe["algorithm_1"] == best_algo)
                            & (contrastframe["algorithm_2"] == worst_algo)
                        )
                        | (
                            (contrastframe["algorithm_2"] == best_algo)
                            & (contrastframe["algorithm_1"] == worst_algo)
                        )
                    ]["Sig"][0]
                    if significance in ["", "."]:
                        new_line = [n_best_algo, n_best_algo + n_worst_algo + 1]
                        if not any(
                            existing_pair[0] <= new_line[0] <= existing_pair[1]
                            and existing_pair[0] <= new_line[1] <= existing_pair[1]
                            for existing_pair in significance_lines
                        ):
                            significance_lines.append(
                                [n_best_algo, n_best_algo + n_worst_algo + 1]
                            )
                            break

            n_sign_lines = len(significance_lines)
            plot_height = 10 + 5

            if len(result_cluster) > 1:
                ax = axes[list_n, cell_n]
            elif len(list_e) > 1:
                ax = axes[cell_n]
            else:
                ax = axes
            ax.set_title(f"{x_axis[list_n]} x {y_axis[cell_n]}", pad=0, y=-0.18)
            ax.get_yaxis().set_visible(False)
            ax.set_xlim(
                min_score - (max_score - min_score) * 0.1,
                max_score + (max_score - min_score) * 0.1,
            )
            ax.set_ylim(0, plot_height)
            ax.spines["bottom"].set_position(("data", 0.0))
            ax.spines[["top", "left", "right"]].set_visible(False)
            ax.invert_xaxis()
            texts = []
            for algo in range(len(scoreframe["Estimate"])):
                ax.plot(
                    [scoreframe["Estimate"][algo], scoreframe["Estimate"][algo]],
                    [0, plot_height - 6],
                    "-",
                    lw=2,
                    label="_not in legend",
                    color=color_dict[scoreframe["algorithm"][algo]],
                )
                texts.append(
                    ax.text(
                        # Names
                        scoreframe["Estimate"][algo],
                        plot_height - 3,
                        scoreframe["algorithm"][algo],
                        horizontalalignment="left",
                        verticalalignment="baseline",
                        rotation=45,
                        rotation_mode="anchor",
                    )
                )
                texts.append(
                    ax.text(
                        # Scores
                        scoreframe["Estimate"][algo],
                        plot_height - 4.5,
                        np.round(scoreframe["Estimate"][algo], 2),
                        horizontalalignment="left",
                        verticalalignment="baseline",
                        rotation=45,
                        rotation_mode="anchor",
                    )
                )
            for n_line, line in enumerate(significance_lines):
                if scoreframe["Estimate"][line[0]] == scoreframe["Estimate"][line[1]]:
                    ax.plot(
                        [
                            scoreframe["Estimate"][line[0]],
                            scoreframe["Estimate"][line[1]],
                        ],
                        [
                            10 / (n_sign_lines + 1) * (n_line + 1) - 0.5,
                            10 / (n_sign_lines + 1) * (n_line + 1) + 0.5,
                        ],
                        "-",
                        lw=2,
                        label="_not in legend",
                        color="gray",
                    )
                else:
                    ax.plot(
                        [
                            scoreframe["Estimate"][line[0]],
                            scoreframe["Estimate"][line[1]],
                        ],
                        [
                            10 / (n_sign_lines + 1) * (n_line + 1),
                            10 / (n_sign_lines + 1) * (n_line + 1),
                        ],
                        "-",
                        lw=6,
                        label="_not in legend",
                        color="gray",
                    )
    if show:
        plt.show()
    return fig, axes
