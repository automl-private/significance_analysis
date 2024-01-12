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

import numpy as np
import pandas as pd
from pymer4.models import Lm, Lmer
from scipy import stats
from scipy.stats import rankdata

pd.set_option("chained_assignment", None)
pd.set_option("display.max_rows", 5000)
pd.set_option("display.max_columns", 5000)
pd.set_option("display.width", 10000)


def load_priorband_data():
    df = pd.read_parquet("priorband_single.parquet")
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
        "pibo": "PiBO",
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


def model(formula: str, data: pd.DataFrame, system_id: str = "algorithm"):
    if "|" in formula:
        model = Lmer(
            formula=formula,
            data=data,
        )

        model.fit(
            factors={system_id: list(data[system_id].unique())},
            REML=False,
            summarize=False,
        )
    else:
        model = Lm(
            formula=formula,
            data=data,
        )
        model.fit(verbose=False, summarize=False)
    return model
