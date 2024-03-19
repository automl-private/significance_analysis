"""import os
import pickle
import sys

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tabulate
from autorank import autorank, create_report, plot_stats
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator"""
# import itertools
import os
import typing

import numpy as np
import pandas as pd
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
        print(f"{f'⚙️ Seed {seed_nr+1}/50':<100}", end="\r", flush=True)
    complete_df = pd.concat(df_collection, ignore_index=True)
    print(f"{f'✅ Loading data done':<100}")
    return complete_df


def combine_bench_prior(row):
    return f"{row['benchmark']}_{row['prior']}"


def add_rel_ranks(row, data: pd.DataFrame, benchmark: str, time: str):
    return rankdata(
        data.loc[
            (data[benchmark] == row[benchmark])
            & (data["seed"] == row["seed"])
            & (data[time] == row[time])
        ]["value"].values
    )[
        data.loc[
            (data[benchmark] == row[benchmark])
            & (data["seed"] == row["seed"])
            & (data[time] == row[time])
        ]["value"]
        .values.tolist()
        .index(row["value"])
    ].astype(
        float
    )


def add_regret(df: pd.DataFrame, benchmark_variable, normalize: False):
    best = {}
    ranges = {}
    print("⚙️ Preparing regret", end="\r", flush=True)
    for benchmark in df[benchmark_variable].unique():
        best[benchmark] = min(df.loc[df[benchmark_variable] == benchmark]["value"])
        ranges[benchmark] = (
            max(df.loc[df[benchmark_variable] == benchmark]["value"]) - best[benchmark]
        )

    def calculate_simple_regret(row, normalize: bool = False):
        if normalize:
            return (
                abs(best[row[benchmark_variable]] - row["value"])
                / ranges[row[benchmark_variable]]
            )
        return abs(best[row[benchmark_variable]] - row["value"])

    if normalize:
        print(f"⚙️ {'Adding regret':<100}", end="\r", flush=True)
        df["regret"] = df.apply(calculate_simple_regret, axis=1, normalize=True)
        print(f"{'✅ Adding regret done':<100}")
    else:
        print(f"{'⚙️ Adding normalized regret':<100}", end="\r", flush=True)
        df["norm_regret"] = df.apply(calculate_simple_regret, axis=1, normalize=False)
        print(f"{'✅ Adding normalized regret done':<100}")
    return df


def rename_algos(row, algo_dict: dict):
    return algo_dict[row["algorithm"]]


def rename_benchmarks(row, bench_dict: dict):
    return bench_dict[row["benchmark"]]


def create_incumbent(
    data, f_space, benchmarks, algos, benchmark_variable, algorithm_variable
):
    dataset = pd.DataFrame()
    for n_f, max_f in enumerate(f_space):
        for b_n, bench in enumerate(benchmarks):
            df_at_point = data.loc[
                (data["used_fidelity"] <= max_f)
                & (data[benchmark_variable] == bench)
                & (data[algorithm_variable].isin(algos))
            ]
            for seed in df_at_point["seed"].unique():
                print(
                    f"{f'⚙️ Fidelity {n_f+1}/{len(f_space)}, Benchmark {b_n+1}/{len(benchmarks)}':<100}",
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
    dataset = dataset.loc[dataset["used_fidelity"] <= max(f_space)]
    return dataset


std_benchmarks = [
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
    "pb_mutation_dynamic_geometric_bo-default-at-target": "PriorBand+BO",
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
    "bo-10": "BO",
    "pibo-no-default": "PiBO",
    "pibo-default-first-10": "PiBO",
    "bohb": "BOHB",
    "priorband_bo": "PriorBand+BO",
}

figures = {}
figures["fig7"] = [
    "random_search_prior",
    "pb_mutation_dynamic_geometric_bo-default-at-target",
    "pibo-default-first-10",
    "bo-10",
    "bohb",
]
figures["fig5"] = [
    "pb_mutation_dynamic_geometric-default-at-target",
    "random_search",
    "hyperband",
]


def get_dataset(
    dataset_name: str,
    algos: typing.Union[list[str], str] = None,
    f_range: list[float] = (1, 24),
    f_steps: int = None,
    priors: list[str] = None,
    benchmarks: list[str] = None,
    rel_ranks: bool = False,
):
    if not priors:
        priors = ["at25", "bad"]
    if not benchmarks:
        benchmarks = std_benchmarks
    if os.path.exists(f"datasets/{dataset_name}.parquet"):
        data = pd.read_parquet(f"datasets/{dataset_name}.parquet")
        return data

    algos = figures[algos] if isinstance(algos, str) else algos
    algorithm_variable = "algorithm"
    benchmark_variable = "bench_prior"
    time_variable = "used_fidelity"

    f_space = (
        np.linspace(
            f_range[0], f_range[1], f_steps if f_steps else f_range[1] - f_range[0] + 1
        )
        .round(2)
        .tolist()
    )

    data = load_priorband_data()
    data = data.loc[
        (data[algorithm_variable].isin(algos))
        & (data["benchmark"].isin(benchmarks))
        & (data["prior"].isin(priors))
    ]
    data["benchmark"] = data.apply(rename_benchmarks, bench_dict=label_dict, axis=1)
    data["bench_prior"] = data.apply(combine_bench_prior, axis=1)
    data = add_regret(data, benchmark_variable=benchmark_variable, normalize=True)
    data = add_regret(data, benchmark_variable=benchmark_variable, normalize=False)
    benchmarks_split = data[benchmark_variable].unique()

    data = create_incumbent(
        data, f_space, benchmarks_split, algos, benchmark_variable, algorithm_variable
    )
    if rel_ranks:
        print(f"{'⚙️ Adding relative ranks':<100}", end="\r", flush=True)
        data["rel_rank"] = data.apply(
            add_rel_ranks,
            data=data,
            benchmark=benchmark_variable,
            time=time_variable,
            axis=1,
        )
    print(f"{'⚙️ Renaming algorithms':<100}", end="\r", flush=True)
    data[algorithm_variable] = data.apply(rename_algos, algo_dict=label_dict, axis=1)
    print(f"{'✅ Dataset loaded':<100}", end="\r", flush=True)
    data.to_parquet(f"datasets/{dataset_name}.parquet")
    return data


def convert_to_autorank(
    data: pd.DataFrame,
    algorithm_variable: str = "algorithm",
    value_variable: str = "value",
    budget_variable: str = "used_fidelity",
    min_f=1,
    max_f=24,
):

    df_autorank = pd.DataFrame()
    for algo in data[algorithm_variable].unique():
        df_autorank[algo] = -data[
            (data[algorithm_variable] == algo)
            & (data[budget_variable] <= max_f)
            & (data[budget_variable] >= min_f)
        ][value_variable].reset_index(drop=True)
    return df_autorank
