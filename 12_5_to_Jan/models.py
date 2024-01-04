"""import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import tabulate
from autorank import autorank, create_report, plot_stats
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator"""
import typing

import pandas as pd
from pymer4.models import Lmer
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


def add_rel_ranks(row, data: pd.DataFrame, benchmark: str):
    values = data.loc[
        (data[benchmark] == row[benchmark]) & (data["seed"] == row["seed"])
    ]["value"].values
    ranked_data = rankdata(values)
    return ranked_data[values.tolist().index(row["value"])].astype(float)


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


def glrt(mod1: Lmer, mod2: Lmer, names: list[str] = None) -> dict[str, typing.Any]:
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
    if names:
        print(
            f"{names[0]} ({round(mod1.logLike,2)}) {'>>' if mod1.logLike>mod2.logLike else '<<' if mod1.logLike<mod2.logLike else '=='} {names[1]} ({round(mod2.logLike,2)})"
        )
        print(
            f"Chi-Square: {chi_square}, P-Value: {1 - stats.chi2.cdf(chi_square, df=delta_params)}"
        )
    return {
        "p": 1 - stats.chi2.cdf(chi_square, df=delta_params),
        "chi_square": chi_square,
        "df": delta_params,
    }


def model(formula: str, system_id: str, data: pd.DataFrame):
    model = Lmer(
        formula=formula,
        data=data,
    )

    model.fit(
        factors={system_id: list(data[system_id].unique())},
        REML=False,
        summarize=False,
    )
    return model
