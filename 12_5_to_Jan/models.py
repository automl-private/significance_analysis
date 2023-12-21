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

pd.set_option("chained_assignment", None)
pd.set_option("display.max_rows", 5000)
pd.set_option("display.max_columns", 5000)
pd.set_option("display.width", 10000)


def load_priorband_data(combine_bench_prior: bool = True):
    df = pd.read_parquet("priorband_single.parquet")
    df = df.reset_index()
    df_collection = []
    for seed_nr in range(50):
        partial_df = df[["benchmark", "prior", "algorithm", "used_fidelity"]]
        partial_df["value"] = df[f"seed-{seed_nr}"]
        partial_df["seed"] = seed_nr
        df_collection.append(partial_df)
        print(f"⚙️ Seed {seed_nr+1}/50", end="\r", flush=True)
    complete_df = pd.concat(df_collection, ignore_index=True)

    if combine_bench_prior:

        def combine_bench_priors(row):
            return f"{row['benchmark']}_{row['prior']}"

        complete_df["bench_prior"] = complete_df.apply(combine_bench_priors, axis=1)
    print("✅ Loading data done")
    return complete_df


def glrt(mod1: Lmer, mod2: Lmer) -> dict[str, typing.Any]:
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
    return {
        "chi_square": chi_square,
        "df": delta_params,
        "p": 1 - stats.chi2.cdf(chi_square, df=delta_params),
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
