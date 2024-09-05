import typing

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pymer4.models import Lm, Lmer
from scipy import stats

ALGORITHM = "algorithm"
VALUE = "value"
SEED = "seed"
BUDGET = "used_fidelity"
BENCHMARK = "benchmark"


def dataframe_validator(
    data: pd.DataFrame,
    algorithm_var: str = ALGORITHM,
    benchmark_var: str = BENCHMARK,
    loss_var: str = VALUE,
    fidelity_var: str = BUDGET,
    seed_var: str = SEED,
    **extra_vars,
) -> typing.Tuple[pd.DataFrame, list[str]]:
    """
    Validates the columns of a pandas DataFrame and converts them to the appropriate data types if necessary.

    Args:
        data (pd.DataFrame): The input DataFrame to be validated.

    Returns:
        tuple: A tuple containing the validated DataFrame and a list of valid column names.

    Raises:
        None

    Examples:
        >>> df = pd.DataFrame({'ALGORITHM': ['A', 'B', 'C'],
        ...                    'SEED': [1, 2, 3],
        ...                    'BENCHMARK': ['X', 'Y', 'Z'],
        ...                    'VALUE': [0.1, 0.2, 0.3],
        ...                    'BUDGET': [100, 200, 300]})
        >>> dataframe_validator(df)
        (  ALGORITHM  SEED BENCHMARK  VALUE  BUDGET
        0         A     1         X    0.1   100.0
        1         B     2         Y    0.2   200.0
        2         C     3         Z    0.3   300.0,
        ['ALGORITHM', 'SEED', 'BENCHMARK', 'VALUE', 'BUDGET'])
    """
    cols = data.dtypes
    valid_columns = []
    for col in [algorithm_var, seed_var, benchmark_var]:
        if col in data.columns:
            if cols[col] != "object":
                print(f"Column {col} is not of type object")
                try:
                    data[col] = data[col].astype(object)
                    valid_columns.append(col)
                except Exception as e:
                    print(f"Error {e}: Could not convert all values of {col} to object.")
            else:
                valid_columns.append(col)
    for col in [loss_var, fidelity_var]:
        if col in data.columns:
            if not cols[col] in ["float", "int"]:
                print(f"Column {col} is not numeric")
                try:
                    data[col] = data[col].astype(np.float64)
                    valid_columns.append(col)
                except Exception as e:
                    print(f"Error {e}: Could not convert all values of {col} to float.")
            else:
                valid_columns.append(col)
    for _kw, col in extra_vars.items():
        if is_numeric_dtype(data[col]):
            data[col] = data[col].astype(np.float64)
        else:
            data[col] = data[col].astype(object)
        valid_columns.append(col)
    return data, valid_columns


def glrt(
    mod1, mod2, names: list[str] = None, returns: bool = False
) -> dict[str, typing.Any]:
    """Generalized Likelihood Ratio Test on two Liner Mixed Effect Models from R

    Args:
        mod1 (Lmer): First, simple model, Null-Hypothesis assumes that this model contains not significantly less information as the second model
        mod2 (Lmer): Second model, Alternative Hypothesis assumes that this model contains significant new information
        names (list[str], optional): Names of the models for the output. Defaults to None.
        returns (bool, optional): If True, the function returns a dictionary with the Chi-Square-Score, Degrees of Freedom and p-value of the test. Defaults to False.

    Returns:
        dict[str,typing.Any]: Result dictionary with Chi-Square-Score, Degrees of Freedom and p-value of the test

    Raises:
        None

    Examples:
        >>> mod1 = model("value ~ algorithm + (1|seed)", data)
        >>> mod2 = model("value ~ algorithm + (1|seed) + (1|benchmark)", data)
        >>> glrt(mod1,mod2,returns=True)
        {'p': 0.0, 'chi_square': 7.0, 'df': 1}
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
            f"{names[0]} ({round(mod1.logLike,2)}) {'==' if p>0.05 or mod1.logLike==mod2.logLike else '>>' if mod1.logLike>mod2.logLike else '<<'} {names[1]} ({round(mod2.logLike,2)})"
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
    factor: typing.Union[str, list[str]] = None,
    dummy=True,
    no_warnings=True,
) -> typing.Union[Lm, Lmer]:
    """
    Model object for Linear (Mixed Effects) Model-based significance analysis.

    Args:
        formula (str): The formula specifying the regression model.
        data (pd.DataFrame): The input data for the regression model.
        system_id (str, optional): The column name in the data representing the system ID. Defaults to "algorithm".
        factor (str or list[str], optional): The column name(s) in the data representing the factor(s) to include in the model. Defaults to None.
        dummy (bool, optional): Whether to include a dummy variable in the model to enforce the use of an LMEM. Defaults to True.
        no_warnings (bool, optional): Whether to suppress warnings during model fitting. Defaults to True.

    Returns:
        Union[Lm, Lmer]: The fitted regression model.

    Raises:
        None

    Examples:
        # Example 1: Perform linear regression
        data = pd.DataFrame({'x': [1, 2, 3], 'y': [2, 4, 6]})
        model('y ~ x', data)

        # Example 2: Perform mixed-effects regression
        data = pd.DataFrame({'x': [1, 2, 3], 'y': [2, 4, 6], 'group': ['A', 'B', 'A']})
        model('y ~ x + (1|group)', data)
    """

    if not "|" in formula:
        if dummy:
            data["dummy"] = "0"
            data.at[data.index[0], "dummy"] = "1"
            formula += "+(1|dummy)"
        else:
            mod = Lm(formula, data)
            mod.fit(cluster=data[system_id], verbose=False, summarize=False)
            return mod
    model = Lmer(
        formula=formula,
        data=data,
    )
    factors = {system_id: list(data[system_id].unique())}
    if factor:
        if isinstance(factor, str):
            factors[factor] = list(data[factor].unique())
        else:
            for f in factor:
                factors[f] = list(data[f].unique())
    model.fit(
        factors=factors,
        REML=False,
        summarize=False,
        verbose=False,
        no_warnings=no_warnings,
    )
    return model


def benchmark_clustering(
    dataset: pd.DataFrame,
    algorithms: typing.Tuple[str, str],
    metafeature_var: str,
    algorithm_var: str = ALGORITHM,
    benchmark_var: str = BENCHMARK,
    loss_var: str = VALUE,
    fidelity_var: str = BUDGET,
    path: str = None,
):
    dataset["benchmark_variant"] = dataset.apply(
        lambda x: f"{x[benchmark_var]} x {x[metafeature_var]}", axis=1
    )
    dataset = dataset.loc[dataset[algorithm_var].isin(algorithms)]
    dataset, cols = dataframe_validator(dataset, metafeature=metafeature_var)
    if not all(
        [
            x in cols
            for x in [
                algorithm_var,
                benchmark_var,
                metafeature_var,
                loss_var,
                fidelity_var,
            ]
        ]
    ):
        raise ValueError("Not all necessary columns are included in the dataset")
    benchmark = "benchmark_variant"
    wins_bench = pd.DataFrame()
    full_wins = []
    full_benchmarks = []
    full_fidelities = []
    for f_n, f in enumerate(dataset[fidelity_var].unique()):
        print(
            f"{f:<{max([str(x) for x in dataset[fidelity_var].unique()],key=len)}} ({f_n+1}/{len(dataset[fidelity_var].unique())})",
            end="\r",
            flush=True,
        )
        wins_budget = []
        for bench in dataset[benchmark].unique():
            if (
                len(
                    dataset.loc[
                        (dataset[fidelity_var] == f) & (dataset[benchmark] == bench)
                    ]
                )
                == 0
            ):
                continue
            full_fidelities.append(f)
            mod = model(
                f"{loss_var}~{algorithm_var}",
                dataset.loc[(dataset[fidelity_var] == f) & (dataset[benchmark] == bench)],
                algorithm_var,
            )
            post_hocs = mod.post_hoc(algorithm_var)
            if post_hocs[1].Sig[0] in ["***", "**", "*"]:
                wins_budget.append(
                    -1
                    if post_hocs[1]
                    .Contrast[0]
                    .rsplit(" - ")[0 if post_hocs[1].Estimate[0] < 0 else 1]
                    == algorithms[0]
                    else 1
                )
                full_wins.append(
                    -1
                    if post_hocs[1]
                    .Contrast[0]
                    .rsplit(" - ")[0 if post_hocs[1].Estimate[0] < 0 else 1]
                    == algorithms[0]
                    else 1
                )
            else:
                wins_budget.append(0)
                full_wins.append(0)
            full_benchmarks.append(bench)
    wins_bench[benchmark] = full_benchmarks
    wins_bench["wins"] = full_wins
    wins_bench["fidelity"] = full_fidelities
    wins_bench["wins"] = wins_bench["wins"].astype(float)
    wins_bench[benchmark] = wins_bench[benchmark].astype(str)
    wins_bench[["benchmark", "metafeature"]] = wins_bench.apply(
        lambda x: pd.Series(x[benchmark].rsplit(" x ", 1)), axis=1
    )
    if path:
        path = path if path.endswith(".parquet") else path + ".parquet"
        wins_bench.to_parquet(path)
    return wins_bench
