# Significance Analysis

[![PyPI version](https://img.shields.io/pypi/v/neural-pipeline-search?color=informational)](https://pypi.org/project/neural-pipeline-search/)
[![Python versions](https://img.shields.io/pypi/pyversions/neural-pipeline-search)](https://pypi.org/project/neural-pipeline-search/)
[![License](https://img.shields.io/pypi/l/neural-pipeline-search?color=informational)](LICENSE)

This package is used to analyse datasets of different HPO-algorithms performing on multiple benchmarks.

## Note

As indicated with the `v0.x.x` version number, Significance Analysis is early stage code and APIs might change in the future.

## Documentation

Please have a look at our documentation (Missing link: https://automl.github.io/) and [example](sign_analysis_example).

## Installation

Using pip

```bash
pip install $$$
```

Using R, >=4.0.0
install packages: Matrix, emmeans, lmerTest

## Usage

1. Generate data from HPO-algorithms on benchmarks, saving data according to our format (Missing link: https://automl.github.io/)
1. Call function `checkSignificance` on dataset, while specifying variable-names

In code, the usage pattern can look like this:

```python
from signficance_analysis import checkSignificance

# 1. Generate/import dataset
data = pd.read_pickle("./exampleDataset.pkl")

# 2. Analyse dataset
checkSignificance(data, "mean", "surrogate_aquisition", "benchark")
```

For more details and features please have a look at our documentation (Missing link: https://automl.github.io/) and [example](sign_analysis_example).
