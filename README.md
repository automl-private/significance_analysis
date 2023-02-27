# Significance Analysis

[![PyPI version](https://img.shields.io/pypi/v/neural-pipeline-search?color=informational)](https://pypi.org/project/neural-pipeline-search/)
[![Python versions](https://img.shields.io/pypi/pyversions/neural-pipeline-search)](https://pypi.org/project/neural-pipeline-search/)
[![License](https://img.shields.io/pypi/l/neural-pipeline-search?color=informational)](LICENSE)

This package is used to analyse datasets of different HPO-algorithms performing on multiple benchmarks.

## Note

As indicated with the `v0.x.x` version number, Significance Analysis is early stage code and APIs might change in the future.

## Documentation

Please have a look at our [example](sign_analysis_example).
The dataset should have the following format:

| system_id<br>(algorithm name) | input_id<br>(benchmark name) | metric<br>(mean/estimate) | optional: bin_id<br>(budget/traininground) |
| ----------------------------- | ---------------------------- | ------------------------- | ------------------------------------------ |
| Algorithm1                    | Benchmark1                   | x.xxx                     | 1                                          |
| Algorithm1                    | Benchmark1                   | x.xxx                     | 2                                          |
| Algorithm1                    | Benchmark2                   | x.xxx                     | 1                                          |
| ...                           | ...                          | ...                       | ...                                        |
| Algorithm2                    | Benchmark2                   | x..xxx                    | 2                                          |

In this dataset, there are two different algorithms, trained on two benchmarks for two iterations each. The variable-names (system_id, input_id...) can be customized, but have to be consistent throughout the dataset, i.e. not "mean" for one benchmark and "estimate" for another. The `Significance Analysis` function is then called with the dataset and the variable-names as parameters.
Optionally the dataset can be binned according to a fourth variable (bin_id) and the analysis is conducted on each of the bins seperately, as shown in the code example above. To do this, provide the name of the bin_id-variable, the bin intervals and the labels for thems.

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
