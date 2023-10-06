import pandas as pd

from significance_analysis import conduct_analysis

if __name__ == "__main__":

    # Load example dataset
    data = pd.read_csv("./significance_analysis_example/example_dataset.csv")

    # First Analysis: Analyse performance of acquisition functions over all benchmarks and trainingrounds
    conduct_analysis(data, "mean", "acquisition", "benchmark")

    # Second Analysis: Analyse performance of acquisition functions over all benchmarks binned by trainingrounds,
    # i.e. performance in the first 8, the next 12, the next 15, the next 10 and the final 5 rounds.
    # Does not print the results, only shows plots.
    conduct_analysis(
        data,
        "mean",
        "acquisition",
        "benchmark",
        bin_id="budget",
        bins=[9, 20, 35, 45],
        verbosity=False,
    )

    # Third Analysis: Analyse performance of acquisition functions on each benchmark seperately over all trainingrounds
    # without showing plots.
    conduct_analysis(
        data,
        "mean",
        "acquisition",
        "benchmark",
        subset=("benchmark", "all"),
        show_plots=False,
    )
