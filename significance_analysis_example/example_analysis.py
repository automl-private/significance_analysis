import pandas as pd

from significance_analysis import conduct_analysis

if __name__ == "__main__":

    data = pd.read_csv("./significance_analysis_example/example_dataset.csv")
    conduct_analysis(
        data,
        "mean",
        "acquisition",
        "benchmark",
        bin_id="acqu_class",
        bins=[["Analytical", "RandomSearch"], ["MonteCarlo"]],
    )
