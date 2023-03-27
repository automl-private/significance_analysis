import pandas as pd

from significance_analysis import conduct_analysis

if __name__ == "__main__":
    data = pd.read_csv("sign_analysis_example/example_dataset.csv")
    bin_labels = ["0-16%", "16-36%", "37-57%", "58-79%", "79-100%"]
    bin_dividers = [0.16, 0.36, 0.57, 0.79]
    conduct_analysis(
        data, "mean", "algorithm", "benchmark", "budget", bin_labels, bin_dividers
    )
