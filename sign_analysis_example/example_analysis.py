import pandas as pd

from significance_analysis import checkSignificance

if __name__ == "__main__":
    data = pd.read_pickle("sign_analysis_example/example_dataset.pkl")
    metric = "mean"
    system_id = "algorithm"
    input_id = "benchmark"
    bin_id = "budget"
    bin_labels = ["short", "long"]
    bin_dividers = [0.4, 1]
    checkSignificance(data, metric, system_id, input_id, bin_id, bin_labels, bin_dividers)
