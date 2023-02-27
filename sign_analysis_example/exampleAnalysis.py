import pandas as pd

from sign_analysis_example.src.signAnalysisFunction import checkSignificance

if __name__ == "__main__":
    data = pd.read_pickle("./exampleDataset.pkl")
    metric = "mean"
    system_id = "surrogate_aquisition"
    input_id = "benchmark"
    bin_id = "budget"
    bin_labels = ["short", "long"]
    bin_dividers = [0.4, 1]
    checkSignificance(data, metric, system_id, input_id, bin_id, bin_labels, bin_dividers)
