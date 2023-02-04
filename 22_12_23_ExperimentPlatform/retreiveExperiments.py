import pandas as pd

numberOfAquFunctions = 2
numberOfSurrFunctions = 2
numberOfBenchmarks = 2

dfList = []

for aqu in range(numberOfAquFunctions):
    for surr in range(numberOfSurrFunctions):
        for ben in range(numberOfBenchmarks):
            print("./results/b" + str(ben) + "a" + str(aqu) + "s" + str(surr) + ".pkl")
            dfList.append(
                pd.read_pickle(
                    "C:/Users/Amega/OneDrive/Desktop/Git/bachelorproject_online/significance_analysis/22_12_23_ExperimentPlatform/results"
                    + "/b"
                    + str(ben)
                    + "a"
                    + str(aqu)
                    + "s"
                    + str(surr)
                    + ".pkl"
                )
            )

df = pd.concat(dfList)
# print(df)

df.to_pickle("./concatData.pkl")
