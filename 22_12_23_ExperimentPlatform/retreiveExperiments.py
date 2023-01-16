import pandas as pd

numberOfAquFunctions = 2
numberOfSurrFunctions = 1
numberOfBenchmarks = 2

dfList = []

for aqu in range(numberOfAquFunctions):
    for surr in range(numberOfSurrFunctions):
        for ben in range(numberOfBenchmarks):
            dfList.append(
                pd.read_pickle(
                    "./results/b" + str(ben) + "a" + str(aqu) + "s" + str(surr) + ".pkl"
                )
            )

df = pd.concat(dfList)
# print(df)

df.to_pickle("./concatData.pkl")
