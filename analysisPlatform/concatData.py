import os

import pandas as pd

numberOfAquFunctions = 2
numberOfSurrFunctions = 2
numberOfBenchmarks = 2

dfList = []
filesList = os.listdir("./experimentPlatform/Results")
print(filesList)

for file in filesList:
    dfList.append(pd.read_pickle(file))

df = pd.concat(dfList)

df.to_pickle("./concatData.pkl")
