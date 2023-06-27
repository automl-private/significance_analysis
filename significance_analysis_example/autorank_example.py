import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from autorank import autorank, plot_stats

np.random.seed(42)
pd.set_option("display.max_columns", 10)
std = 0.3
means = [0.2, 0.3, 0.5, 0.8, 0.85, 0.9]
sample_size = 5
data = pd.DataFrame()
for i, mean in enumerate(means):
    data["pop_%i" % i] = np.random.normal(mean, std, sample_size)  # .clip(0, 1)
# print(data)
res = autorank(data, alpha=0.05, verbose=True)
print(res)
plot_stats(res)
plt.show()
while True:
    a = 2
# print(res)
# create_report(res)
