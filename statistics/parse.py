import numpy as np
import pandas as pd
import json

f = open("models_accuracies.txt", "r")
data = f.readlines()
data = json.loads(data[0])

ranking = {"name": [], "acc": []}

for key in data.keys():
    data[key] = (np.mean(data[key]), np.std(data[key]))

for (key, value) in sorted(data.items(), key=lambda x: x[1][0], reverse=True):
    if "Oracle" not in key:
        print("%s: %.4f +- %.4f" % (key, value[0], value[1]))
        ranking["name"].append(key)
        ranking["acc"].append("%.4f (+- %.4f)" % (value[0], value[1]))

pd.DataFrame(data=ranking).to_csv("ranking.csv", index=False)
