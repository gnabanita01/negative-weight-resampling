import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load datasets
virtual = pd.read_csv("virtual_events.csv")
real = pd.read_csv("real_events.csv")
#Born Projection
real["pt"] = real["pt_real"] + real["z_gluon"]
real["y"] = real["y_real"]

#Keep only needed columns
real = real[["pt", "y", "weight"]]
virtual = virtual[["pt", "y", "weight"]]

#Combine datasets
events = pd.concat([virtual, real], ignore_index=True)

#Check combined dataset
print(events.head())
neg = (events["weight"] < 0).sum()
print("Number of negative weight events:", neg)
plt.hist(events["pt"], bins=50, weights=events["weight"])
plt.xlabel("pt")
plt.ylabel("Events")
plt.title("pt distribution before resampling")
plt.show()
events_before = events.copy()
def distance(e1, e2):
    return np.sqrt((e1["pt"] - e2["pt"])**2 + 100*(e1["y"] - e2["y"])**2)
for i in range(len(events)):

    if events.loc[i, "weight"] < 0:

        cell = [i]
        total_weight = events.loc[i, "weight"]

        distances = []

        for j in range(len(events)):
            if j != i:
                d = distance(events.loc[i], events.loc[j])
                distances.append((d, j))

        distances.sort()
        for d, j in distances:
            cell.append(j)
            total_weight += events.loc[j, "weight"]

            if total_weight >= 0:
                break
            abs_sum = sum(abs(events.loc[k, "weight"]) for k in cell)
        cell_sum = sum(events.loc[k, "weight"] for k in cell)

        for k in cell:
            events.loc[k, "weight"] = abs(events.loc[k, "weight"]) / abs_sum * cell_sum
            #all weights > 0
#total weight preserved
print("Checking negative weights...")
neg_after = (events["weight"] < 0).sum()
print("Negative weights after resampling:", neg_after)