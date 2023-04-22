import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

sys.path.append("..")

from datasetutils.dataset import Dataset

from sklearn.preprocessing import LabelEncoder


def main():
    dataset = Dataset("../datasetutils/db/methods.csv")  # use already preprocessed db
    df = dataset.dataframe["original"]
    labels = ["source", "sink", "neithernor"]

    X_transformer = LabelEncoder()
    X_transformer.fit([True, False])
    colors = ["b", "r", "g"]

    feature_keys = dataset.feature_keys
    target = dataset.target_key
    entries = []
    sizes = []
    for l, c in zip(labels, colors):
        entry = df.loc[df["class"] == l][feature_keys]
        entry = entry.apply(X_transformer.transform).values
        size = len(entry)
        sizes.append(size)
        entry = np.sum(entry, axis=0)
        entries.append(entry)

        # print(stats)
        angles = np.linspace(0, 2 * np.pi, len(feature_keys), endpoint=False)

        # stats = np.concatenate((stats,[stats[0]]))
        # angles = np.concatenate((angles,[angles[0]]))

        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, np.true_divide(entry, size), "o-", linewidth=1, color=c)
        ax.fill(angles, np.true_divide(entry, size), alpha=0.25)
        ax.set_thetagrids(angles * 180 / np.pi, [])
        ax.set_title(l)
        ax.grid(True)

        # plt.show()
        plt.savefig("./figures/radar/" + l + ".png", dpi=800)

    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.set_title("classes")
    for entry, size, color in zip(entries, sizes, colors):
        ax.plot(angles, np.true_divide(entry, size), "o-", linewidth=1, color=color)
        ax.fill(angles, np.true_divide(entry, size), alpha=0.25)
    ax.set_thetagrids(angles * 180 / np.pi, [])
    ax.set_title("Proportion by Class")
    ax.legend(labels, loc=(0.9, 0.95), labelspacing=0.1, fontsize="small")
    ax.grid(True)

    # plt.show()
    plt.savefig("./figures/radar/mixed.png", dpi=800)


if __name__ == "__main__":
    main()
