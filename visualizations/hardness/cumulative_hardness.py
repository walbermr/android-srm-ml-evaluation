import json, argparse, operator

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from deslib.util.instance_hardness import kdn_score

import sys

sys.path.append("../../")

from datasetutils.dataset import Dataset
from experiments.base_experiments import BasicExperiments


class Entry:
    def __init__(self):
        self.metric = {}

    def add(self, dimm, metric):
        self.metric[dimm] = metric


def get_entry(entries, algo):
    entry = entries[algo]
    x = sorted([int(x_i) for x_i in entry.metric.keys()])
    x = [str(x_i) for x_i in x]

    y = [entry.metric[k] for k in x]
    y.append(entries["baseline"].metric["213"])
    y = np.array(y)

    return x, y


def main(neighbors, reduction_algorithm, selected_entry, selected_dimm=100):
    samples_path = "../../preprocessing/reduced_dataset/"
    baseline_samples = "../../datasetutils/sampled_db/"
    (
        recursive_directories,
        path_is_recursive,
    ) = BasicExperiments.get_recursive_directories(samples_path)
    dataset = Dataset(
        "../../datasetutils/db/methods.csv"
    )  # use already preprocessed db

    dimm_map = {
        2: 0,
        3: 1,
        5: 2,
        10: 3,
        15: 4,
        20: 5,
        25: 6,
        50: 7,
        100: 8,
        150: 9,
        200: 10,
    }
    entries = {}
    dimmensions = []
    for sub_directory in recursive_directories:
        for n_components in sub_directory:
            n_components_name = list(n_components.keys())[0]
            n_components_path = n_components[n_components_name]
            n_components_path += "/0.80"

            dataset.load_samples(
                n_components_path, recursive_reading=path_is_recursive
            )
            hardnesses = []

            algorithm, dimm = n_components_name.split("_")

            if dimm not in dimmensions:
                dimmensions.append(dimm)

            if selected_entry == "all":
                for i in range(0, len(dataset.dataframe["samples"])):
                    X_train, Y_train, X_test, Y_test = dataset.get_sampled(i)
                    hardnesses.extend(kdn_score(X_test, Y_test, neighbors)[0])
            else:
                X_train, Y_train, X_test, Y_test = dataset.get_sampled(
                    int(selected_entry)
                )
                hardnesses.extend(kdn_score(X_test, Y_test, neighbors)[0])

            hardnesses = np.array(hardnesses)

            if algorithm not in entries:
                entries[algorithm] = Entry()

            entries[algorithm].add(dimm, hardnesses)

    dataset.load_samples(baseline_samples)
    baseline_hardnesses = []
    if selected_entry == "all":
        for i in range(0, len(dataset.dataframe["samples"])):
            X_train, Y_train, X_test, Y_test = dataset.get_sampled(i)
            baseline_hardnesses.extend(kdn_score(X_test, Y_test, neighbors)[0])
    else:
        X_train, Y_train, X_test, Y_test = dataset.get_sampled(
            int(selected_entry)
        )
        baseline_hardnesses.extend(kdn_score(X_test, Y_test, neighbors)[0])

    entries["baseline"] = Entry()
    entries["baseline"].add("213", baseline_hardnesses)

    colors = {
        "autoencoder": "tab:red",
        "autoencodertorch": "tab:red",
        "triplet": "tab:green",
        "pca": "tab:blue",
        "logpca": "tab:cyan",
        "baseline": "tab:orange",
    }

    fig, ax = plt.subplots()

    entry = entries[reduction_algorithm]
    x = sorted([int(x_i) for x_i in entry.metric.keys()])
    x = [str(x_i) for x_i in x]

    y = [entry.metric[k] for k in x]
    y.append(entries["baseline"].metric["213"])
    y = np.array(y)

    i = dimm_map[selected_dimm]

    ax.hist(
        get_entry(entries, "pca")[1][i].transpose(),
        50,
        density=True,
        histtype="step",
        cumulative=True,
        label="pca",
        color=colors["pca"],
    )
    ax.hist(
        get_entry(entries, "logpca")[1][i].transpose(),
        50,
        density=True,
        histtype="step",
        cumulative=True,
        label="logpca",
        color=colors["logpca"],
    )
    ax.hist(
        get_entry(entries, "autoencodertorch")[1][i].transpose(),
        50,
        density=True,
        histtype="step",
        cumulative=True,
        label="autoencoder",
        color=colors["autoencodertorch"],
    )
    ax.hist(
        get_entry(entries, "baseline")[1][0].transpose(),
        50,
        density=True,
        histtype="step",
        cumulative=True,
        label="baseline",
        color=colors["baseline"],
    )
    ax.hist(
        get_entry(entries, "triplet")[1][i].transpose(),
        50,
        density=True,
        histtype="step",
        cumulative=True,
        label="triplet",
        color=colors["triplet"],
    )
    plt.hlines(
        np.arange(0.05, 1.0, 0.05),
        0,
        1.0,
        colors="lightgrey",
        linestyles="dashed",
    )
    plt.yticks(np.arange(0.0, 1.05, 0.1))
    plt.xticks(np.arange(0.0, 1.05, 0.1))

    plt.xlabel("kDN score")
    plt.ylabel("Cumulative Distribution")
    plt.legend(loc="lower right")
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.savefig("../cumulative-transformed.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--neighbors",
        action="store",
        dest="neighbors",
        default=7,
        type=int,
        help="Define how many neighbors to consider during hardness calculation",
    )
    parser.add_argument(
        "-r",
        "--reduction_algorithm",
        dest="reduction_algorithm",
        default="pca",
        help="Define which model to compare embedding reductions. (pca, triplet or autoencoder)",
    )
    parser.add_argument(
        "-e",
        "--entries",
        dest="entries",
        default="all",
        help="Define which sample to use, ranging from '0' to '29' or 'all'.",
    )
    parser.add_argument(
        "-d",
        "--dimm",
        dest="dimm",
        default=100,
        help="Define which dimmension to use, choosing from 2, 3, 5, 10, 15, 20, 25, 50, 100, 150, 200",
    )

    args = parser.parse_args()

    entries = args.entries
    reduction_algorithm = args.reduction_algorithm
    neighbors = args.neighbors
    dimm = int(args.dimm)

    main(neighbors, reduction_algorithm, entries, selected_dimm=dimm)
