from multiprocessing.sharedctypes import Value
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

from torch import inverse

sys.path.append("..")

from datasetutils.dataset import Dataset

from experiments.hardness_tracking import load_embeddings

from preprocessing.dimm_reduction import (
    sklearnPCA,
    TripletEmbedding,
    logisticPCA,
    AutoEncoder,
)


def plot_full_distances(
    x,
    y,
    i=117,
    max_x=30,
    max_y=105,
    bin_step=1,
    normalize_dist=False,
    name="baseline",
    inverse_transform=None,
    plot_type="binary",
):
    bins = np.arange(0, max_x, step=bin_step)

    masks = {}
    classes = list(np.unique(y))
    classes.sort()
    for c in classes:
        if plot_type == "binary":
            key = "positive" if c == y[i] else "negative"
        elif plot_type == "full":
            key = inverse_transform([c])[0]
        else:
            raise ValueError("Wrong plot type.")

        if key not in masks:
            masks[key] = y == c
        else:
            masks[key] = np.logical_or(masks[key], y == c)

    k = inverse_transform([y[i]])[0] if plot_type == "full" else "positive"
    masks[k][i] = False

    if name == "baseline":
        distances = abs(x - x[i]).sum(axis=1)
    else:
        distances = np.linalg.norm(x - x[i], axis=1)

    if normalize_dist:
        distances = distances / distances.mean()
    plt.figure(figsize=(8, 4), dpi=80)
    count, _ = np.histogram(distances, bins)

    masked_distances = [distances[mask] for mask in masks.values()]
    labels = (
        inverse_transform(masks.keys())
        if plot_type == "full"
        else masks.keys()
    )
    labels = list(labels)
    colors = ["tab:green", "tab:red", "tab:blue"]

    _, bins, _ = plt.hist(
        masked_distances,
        bins,
        # alpha=0.5,
        histtype="bar",
        ec="black",
        label=labels,
        align="left",
        color=colors[: len(classes) - 1],
        stacked=True,
    )
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.xticks(range(max_x + 1))
    plt.xlim(-(3 * bin_step / 4), max_x + (3 * bin_step / 4))
    plt.yticks(range(0, max_y + 1, 10))
    plt.ylim(0, max_y + 1)
    plt.grid(axis="y", alpha=0.75)
    plt.legend(loc="upper right")

    if max(count) > (max_y + 1):
        warnings.warn(
            "Max count (%d) for %s will overflow the plot (%d)."
            % (max(count), name, (max_y + 1))
        )

    for x, num in zip(bins, count):
        if x == 0:
            num = num - 1
        if num != 0:
            if num >= 10:
                plt.text(x - (bin_step / 2), num + 0.5, num, fontsize=10)
            else:
                plt.text(x - (bin_step / 3), num + 0.5, num, fontsize=10)

    # plt.show()
    if not os.path.isdir("./anchor"):
        os.makedirs("./anchor")

    plt.savefig("./anchor/full-instance-%s-transformed.pdf" % (name))


dataset = Dataset("../datasetutils/db/methods.csv")
dataset.load_samples("../datasetutils/sampled_db", recursive_reading=False)
embeddings_path = "../data/saved_models/"
embedding_dimmensions = [3, 25, 100]
embeddings = {
    "pca": sklearnPCA,
    "logpca": logisticPCA,
    "triplet": TripletEmbedding,
    "autoencodertorch": AutoEncoder,
}

embeddings = load_embeddings(
    dataset,
    embeddings_path,
    ["triplet", "logpca", "pca", "autoencodertorch"],
    embedding_dimmensions,
    embeddings,
)

x_train, y_train, _, _ = dataset.get_sampled(0)
x_train = x_train.values

plot_full_distances(
    x_train, y_train, inverse_transform=dataset.y_transformer.inverse_transform
)  # baseline
plot_full_distances(
    embeddings["triplet"][0][0].transform(x_train),
    y_train,
    max_x=3,
    bin_step=0.05,
    normalize_dist=True,
    name="triplet",
    inverse_transform=dataset.y_transformer.inverse_transform,
)
plot_full_distances(
    embeddings["pca"][2][0].transform(x_train),
    y_train,
    max_x=3,
    bin_step=0.05,
    normalize_dist=True,
    name="pca",
    inverse_transform=dataset.y_transformer.inverse_transform,
)
plot_full_distances(
    embeddings["logpca"][2][0].transform(x_train),
    y_train,
    max_x=3,
    bin_step=0.05,
    normalize_dist=True,
    name="logpca",
    inverse_transform=dataset.y_transformer.inverse_transform,
)
plot_full_distances(
    embeddings["autoencodertorch"][2][0].transform(x_train),
    y_train,
    max_x=3,
    bin_step=0.05,
    normalize_dist=True,
    name="autoencodertorch",
    inverse_transform=dataset.y_transformer.inverse_transform,
)
