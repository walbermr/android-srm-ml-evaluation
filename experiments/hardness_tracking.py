import pandas as pd
import numpy as np
import os
import math
import csv
import re

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

from copy import deepcopy
from tqdm import tqdm, trange

from preprocessing.dimm_reduction import (
    TripletEmbedding,
)

from deepembeddings.autoencoder import AutoEncoder

from deslib.util.instance_hardness import kdn_score

import matplotlib.pyplot as plt

from scipy.stats import wilcoxon


def train_embeddings(
    dataset, model_names, embedding_dimmensions, embeddings_path, model_classes
):
    trained_models = {k: [] for k in model_classes.keys()}

    X_transformer = LabelEncoder()
    X_transformer.fit([True, False])

    y_transformer = LabelEncoder()

    X_transform = lambda x: x.apply(X_transformer.transform).values.astype(
        np.float32
    )
    y_transform = lambda y: y.astype(np.float32)

    t = trange(0, len(model_names), desc="Bar desc", leave=True)

    for i in t:
        model_name = model_names[i]
        for dimmension in embedding_dimmensions:
            models_lists = []
            for j in range(len(dataset.dataframe["samples"])):
                t.set_description("Training %s sample %d" % (model_name, j))
                X_train, y_train, _, _ = dataset.get_sampled(j)
                X_train, y_train = X_transform(X_train), y_transform(y_train)

                m = model_classes[model_name](dimmension)
                m.fit(X_train, y_train)

                model_path = os.path.join(
                    embeddings_path, model_name, str(dimmension)
                )

                if not os.path.isdir(
                    os.path.join(embeddings_path, model_name)
                ):
                    os.mkdir(os.path.join(embeddings_path, model_name))
                if not os.path.isdir(model_path):
                    os.mkdir(model_path)

                file_name = "sample_%d.model" % (j)
                m.save(os.path.join(model_path, file_name))
                models_lists.append(m)

            trained_models[model_name].append(models_lists)

    return trained_models


def load_embeddings(
    dataset, embeddings_path, model_names, embedding_dimmensions, model_classes
):
    # trained_models = {"triplet": [], "pca": [], "autoencoder": []}
    trained_models = {k: [] for k in model_classes.keys()}

    t = trange(0, len(model_names), desc="Bar desc", leave=True)

    for i in t:
        model_name = model_names[i]
        for dimmension in embedding_dimmensions:
            models_lists = []
            for j in range(len(dataset.dataframe["samples"])):
                t.set_description("Loading %s sample %d" % (model_name, j))
                model_path = os.path.join(
                    embeddings_path, model_name, str(dimmension)
                )
                file_name = "sample_%d.model" % (j)

                model_class = model_classes[model_name]
                models_lists.append(
                    model_class.load(
                        os.path.join(os.path.join(model_path, file_name))
                    )
                )

            trained_models[model_name].append(models_lists)

    return trained_models


def get_boolean_comparison(y_pred, comparison):
    """This function gets only the instances that are missed by y_pred and are right in comparison"""
    a = y_pred
    b = comparison

    c = np.logical_xor(a, b)
    d = np.logical_not(a)
    return np.logical_and(c, d)


def track_hardness(
    dataset,
    samples_path="./datasetutils/sampled_db/",
    train=True,
    embeddings_path="./data/saved_models/",
    base_models=["triplet"],
    comparison_models=["pca", "autoencoder", "original"],
    embedding_dimmensions=[2, 3, 5, 10, 15, 20, 25, 50, 100, 150, 200],
    classifier_class=KNeighborsClassifier,
    models={"triplet": TripletEmbedding},
):
    dataset.load_samples(samples_path, recursive_reading=False)

    model_names = base_models + comparison_models
    # train/load embeddings
    if train:
        print("Training embedding models.")
        embeddings = train_embeddings(
            dataset,
            list(set(model_names) - set(["original"])),
            embedding_dimmensions,
            embeddings_path,
            models,
        )
    else:
        print("Loading embedding models.")
        embeddings = load_embeddings(
            dataset,
            embeddings_path,
            list(set(model_names) - set(["original"])),
            embedding_dimmensions,
            models,
        )

    X_transformer = LabelEncoder()
    X_transformer.fit([True, False])

    y_transformer = LabelEncoder()
    y_keys = ["source", "sink", "neithernor"]
    y_transformer.fit(y_keys)

    X_transform = lambda x: x.apply(X_transformer.transform).values.astype(
        np.float32
    )
    y_transform = lambda y: y.astype(np.float32)

    table = {}
    hardness_table = {}
    pairwise_hardness_table = {}
    for n in model_names:
        hardness_table[n] = []
    for model_name in base_models + comparison_models:
        hardness_table[model_name] = [[] for i in embedding_dimmensions]

    for model_name in base_models:
        models_comparison = list(set(comparison_models) - set([model_name]))
        models_comparison.sort()
        append_base_model_hardness = (
            True  # controls when append base model hardness to hardness_table
        )
        for model_idx, model_comparison in enumerate(models_comparison):
            table_entry = "%s-%s" % (model_name, model_comparison)
            table[table_entry] = {
                "Dimmensions": embedding_dimmensions + ["total"]
            }
            pairwise_hardness_table[table_entry] = [
                [[], []] for i in range(len(embedding_dimmensions))
            ]

            for k in y_keys + ["total"]:
                table[table_entry][k] = [
                    [] for i in range(len(embedding_dimmensions) + 1)
                ]

            for i in range(len(embedding_dimmensions)):
                for j in range(len(dataset.dataframe["samples"])):
                    classifier_a = classifier_class()
                    model_a = embeddings[model_name][i][j]

                    X_train, y_train, X_test, y_test = dataset.get_sampled(j)
                    X_test, y_test = X_transform(X_test), y_transform(y_test)

                    X_train_transformed = model_a.transform(X_train)
                    classifier_a.fit(X_train_transformed, y_train)

                    model_a_X_test = model_a.transform(X_test)
                    model_a_output = (
                        classifier_a.predict(model_a_X_test) == y_test
                    )

                    print(
                        "Compare %s with %s on %d dimmns, have complement? "
                        % (
                            model_name,
                            model_comparison,
                            embedding_dimmensions[i],
                        ),
                        end="",
                    )
                    if model_comparison != "original":
                        model_b = embeddings[model_comparison][i][j]
                        X_train_transformed = model_b.transform(X_train)
                        model_b_X_test = model_b.transform(X_test)
                    else:
                        X_train_transformed = X_train
                        model_b_X_test = X_test

                    classifier_b = classifier_class()
                    classifier_b.fit(X_train_transformed, y_train)
                    model_b_output = (
                        classifier_b.predict(model_b_X_test) == y_test
                    )

                    comparison = get_boolean_comparison(
                        model_a_output, model_b_output
                    )
                    class_frequency = (
                        pd.Series(y_test[comparison]).value_counts().to_dict()
                    )
                    keys = list(class_frequency.keys())

                    model_a_hardness = kdn_score(model_a_X_test, y_test, 7)[0][
                        comparison
                    ]
                    model_b_hardness = kdn_score(model_b_X_test, y_test, 7)[0][
                        comparison
                    ]

                    if append_base_model_hardness:
                        hardness_table[model_name][i].append(model_a_hardness)

                    hardness_table[model_comparison][i].append(
                        model_b_hardness
                    )
                    pairwise_hardness_table[table_entry][i][
                        0
                    ] += model_a_hardness.tolist()
                    pairwise_hardness_table[table_entry][i][
                        1
                    ] += model_b_hardness.tolist()

                    comparison_indexes = [
                        i
                        for i, e in enumerate(comparison.tolist())
                        if e == True
                    ]

                    for k in keys:
                        freq = class_frequency.pop(k)
                        new_key = y_transformer.inverse_transform([int(k)])[0]
                        class_frequency[new_key] = int(freq)

                    missing_keys = list(
                        set(y_keys) - set(class_frequency.keys())
                    )
                    class_frequency["total"] = comparison.sum()
                    for k in y_keys + ["total"]:
                        if k not in missing_keys:
                            table[table_entry][k][i].append(class_frequency[k])
                        else:
                            class_frequency[k] = 0
                            table[table_entry][k][i].append(0)

                    print(comparison.any())
                    print(class_frequency)
            append_base_model_hardness = False
        print("\n")

    print("Total instances per test ", end="")
    print(y_test.size)

    for i in table.keys():
        for j in y_keys + ["total"]:
            label_total = []  # this gets the total per table column
            for k in range(len(embedding_dimmensions)):
                label_total += table[i][j][k]
                table[i][j][k] = "%0.4f +- %0.4f" % (
                    np.mean(table[i][j][k]),
                    np.std(table[i][j][k]),
                )
            k = len(embedding_dimmensions)
            table[i][j][k] = "%0.4f +- %0.4f" % (
                np.mean(label_total),
                np.std(label_total),
            )

    for k in table.keys():
        print(k)
        print(table[k])

        if not os.path.isdir("latex_output"):
            os.mkdir("latex_output")

        with open("latex_output/%s_instances_stats.tex" % (k), "w") as f:
            f.write(pd.DataFrame(data=table[k]).to_latex(index=False))

    for m in model_names:
        plt.figure()
        plt.hist(np.concatenate(np.concatenate(hardness_table[m])))
        plt.title(m)

    plt.show()

    return hardness_table, pairwise_hardness_table


def hardness_statistics(
    name_a,
    name_b,
    hardness_table,
    embedding_dimmensions=[2, 3, 5, 10, 15, 20, 25, 50, 100, 150, 200],
):
    table_entry = "%s-%s" % (name_a, name_b)

    for i, dimm in enumerate(embedding_dimmensions):
        hardness_a = np.array(hardness_table[table_entry][i][0])
        hardness_b = np.array(hardness_table[table_entry][i][1])

        _, p = wilcoxon(hardness_a, hardness_b)
        print("%s dimm %d: %0.4f" % (table_entry, dimm, p))

    hardness_a = np.concatenate([t[0] for t in hardness_table[table_entry]])
    hardness_b = np.concatenate([t[1] for t in hardness_table[table_entry]])

    _, p = wilcoxon(hardness_a, hardness_b)
    print("%s all dimms: %0.4f\n" % (table_entry, p))
