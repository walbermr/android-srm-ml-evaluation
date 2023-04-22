from fileinput import filename
import multiprocessing as mp
import functools
import pandas as pd
import numpy as np
import rpy2
import os
import sys
import json

import warnings

warnings.simplefilter("ignore", UserWarning)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from ensemble_classifiers import Ensemble, PoolManager
from deslib.des.knora_e import KNORAE
from deslib.des.knora_u import KNORAU
from deslib.des.meta_des import METADES

from copy import deepcopy

from preprocessing.dimm_reduction import (
    sklearnPCA,
    TripletEmbedding,
    logisticPCA,
    parse_output,
    save_to_file,
)

from deepembeddings.autoencoder import AutoEncoder

import matplotlib.pyplot as plt


def create_ablation_table(classifiers, embeedings, dimmensions, split_sizes):
    table = {}
    for c in classifiers:
        table[c] = {}
        for k in embeedings:
            if k != "baseline":
                table[c][k] = {
                    i: {j: [] for j in dimmensions} for i in split_sizes
                }

        table[c]["baseline"] = {i: {213: []} for i in split_sizes}

    return table


def load_presaved_transformation(model, dimm, sample, set="train"):
    path = os.path.join(
        "preprocessing",
        "reduced_dataset",
        model,
        "%d_components" % (dimm),
        "sample_%d_%s.csv" % (sample, set),
    )

    if os.path.exists(path):
        df = pd.read_csv(path, sep=";").values[:, :-1]
    else:
        df = None

    return df


def create_path(path_list: list):
    accumulated_path = "./"
    for p in path_list:
        accumulated_path = os.path.join(accumulated_path, p)
        if not os.path.isdir(accumulated_path):
            os.makedirs(accumulated_path)

    return accumulated_path


def ablation_worker(
    dataset,
    classifiers,
    embedding_models,
    embedding_dimmensions,
    train_split_sizes,
    experiments,
    embedding,
    load_saved=True,
    embeddings_path="",
):
    table = create_ablation_table(
        classifiers, embedding_models, embedding_dimmensions, train_split_sizes
    )

    for train_size in train_split_sizes:
        # clear globals every train split size
        PoolManager().clear()

        for k, embedding_dimmension in enumerate(embedding_dimmensions):
            PoolManager().clear()

            if embedding != "baseline":
                embeddings_pool = []

            for classifier_name in classifiers:
                classifier = classifiers[classifier_name]
                for j in range(experiments):
                    X_train, y_train, X_test, y_test = dataset.get_sampled(j)
                    X_train, X_test = X_train.values, X_test.values

                    if train_size != 0.8:
                        X_train, _, y_train, _ = train_test_split(
                            X_train,
                            y_train,
                            test_size=1 - (train_size / 0.8),
                            stratify=y_train,
                            random_state=199,
                        )

                    current_embedding_path = create_path(
                        [
                            embeddings_path,
                            embedding,
                            str(embedding_dimmension),
                            str(train_size).replace(".", ""),
                        ]
                    )
                    embedding_file_name = "sample_%d.model" % (j)
                    embedding_model_path = os.path.join(
                        current_embedding_path, embedding_file_name
                    )

                    try:
                        if (
                            embedding != "baseline"
                            and len(embeddings_pool) < experiments
                        ):
                            trained_embedding = os.path.isfile(
                                embedding_model_path
                            )
                            if load_saved and trained_embedding:
                                try:
                                    embeddings_pool.append(
                                        embedding_models[embedding](
                                            embedding_dimmension
                                        ).load(embedding_model_path)
                                    )
                                except Exception as err:
                                    raise Exception(err)
                            else:
                                embeddings_pool.append(
                                    embedding_models[embedding](
                                        embedding_dimmension
                                    )
                                )
                                X_emb_train = embeddings_pool[j].fit(
                                    X_train, y_train
                                )
                                embeddings_pool[j].save(embedding_model_path)

                            X_emb_train = embeddings_pool[j].transform(X_train)
                            X_emb_test = embeddings_pool[j].transform(X_test)

                        elif (
                            embedding != "baseline"
                            and len(embeddings_pool) == experiments
                        ):
                            X_emb_train = embeddings_pool[j].transform(X_train)
                            X_emb_test = embeddings_pool[j].transform(X_test)
                        else:
                            # continue loop if baseline and k>0, as k=0 is 213 dims
                            if k != 0:
                                continue
                            embedding_dimmension = 213
                            X_emb_train = X_train
                            X_emb_test = X_test
                    except (
                        ValueError,
                        AttributeError,
                        rpy2.rinterface_lib.embedded.RRuntimeError,
                    ) as e:
                        # handles the case if quantity of samples is lower than ouput dimmensions by PCA
                        # any other case, yields an error
                        if embedding == "pca" or embedding == "logpca":
                            acc = np.nan
                        else:
                            print(
                                "Error at %s %d"
                                % (embedding, embedding_dimmension)
                            )
                            raise e
                    else:
                        c = classifier()
                        c.fit(X_emb_train, y_train)
                        y_pred = c.predict(X_emb_test)
                        acc = accuracy_score(y_test, y_pred)
                        del c

                    table[classifier_name][embedding][train_size][
                        embedding_dimmension
                    ].append(acc)

                    train_output = parse_output(
                        X_emb_train,
                        dataset.y_transformer.inverse_transform(
                            y_train.astype(int)
                        ),
                    )
                    test_output = parse_output(
                        X_emb_test,
                        dataset.y_transformer.inverse_transform(
                            y_test.astype(int)
                        ),
                    )
                    save_to_file(
                        train_output,
                        embedding,
                        embedding_dimmension,
                        train_size,
                        "sample_%d_train" % (j),
                    )
                    save_to_file(
                        test_output,
                        embedding,
                        embedding_dimmension,
                        train_size,
                        "sample_%d_test" % (j),
                    )

                    if not np.isnan(acc):
                        sys.stdout.write("\033[K")
                        print(
                            "{0: <13} tz: {1: <1} {2: <40} dimm: {3: <4} acc: {4: <.2}".format(
                                embedding,
                                train_size,
                                classifier_name,
                                embedding_dimmension,
                                acc,
                            ),
                            end="\r",
                        )

    return {embedding: table}


def load_table_multiple_files():
    table = {}
    path = "./visualizations/ablation/results/"
    filenames = os.listdir(path)
    files = [
        os.path.join(path, f)
        for f in filenames
        if os.path.isfile(os.path.join(path, f))
    ]
    filenames = [f.replace(".txt", "") for f in filenames]

    for i, file in enumerate(files):
        with open(file, "r") as f:
            c_file = json.loads(f.read())
            table[filenames[i]] = c_file

    return table


def load_table():
    with open("./data/ablation/result_table.txt", "r") as f:
        return json.loads(f.read())


def save_table(table):
    if not os.path.isdir("./data/ablation"):
        os.makedirs("./data/ablation")

    with open("./data/ablation/result_table.txt", "w+") as f:
        f.write(json.dumps(table))


def ablation(
    dataset,
    samples_path="./datasetutils/sampled_db/",
    train=False,
    embeddings_path="./data/saved_models/",
    embedding_dimmensions=[2, 3, 5, 10, 15, 20, 25, 50, 100, 150, 200],
    train_split_sizes=[0.8],
    experiments=30,
    use_multiprocess=True,
    load_saved_table=False,
):
    classifiers = {
        "MLP": functools.partial(
            MLPClassifier,
            100,
            max_iter=1000,
            random_state=199,
        ),
        "KNN": functools.partial(KNeighborsClassifier, n_neighbors=7),
        "SVM Linear": functools.partial(
            SVC, kernel="linear", random_state=199
        ),
        "Random Forest": functools.partial(
            RandomForestClassifier,
            n_estimators=100,
            random_state=199,
        ),
        "KNORA-U": functools.partial(
            Ensemble,
            RandomForestClassifier(n_estimators=100, random_state=199),
            KNORAU,
            test_size=None,
            pool_training_scheme=None,
        ),
        "KNORA-E": functools.partial(
            Ensemble,
            RandomForestClassifier(n_estimators=100, random_state=199),
            KNORAE,
            test_size=None,
            pool_training_scheme=None,
        ),
        "META-DES": functools.partial(
            Ensemble,
            RandomForestClassifier(n_estimators=100, random_state=199),
            METADES,
            test_size=None,
            pool_training_scheme=None,
        ),
    }

    embedding_models = {
        "triplet": TripletEmbedding,
        "pca": sklearnPCA,
        "autoencodertorch": functools.partial(
            AutoEncoder,
            n_epochs=100000,
            batch_size=100,
            verbose=False,
            early_stopping=100,
        ),
        "logpca": logisticPCA,
    }

    if load_saved_table and os.path.isfile("./data/ablation/result_table.txt"):
        table = load_table()
    else:
        table = create_ablation_table(
            classifiers,
            embedding_models,
            embedding_dimmensions,
            train_split_sizes,
        )

    if train:
        dataset.load_samples(
            samples_path, recursive_reading=False
        )  # remake the divisions

        worker = functools.partial(
            ablation_worker,
            dataset,
            classifiers,
            embedding_models,
            embedding_dimmensions,
            train_split_sizes,
            experiments,
            embeddings_path=embeddings_path,
        )

        if use_multiprocess:
            with mp.Pool(len(embedding_models)) as p:
                results = p.map(worker, embedding_models)
                for r in results:
                    for k in r:
                        for c in r[k]:
                            if c not in table:
                                table[c] = {}
                            table[c][k] = r[k][c][k]
        else:
            for e in embedding_models:
                results = worker(e)
                for c in results[e]:
                    table[c][e] = results[e][c][e]

        if not os.path.isdir("./visualizations/ablation/results"):
            os.makedirs("./visualizations/ablation/results")

        for c in classifiers:
            # save_data
            with open(
                "./visualizations/ablation/results/%s.txt" % (c), "w"
            ) as f:
                f.write(json.dumps(table[c]))

    save_table(table)

    # overall_ablation(classifiers.keys())
    # ablation_rate(classifiers.keys())
    # chart_pairwise_comparison()
    # table_pairwise_comparison()
    ablation_by_embeddings(classifiers.keys())


def print_baseline_bar(value, x):
    mean = value.mean()
    std = value.std()
    plt.plot(x, [mean] * len(x), color="k")
    plt.fill_between(x, mean - std, mean + std, alpha=0.2, color="k")


def get_markers():
    return ["o", "v", "^", "<", ">", "s", "*", "h", "H", "D", "d", "P", "X"]


def save_plot(*path_structure, path="./visualizations/ablation"):
    for p in path_structure[:-1]:
        path = os.path.join(path, p)
        print("creating %s" % (path))
        if not os.path.exists(path):
            os.makedirs(path)

    path = os.path.join(path, path_structure[-1].replace(" ", "_") + ".pdf")
    plt.savefig(path)


# colocar baseline no valor 213 do x
def ablation_by_embeddings(
    classifier_names,
    embedding_dimmensions=[2, 3, 5, 10, 15, 20, 25, 50, 100, 150, 200],
    embedding_test_sizes=[0.8],
):
    filled_markers = get_markers()
    embedding_names = []
    table_data = {}
    x = [str(x_i) for x_i in embedding_dimmensions]

    for classifier_name in classifier_names:
        with open(
            "./visualizations/ablation/results/%s.txt" % (classifier_name), "r"
        ) as f:
            s = f.read()
            table_data[classifier_name] = json.loads(s)

    for n in table_data[list(classifier_names)[0]]:
        embedding_names.append(n)

    # fazer iterador com produto cartesiano
    for embedding_name in embedding_names:
        plt.clf()
        plt.cla()
        if embedding_name == "baseline":
            continue

        # table_data[classifier_name][embedding_name]["0.8"]["213"] = \
        #     table_data[classifier_name]["baseline"]["0.8"]["213"]
        for i, classifier_name in enumerate(classifier_names):
            if embedding_name not in table_data[classifier_name]:
                continue

            line = table_data[classifier_name][embedding_name]["0.8"]
            line_mean = [np.mean(line[x]) for x in line]
            line_err = [np.std(line[x]) for x in line]

            plt.errorbar(
                x,
                line_mean,
                yerr=line_err,
                label=classifier_name,
                fmt="%c--" % (filled_markers[i]),
            )

        plt.ylim((0.40, 0.95))
        plt.yticks(np.arange(0.40, 1.0, 0.05))
        plt.legend(loc="lower right")
        plt.xlabel("Dimmensions", fontsize=16)
        plt.ylabel("Accuracy", fontsize=16)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.hlines(
            np.arange(0.45, 0.95, 0.05),
            -1,
            len(x),
            colors="lightgrey",
            linestyles="dashed",
        )
        plt.xlim((-1, len(x)))
        save_plot(*["paper", embedding_name])


def overall_ablation(
    classifier_names,
    embedding_dimmensions=[2, 3, 5, 10, 15, 20, 25, 50, 100, 150, 200],
):
    filled_markers = get_markers()

    for classifier_name in classifier_names:
        with open(
            "./visualizations/ablation/results/%s.txt" % (classifier_name), "r"
        ) as f:
            s = f.read()
            table_data = json.loads(s)

            # print_data
            for model_name in table_data:
                plt.clf()
                plt.cla()

                baseline_value = np.array(table_data["baseline"]["0.8"]["213"])
                x = []
                print(model_name)
                for i, line in enumerate(table_data[model_name]):
                    train_size = round(float(line), 2)
                    print(train_size)

                    if model_name == "baseline":
                        x = ["213"]
                    else:
                        x = [str(x_i) for x_i in embedding_dimmensions]
                    y = [
                        np.mean(table_data[model_name][line][i])
                        for i in table_data[model_name][line]
                    ]

                    yerr = [
                        np.std(table_data[model_name][line][i])
                        for i in table_data[model_name][line]
                    ]
                    plt.errorbar(
                        x,
                        y,
                        yerr=yerr,
                        label="%d%% " % (train_size * 100),
                        fmt="%c--" % (filled_markers[i]),
                    )
                    [
                        print("%.4f(%.4f)" % (mean, std))
                        for mean, std in zip(y, yerr)
                    ]
                    print("\n")

                if model_name != "baseline":
                    print_baseline_bar(baseline_value, x)

                plt.ylim((0.40, 0.95))
                plt.yticks(np.arange(0.40, 1.0, 0.05))
                plt.legend(loc="lower right")
                plt.xlabel("Dimmensions", fontsize=16)
                plt.ylabel("Accuracy", fontsize=16)
                plt.yticks(fontsize=14)
                plt.xticks(fontsize=14)

                save_plot(*["acc", model_name, classifier_name])


def ablation_rate(
    classifier_names,
    embedding_dimmensions=[2, 3, 5, 10, 15, 20, 25, 50, 100, 150, 200],
    train_sizes=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
):
    filled_markers = get_markers()

    for classifier_name in classifier_names:
        with open(
            "./visualizations/ablation/results/%s.txt" % (classifier_name), "r"
        ) as f:
            s = f.read()
            table_data = json.loads(s)
            baseline_table = table_data["baseline"]

            for model_name in table_data:
                plt.clf()
                plt.cla()
                if model_name == "baseline":
                    continue
                data = table_data[model_name]
                mean_data = []
                std_data = []

                for i, p in enumerate(train_sizes):
                    baseline_data = np.array(baseline_table[str(p)]["213"])
                    embeddings_data = np.array(
                        [data[str(p)][str(d)] for d in embedding_dimmensions]
                    )
                    embeddings_rate = (
                        embeddings_data / baseline_data - 1.0
                    ) * 100

                    embeddings_mean_rate = np.array(
                        [np.mean(x) for x in embeddings_rate]
                    )
                    embeddings_mean_std = np.array(
                        [np.std(x) for x in embeddings_rate]
                    )
                    mean_data.append(embeddings_mean_rate)
                    std_data.append(embeddings_mean_std)

                    x = [str(x_i) for x_i in embedding_dimmensions]
                    y = embeddings_mean_rate
                    yerr = embeddings_mean_std
                    plt.errorbar(
                        x,
                        y,
                        label="%d%% " % (p * 100),
                        fmt="%c--" % (filled_markers[i]),
                    )
                    plt.ylim((-40, 25))
                    plt.yticks(np.arange(-40, 30, 10))
                    plt.axhline(0, color="k")

                if classifier_name == "random_forest" and model_name == "pca":
                    plt.legend(loc="upper right")
                else:
                    plt.legend(loc="lower right")

                plt.xlabel("Dimmensions")
                plt.ylabel("Accuracy Rate (%)")

                save_plot(*["rate", model_name, classifier_name])


def table_pairwise_comparison(
    embedding_dimmensions=[2, 3, 5, 10, 15, 20, 25, 50, 100, 150, 200]
):
    base_classifier = "mlp"
    classifier = "knn"
    model = "triplet"
    train_size = "0.8"

    with open(
        "./visualizations/%s_ablation_results.txt" % (classifier), "r"
    ) as f:
        s = f.read()
        table_data = json.loads(s)
    with open(
        "./visualizations/%s_ablation_results.txt" % (base_classifier), "r"
    ) as f:
        s = f.read()
        base_table_data = json.loads(s)

    # print_data
    data = table_data[model]
    baseline_table = base_table_data["baseline"]

    baseline_data = np.array(baseline_table[str(train_size)]["2"])
    embeddings_data = np.array(
        [data[str(train_size)][str(d)] for d in embedding_dimmensions]
    )

    embeddings_mean = np.array([np.mean(x) for x in embeddings_data])
    embeddings_mean_std = np.array([np.std(x) for x in embeddings_data])

    print("%s" % (model))
    for i in range(len(embeddings_mean)):
        print("%0.4f(%0.4f)" % (embeddings_mean[i], embeddings_mean_std[i]))

    print(
        "%s %0.4f(%0.4f)"
        % (
            classifier,
            np.mean(table_data["baseline"][train_size]["2"]),
            np.std(table_data["baseline"][train_size]["2"]),
        )
    )
    print(
        "%s %0.4f(%0.4f)"
        % (base_classifier, np.mean(baseline_data), np.std(baseline_data))
    )


def stat_pairwise_comparison(
    embedding_dimmensions=[2, 3, 5, 10, 15, 20, 25, 50, 100, 150, 200],
    train_sizes=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
):
    pass


def chart_pairwise_comparison(
    classifier_names,
    embedding_dimmensions=[2, 3, 5, 10, 15, 20, 25, 50, 100, 150, 200],
    train_sizes=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
):
    base_classifier = "mlp"
    classifier = "knn"
    model = "triplet"
    filled_markers = [
        "o",
        "v",
        "^",
        "<",
        ">",
        "s",
        "*",
        "h",
        "H",
        "D",
        "d",
        "P",
        "X",
    ]

    with open(
        "./visualizations/%s_ablation_results.txt" % (classifier), "r"
    ) as f:
        s = f.read()
        table_data = json.loads(s)
    with open(
        "./visualizations/%s_ablation_results.txt" % (base_classifier), "r"
    ) as f:
        s = f.read()
        base_table_data = json.loads(s)

    # print_data
    data = table_data[model]
    baseline_table = base_table_data["baseline"]
    mean_data = []
    std_data = []

    for i, p in enumerate(train_sizes):
        baseline_data = np.array(baseline_table[str(p)]["2"])
        embeddings_data = np.array(
            [data[str(p)][str(d)] for d in embedding_dimmensions]
        )
        embeddings_rate = (embeddings_data / baseline_data - 1.0) * 100

        embeddings_mean_rate = np.array([np.mean(x) for x in embeddings_rate])
        embeddings_mean_std = np.array([np.std(x) for x in embeddings_rate])
        mean_data.append(embeddings_mean_rate)
        std_data.append(embeddings_mean_std)

        x = [str(x_i) for x_i in embedding_dimmensions]
        y = embeddings_mean_rate
        yerr = embeddings_mean_std
        plt.errorbar(
            x, y, label="%d%% " % (p * 100), fmt="%c--" % (filled_markers[i])
        )
        plt.ylim((-40, 20))
        plt.yticks(np.arange(-40, 30, 10))
        plt.axhline(0, color="k")

        plt.legend(loc="lower right")

        plt.xlabel("Dimmensions")
        plt.ylabel("Accuracy Rate (%)")

        if not os.path.exists("./visualizations/ablation/rate"):
            os.makedirs("./visualizations/ablation/rate")

        plt.savefig(
            "./visualizations/ablation/rate/%sx%s_%s.png"
            % (base_classifier, classifier, model)
        )
