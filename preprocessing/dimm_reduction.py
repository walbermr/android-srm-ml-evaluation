from tabnanny import verbose
import torch
import math
import os
import sys
import pickle
import itertools
import functools

import pandas as pd
import numpy as np
import multiprocessing as mp

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.neural_network._base import ACTIVATIONS
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import VarianceThreshold

from sklearn.base import BaseEstimator

from deepembeddings.triplet import TripletEmbedding
from deepembeddings.autoencoder import AutoEncoder
from torch._C import Value

from datasetutils.dataset import Dataset
from orderedset import OrderedSet

np.random.seed(seed=199)

import rpy2
import rpy2.robjects.numpy2ri
import rpy2.robjects.packages as rpackages

rutils = rpackages.importr("utils")
rutils.chooseCRANmirror(ind=1)
from rpy2.robjects.vectors import StrVector

packnames = ("logisticPCA",)
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    print(names_to_install)
    rutils.install_packages(StrVector(names_to_install))
from rpy2.robjects.packages import importr

rpy2.robjects.numpy2ri.activate()
logPCA = importr("logisticPCA")


class DimmClassBase(object):
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        obj = None
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj


class logisticPCA(DimmClassBase):
    def __init__(self, n_components):
        self._dim = n_components

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)

    def fit(self, x, y):
        # output = logPCA.LogisticPCA(x, self._dim)

        # self.mu = np.array(output[0])
        # self.u = np.array(output[1])
        # PCs = np.array(output[2])
        # self.m = np.array(output[3])

        # return PCs

        self._model = logPCA.logisticPCA(x, self._dim)

    def transform(self, x):
        # explanation:
        # teta = self.m*(2*x - 1)
        # return (teta - self.mu)@self.u #centering the data
        return np.array(logPCA.predict_lpca(self._model, x))


class sklearnPCA(DimmClassBase):
    def __init__(self, n_components=3):
        self.red = PCA(n_components=n_components)

    def fit(self, X, y=None):
        self.red.fit(X, y)

    def transform(self, X):
        return self.red.transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


def tsne(data):
    red = TSNE(n_components=3)
    return red.fit_transform(data)


def pca(data):
    red = PCA(n_components=3)
    return red.fit_transform(data)


def truncated_svd(data):
    red = TruncatedSVD(n_components=50)
    return red.fit_transform(data)


class sklearnAutoEncoder(BaseEstimator, DimmClassBase):
    def __init__(self, n_components=3, mode="regressor"):
        self.n_components = n_components
        if mode == "regressor":
            self.red = MLPRegressor(
                (100, n_components, 100),
                activation="identity",
                random_state=199,
            )
        elif mode == "classifier":
            self.red = MLPClassifier(
                (100, n_components, 100),
                activation="logistic",
                max_iter=10000,
                random_state=199,
            )
        else:
            raise ValueError("%s autoencoder not implemented" % (mode))

    def fit(self, X, y=None):
        self.red.fit(X, X)

    def transform(self, X):
        hidden_layer_sizes = self.red.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)
        layer_units = [X.shape[1]] + hidden_layer_sizes + [self.red.n_outputs_]
        activations = [X]
        for i in range(self.red.n_layers_ - 1):
            activations.append(np.empty((X.shape[0], layer_units[i + 1])))
        self.red._forward_pass(activations)
        return activations[-3]

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


def kpca(data):
    red = KernelPCA(n_components=3)
    return red.fit_transform(data)


def parse_output(X, y):
    output = {}
    for i in range(X.shape[1]):
        output["V%d" % (i)] = X[:, i]
    output["class"] = y
    return output


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_to_file(data, method_name, n_components, train_size, file_prefix):
    df = pd.DataFrame(data=data)
    path = "./preprocessing/reduced_dataset/%s/%d_components/%0.2f/" % (method_name, n_components, train_size)
    try:
        makedirs(path)
    except:
        pass
    df.to_csv(path + file_prefix + ".csv", index=False, sep=";")


def dimm_reduction_worker(
    j, dataset:Dataset=None, component=0, train_size=0.8, batch_size=8, n_epochs=40, early_stopping=5, verbose=False,
):
    X_transformer = LabelEncoder()
    X_transformer.fit([True, False])

    y_transformer = LabelEncoder()
    y_transformer.fit(["source", "sink", "neithernor"])

    save_embeddings_path = "./data/saved_models/"
    methods = {
        "pca": sklearnPCA,
        "logpca": logisticPCA,
        # "autoencoder": sklearnAutoEncoder,
        "triplet": TripletEmbedding,
        # "autoencodertest": functools.partial(
        #     sklearnAutoEncoder, mode="classifier"
        # ),
        "autoencodertorch": AutoEncoder
        # "triplettest": TripletEmbedding,
    }

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

    for method_name in methods.keys():
        sys.stdout.write("\033[K")
        print("%s_%d" % (method_name, component))

        if method_name == "autoencodertorch":
            m = methods[method_name](
                n_components=component,
                batch_size=batch_size,
                n_epochs=n_epochs,
                early_stopping=early_stopping,
                verbose=verbose,
            )
        else:
            m = methods[method_name](n_components=component)

        train_transformed = m.fit_transform(X_train, y_train)
        test_transformed = m.transform(X_test)
        train_output = parse_output(
            train_transformed,
            y_transformer.inverse_transform(y_train.astype(int)),
        )
        test_output = parse_output(
            test_transformed,
            y_transformer.inverse_transform(y_test.astype(int)),
        )

        save_to_file(
            train_output, method_name, component, train_size, "sample_%d_train" % (j)
        )
        save_to_file(
            test_output, method_name, component, train_size, "sample_%d_test" % (j)
        )
        model_path = os.path.join(
            save_embeddings_path, method_name, str(component), str(train_size)
        )

        _file_thread_lock.acquire()

        try:
            os.makedirs(model_path)
        except:
            pass

        _file_thread_lock.release()

        file_name = "sample_%d.model" % (j)
        m.save(os.path.join(model_path, file_name))


def _process_initializer(lock):
    global _file_thread_lock
    _file_thread_lock = lock


def create_embeddings_main(dataset):
    components = [2, 3, 5, 10, 15, 20, 25, 50, 100, 150, 200]
    train_sizes = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    for (component, train_size) in [(i, j) for i in components for j in train_sizes]:
        lock = mp.Lock()
        with mp.Pool(5, initializer=_process_initializer, initargs=(lock, )) as p:
            p.map(
                functools.partial(
                    dimm_reduction_worker,
                    dataset=dataset,
                    batch_size=200,
                    n_epochs=100000,
                    early_stopping=1000,
                    component=component,
                    train_size=train_size,
                    verbose=False,
                ),
                range(30),
            )


if __name__ == "__main__":
    create_embeddings_main(Dataset.create())
