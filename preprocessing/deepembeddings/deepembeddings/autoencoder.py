from tabnanny import verbose
import torch
import pickle

import numpy as np
import pandas as pd

import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from orderedset import OrderedSet

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from torchvision import transforms

from torch.optim import lr_scheduler
from torch.autograd import Variable

from deepembeddings.utils.trainer import fit
from deepembeddings.utils.datasets import SimpleDataset
from deepembeddings.utils.networks import AutoEncoderEmbedding
from deepembeddings.utils.losses import AutoEncoderLoss


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(199)
torch.cuda.manual_seed_all(199)
np.random.seed(199)

cuda = False


def use_cuda(value):
    global cuda
    cuda = value


class AutoEncoder(BaseEstimator):
    def __init__(
        self,
        n_components=3,
        lr=1e-3,
        n_epochs=40,
        batch_size=8,
        n_workers=1,
        architecture=None,
        verbose=False,
        activation=nn.Sigmoid,
        decoder_loss=0.0,
        decoder_architecture=None,
        early_stopping=5,
    ):
        self.architecture = architecture
        self.decoder_architecture = decoder_architecture
        self.decoder_loss = decoder_loss
        self.model = None
        self.n_components = n_components
        self.loss_fn = AutoEncoderLoss()
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.log_interval = 500
        self.n_workers = n_workers
        self.activation = activation
        self.early_stopping = early_stopping

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        obj = None
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight)
            m.bias.data.fill_(0.01)

    def fit(self, X, y):
        if self.architecture is None:
            self.architecture = nn.Sequential(
                OrderedDict(
                    [
                        ("input", nn.Linear(X.shape[1], self.n_components)),
                        ("input_act", self.activation()),
                        (
                            "bottleneck",
                            nn.Linear(self.n_components, self.n_components),
                        ),
                        ("bottleneck_act", self.activation()),
                        ("output", nn.Linear(self.n_components, X.shape[1])),
                        ("output_act", nn.Sigmoid()),
                    ]
                )
            )
            self.architecture.apply(self.init_weights)
            self.model = AutoEncoderEmbedding(self.architecture)

        if cuda:
            self.model.cuda()

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        X_train, X_val, _, _ = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=199
        )

        triplet_train_dataset = SimpleDataset(
            X_train, X_train, is_training=True
        )
        triplet_test_dataset = SimpleDataset(X_val, X_val, is_training=False)

        kwargs = (
            {"num_workers": self.n_workers, "pin_memory": True} if cuda else {}
        )

        triplet_train_loader = torch.utils.data.DataLoader(
            triplet_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            **kwargs,
        )
        triplet_test_loader = torch.utils.data.DataLoader(
            triplet_test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            **kwargs,
        )

        fit(
            triplet_train_loader,
            triplet_test_loader,
            self.model,
            self.loss_fn,
            optimizer,
            None,
            self.n_epochs,
            cuda,
            self.log_interval,
            early_stopping=self.early_stopping,
            verbose=self.verbose,
        )

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if cuda:
            return (
                self.model.get_embedding(torch.Tensor(X).float().cuda())
                .data.cpu()
                .numpy()
            )
        else:
            return (
                self.model.get_embedding(torch.Tensor(X).float())
                .data.cpu()
                .numpy()
            )

    def compare(self, u, v):
        u = self.transform(u)
        v = self.transform(v)
        return np.sqrt(((u - v) ** 2).sum())

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
