from sched import scheduler
import torch
import pickle

import numpy as np
import pandas as pd

import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

from orderedset import OrderedSet

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from torchvision import transforms

from torch.optim import lr_scheduler
from torch.autograd import Variable

from deepembeddings.utils.trainer import fit
from deepembeddings.utils.datasets import TripletBinaryDataset, BinaryLoader
from deepembeddings.utils.networks import MLPEmbedding, TripletNet
from deepembeddings.utils.losses import TripletLoss


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(199)
torch.cuda.manual_seed_all(199)
np.random.seed(199)

cuda = False


def use_cuda(value):
    global cuda
    cuda = value


class TripletEmbedding(BaseEstimator):
    def __init__(
        self,
        n_components=3,
        margin=1.0,
        lr=1e-3,
        n_epochs=40,
        batch_size=8,
        n_workers=1,
        architecture=None,
        verbose=False,
        activation=nn.ReLU,
        decoder_loss=0.0,
        decoder_architecture=None,
        use_scheduler=True,
    ):
        self.architecture = architecture
        self.decoder_architecture = decoder_architecture
        self.decoder_loss = decoder_loss
        self.embedding_net = None
        self.model = None
        self.n_components = n_components
        self.margin = margin
        self.loss_fn = TripletLoss(self.margin)
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.log_interval = 500
        self.n_workers = n_workers
        self.activation = activation
        self.use_scheduler = use_scheduler

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        obj = None
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj

    def fit(self, X, y):
        if self.architecture is None:
            self.architecture = nn.Sequential(
                nn.Linear(X.shape[1], 100),
                self.activation(),
                nn.Linear(100, 100),
                self.activation(),
                nn.Linear(100, self.n_components),
            )
            self.embedding_net = MLPEmbedding(self.architecture)
            self.model = TripletNet(self.embedding_net)

            # if self.decoder_architecture is None and self.decoder_loss != 0:
            #     self.decoder_architecture = nn.Sequential(nn.Linear(self.n_components, 100),
            #                 self.activation,
            #                 nn.Linear(100, 100),
            #                 self.activation,
            #                 nn.Linear(100, X.shape[1]),
            #                 self.activation
            #                 )

        if cuda:
            self.model.cuda()

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        if self.use_scheduler:
            scheduler = lr_scheduler.StepLR(
                optimizer, 8, gamma=0.1, last_epoch=-1
            )
        else:
            scheduler = None

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=199
        )

        triplet_train_dataset = TripletBinaryDataset(
            X_train, y_train, is_training=True
        )  # Returns triplets of images
        train_loader = BinaryLoader(X_train, y_train)
        triplet_test_dataset = TripletBinaryDataset(
            X_val, y_val, is_training=False
        )
        val_loader = BinaryLoader(X_val, y_val)

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
            scheduler,
            self.n_epochs,
            cuda,
            self.log_interval,
            early_stopping=5,
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
