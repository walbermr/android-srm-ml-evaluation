import torch

import numpy as np

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
from deepembeddings.utils.datasets import SiameseBinaryDataset, BinaryLoader
from deepembeddings.utils.networks import MLPEmbedding, SiameseNet
from deepembeddings.utils.losses import ContrastiveLoss


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(199)
torch.cuda.manual_seed_all(199)
np.random.seed(199)

cuda = torch.cuda.is_available()


class SiameseEmbedding(BaseEstimator):
    def __init__(
        self,
        margin=1.0,
        lr=1e-3,
        n_epochs=100,
        batch_size=64,
        n_workers=1,
        architecture=nn.Sequential(
            nn.Linear(213, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 3),
        ),
        verbose=False,
    ):

        self.margin = margin
        self.embedding_net = MLPEmbedding(architecture)
        self.model = SiameseNet(self.embedding_net)
        self.loss_fn = ContrastiveLoss(self.margin)
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer, 8, gamma=0.1, last_epoch=-1
        )
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.log_interval = 500
        self.n_workers = n_workers

        if cuda:
            self.model.cuda()

    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=199
        )

        triplet_train_dataset = SiameseBinaryDataset(
            X_train, y_train, is_training=True
        )  # Returns triplets of images
        train_loader = BinaryLoader(X_train, y_train)
        triplet_test_dataset = SiameseBinaryDataset(X_val, y_val, is_training=False)
        val_loader = BinaryLoader(X_val, y_val)

        kwargs = {"num_workers": self.n_workers, "pin_memory": True} if cuda else {}

        triplet_train_loader = torch.utils.data.DataLoader(
            triplet_train_dataset, batch_size=self.batch_size, shuffle=True, **kwargs
        )
        triplet_test_loader = torch.utils.data.DataLoader(
            triplet_test_dataset, batch_size=self.batch_size, shuffle=False, **kwargs
        )

        fit(
            triplet_train_loader,
            triplet_test_loader,
            self.model,
            self.loss_fn,
            self.optimizer,
            self.scheduler,
            self.n_epochs,
            cuda,
            self.log_interval,
            early_stopping=5,
        )

    def transform(self, X):
        return self.model.get_embedding(torch.Tensor(X)).data.cpu().numpy()

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
