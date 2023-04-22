import torch
import numpy as np


class EarlyStopping(object):
    def __init__(self, mode="min", min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

    def step(self, metrics):
        if self.patience == 0:
            return False

        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if self.patience == 0:
            self.is_better = lambda a, b: True
            return

        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (
                    best * min_delta / 100
                )
            if mode == "max":
                self.is_better = lambda a, best: a > best + (
                    best * min_delta / 100
                )


def fit(
    train_loader,
    val_loader,
    model,
    loss_fn,
    optimizer,
    scheduler,
    n_epochs,
    cuda,
    log_interval,
    metrics=[],
    start_epoch=0,
    early_stopping=0,
    verbose=False,
):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    es = EarlyStopping(patience=early_stopping)

    for epoch in range(0, start_epoch):
        if scheduler is not None:
            scheduler.step()

    for epoch in range(start_epoch, n_epochs):

        # Train stage
        train_loss, metrics = train_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            cuda,
            log_interval,
            metrics,
        )

        if scheduler is not None:
            scheduler.step()

        message = "Epoch: {}/{}. Train set: Average loss: {:.4f}".format(
            epoch + 1, n_epochs, train_loss
        )
        for metric in metrics:
            message += "\t{}: {}".format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(
            val_loader, model, loss_fn, cuda, metrics
        )
        val_loss /= len(val_loader)

        message += (
            "\nEpoch: {}/{}. Validation set: Average loss: {:.4f}".format(
                epoch + 1, n_epochs, val_loss
            )
        )
        for metric in metrics:
            message += "\t{}: {}".format(metric.name(), metric.value())

        if es.step(val_loss):
            break

        if verbose:
            print(message)


def train_epoch(
    train_loader,
    model,
    loss_fn,
    optimizer,
    cuda,
    log_interval,
    metrics,
    verbose=False,
):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        data = tuple(d.float() for d in data)
        if target is not None:
            target = target.float()

        if cuda:
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = (
            loss_outputs[0]
            if type(loss_outputs) in (tuple, list)
            else loss_outputs
        )
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = "Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                batch_idx * len(data[0]),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                np.mean(losses),
            )
            for metric in metrics:
                message += "\t{}: {}".format(metric.name(), metric.value())

            if verbose:
                print(message)
            losses = []

    total_loss /= batch_idx + 1
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)

            data = tuple(d.float() for d in data)
            if target is not None:
                target = target.float()

            if cuda:
                data = data.cuda()
                target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = (
                loss_outputs[0]
                if type(loss_outputs) in (tuple, list)
                else loss_outputs
            )
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics
