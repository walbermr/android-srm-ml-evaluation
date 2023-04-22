import os
import sys
from tkinter.ttk import Separator

from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import (
    FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super().__init__()
        self.fig = plt.Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvas(self.fig)
        self.axes = self.fig.add_subplot(111, projection="3d")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.canvas)

        self.parent = parent

    def compute_figure(self, signal):
        dataset = self.parent.dataset
        self.axes.clear()

        if dataset is None:
            self.canvas.draw()
            return

        classes = ["source", "sink", "neithernor"]
        colors = ["r", "g", "b"]

        for cl, c in zip(classes, colors):
            selection = dataset.loc[dataset["class"] == cl].drop(
                columns=["class"]
            )
            try:
                xs = selection["V1"]
                ys = selection["V2"]
                zs = selection["V3"]
            except:
                xs = selection["V0"]
                ys = selection["V1"]
                zs = selection["V2"]
            self.axes.scatter(xs, ys, zs, c=c, label=cl)

        self.axes.set_xlabel("Component 1")
        self.axes.set_ylabel("Component 2")
        self.axes.set_zlabel("Component 3")
        self.axes.legend()

        self.canvas.draw()


class Window(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.file_menu = QtWidgets.QMenu("&File", self)
        self.file_menu.addAction(
            "&Quit", self.fileQuit, QtCore.Qt.CTRL + QtCore.Qt.Key_Q
        )
        self.menuBar().addMenu(self.file_menu)

        self.dataset = None
        self.plot_widget = PlotWidget(parent=self, width=10, height=8)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.plot_widget.canvas, self)

        self.name_path_pair = {
            "PCA": "pca.csv",
            "Logistic PCA": "logpca.csv",
            "Truncated SVD": "truncated_svd.csv",
            "TSNE": "tsne.csv",
            "Autoencoder": "autoencoder.csv",
            "Autoencoder Test": "autoencodertest.csv",
            "Autoencoder Torch": "autoencodertorch.csv",
            # 'KPCA': 'kpca.csv',
            "Logistic SVD": "logsvd.csv",
            "Triplet Loss": "triplet.csv",
            "Triplet Loss Test": "triplettest.csv",
        }

        self.drop_down = QtWidgets.QComboBox(self)
        self.drop_down.addItem("--None--")
        for k in self.name_path_pair.keys():
            self.drop_down.addItem(k)
        self.drop_down.move(400, 8)

        self.drop_down.activated[str].connect(self.change_dataset)

        self.main_widget = QtWidgets.QWidget(self)
        main_layout = QtWidgets.QVBoxLayout(self.main_widget)
        main_layout.addWidget(self.plot_widget)

        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.drop_down)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        # Generate the chart for t=0 when the window is openned
        self.plot_widget.compute_figure(self.dataset)

    def change_dataset(self, text):
        try:
            dataset_file = os.path.join(
                "../preprocessing/transformations", self.name_path_pair[text]
            )
            self.dataset = pd.read_csv(dataset_file, sep=";")
        except:
            if text == "--None--":
                self.dataset = None
                self.plot_widget.compute_figure(1)
            else:
                self.statusBar().showMessage(
                    "%s dataset does not exist" % (text.lower()), 2000
                )

        self.plot_widget.compute_figure(1)

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())
