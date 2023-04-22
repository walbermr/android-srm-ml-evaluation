import pandas as pd
import re, os

from arfftocsv.arffToCsv import toCsv as arffToCSV

from sklearn.model_selection import train_test_split

from shutil import copy

from glob import glob

from os import listdir
from os.path import isfile, join

from sklearn.preprocessing import LabelEncoder


class Sample(object):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


class Dataset(object):
    def __init__(
        self,
        path_name=None,
        crude=False,
        update=False,
        load_sampled=False,
        verbose=False,
    ):
        """Receive dataset path and return a Dataset class containing pandas and list dataset
        : param path_name: path to the dataset

        : return: return a Dataset object
        """
        self.dataframe = {}
        self.data = []
        self.target = []
        self.verbose = verbose

        self.X_transformer = LabelEncoder()
        self.y_transformer = LabelEncoder()

        self.X_transformer.fit([True, False])
        self.y_transformer.fit(["source", "sink", "neithernor"])

        if update:
            for f in glob(r"./SuSi/dataset/train/*.arff"):
                copy(f, "./datasetutils/crude_db")
                print("copy %s" % (f))

        train_file = []

        if crude:
            if os.path.isfile(path_name):
                print("Reading file %s. " % (path_name), end="")
                f_lines = open(path_name)
                arffToCSV(f_lines, "./datasetutils/db/temp")
                df = pd.read_csv(
                    "./datasetutils/db/temp.csv", sep=";", low_memory=False
                )
                self.keys = df.keys
                self.feature_keys = self.keys()[:-2]

                df, fixes = self.fix_errors(df)
                if fixes > 0:
                    print("Fixed removing %d entries. " % (fixes), end="")
                else:
                    print("Nothing to be done. ", end="")

                print("File size: %d." % (df.shape[0]))
                train_file.append(df)
            else:
                files = [
                    f
                    for f in listdir(path_name)
                    if isfile(join(path_name, f)) and f != "susi.arff"
                ]
                for f in files:
                    print("Reading file %s. " % (f), end="")
                    f_lines = open(path_name + f)
                    arffToCSV(f_lines, "./datasetutils/db/temp")
                    df = pd.read_csv(
                        "./datasetutils/db/temp.csv", sep=";", low_memory=False
                    )
                    self.keys = df.keys
                    self.feature_keys = self.keys()[:-2]

                    df, fixes = self.fix_errors(df)
                    if fixes > 0:
                        print("Fixed removing %d entries. " % (fixes), end="")
                    else:
                        print("Nothing to be done. ", end="")

                    print("File size: %d." % (df.shape[0]))

                    train_file.append(df)
            self.dataframe["original"] = pd.concat(train_file, sort=False)
            self.dataframe["original"] = self.dataframe["original"].dropna()
            self.dataframe["original"] = self.dataframe[
                "original"
            ].drop_duplicates(subset=self.dataframe["original"].keys())
            # self.dataframe['train'] = pd.concat(train_file).drop_duplicates(subset=['id'])
            self.dataframe["original"].to_csv(
                "./datasetutils/db/methods.csv", sep=";", index=False
            )
        else:
            self.dataframe["original"] = pd.read_csv(
                path_name, sep=";", low_memory=False
            )

        self.keys = self.dataframe["original"].keys
        self.feature_keys = self.keys()[:-2]

        self.data, self.target = self.__get_features_targets(
            self.dataframe["original"], self.feature_keys, "class"
        )

        self.data = self.data.apply(self.X_transformer.transform)
        self.target = self.y_transformer.transform(self.target)

        print("Dataframe size: %d" % (self.dataframe["original"].shape[0]))
        print("Dataframe features: %d" % (self.dataframe["original"].shape[1]))
        print(self.dataframe["original"]["class"].value_counts())

    @staticmethod
    def create(csv_path="./datasetutils/db/methods.csv", crude_path="./datasetutils/crude_db/", sample_path="./datasetutils/sampled_db/"):
        if os.path.isfile(csv_path):
            dataset = Dataset(
                csv_path
            )  # use already preprocessed db
        else:
            dataset = Dataset(crude_path, crude=True)  # get crude db
            dataset.make_samples()

        dataset.load_samples(sample_path, recursive_reading=False)  # remake the divisions
        return dataset
    
    def __get_features_targets(self, full, feature_keys, target_key):
        return full[feature_keys], full[target_key]

    def split_train_test(self, data):
        train, test = train_test_split(data, test_size=0.2)
        X_train, y_train = self.__get_features_targets(
            train, self.feature_keys, "class"
        )
        X_test, y_test = self.__get_features_targets(
            test, self.feature_keys, "class"
        )

        return X_train, X_test, y_train, y_test

    def fix_errors(self, df):
        """Fix the a dataframe droping rows with unkown features
        : param dataframe: the dataframe to be fixed
        : return: the fixed dataframe
        """
        print("Fixing. ", end="")
        fixes = 0
        for index, method in df.iterrows():
            for f in self.feature_keys:
                if method[f] == "?":
                    fixes += 1
                    df.drop(index, inplace=True)
                    break
                if df.loc[index, f] == "false":
                    df.loc[index, f] = False
                elif df.loc[index, f] == "true":
                    df.loc[index, f] = True

        return df, fixes

    def load_samples(self, path, samples=30, recursive_reading=False):
        self.dataframe["samples"] = []
        self.sample_size = samples
        file_prefix = "sample_"
        file_extension = ".csv"

        for i in range(0, samples):
            if self.verbose:
                print("Loading sample %d... " % (i), end="")
            train = pd.read_csv(
                os.path.join(path, file_prefix)
                + str(i)
                + "_train"
                + file_extension,
                sep=";",
                low_memory=False,
            )
            test = pd.read_csv(
                os.path.join(path, file_prefix)
                + str(i)
                + "_test"
                + file_extension,
                sep=";",
                low_memory=False,
            )

            if recursive_reading:
                self.feature_keys = train.keys()[:-1]
            else:
                self.feature_keys = train.keys()[:-2]

            X_train, y_train = self.__get_features_targets(
                train, self.feature_keys, "class"
            )
            X_test, y_test = self.__get_features_targets(
                test, self.feature_keys, "class"
            )

            if recursive_reading:
                sample = Sample(
                    X_train,
                    self.y_transformer.transform(y_train),
                    X_test,
                    self.y_transformer.transform(y_test),
                )
            else:
                sample = Sample(
                    X_train.apply(self.X_transformer.transform),
                    self.y_transformer.transform(y_train),
                    X_test.apply(self.X_transformer.transform),
                    self.y_transformer.transform(y_test),
                )

            self.dataframe["samples"].append(sample)

            if self.verbose:
                print("Loaded.")

    def make_samples(self, path, samples=30, test_size=0.2):
        file_prefix = "sample_"
        file_extension = ".csv"

        print("Creating samples...", end="")

        for i in range(0, samples):
            train, test = train_test_split(
                self.dataframe["original"],
                test_size=test_size,
                stratify=self.dataframe["original"]["class"],
            )

            train.to_csv(
                path + file_prefix + str(i) + "_train" + file_extension,
                sep=";",
                index=False,
            )
            test.to_csv(
                path + file_prefix + str(i) + "_test" + file_extension,
                sep=";",
                index=False,
            )
        print("Done.")

        self.load_samples(path, samples=samples)

    def get_sampled(self, i):
        return (
            self.dataframe["samples"][i].X_train,
            self.dataframe["samples"][i].y_train,
            self.dataframe["samples"][i].X_test,
            self.dataframe["samples"][i].y_test,
        )
