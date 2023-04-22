import os
import re
import math
import json
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class BasicExperiments(object):
    def __init__(
        self,
        models,
        dataset,
        samples_path="./datasetutils/sampled_db/",
        names=None,
        reduction_algorithm=None,
        use_latex=True,
        labels=["source", "sink", "neithernor"],
        experiment_name="default",
    ):
        self.models = models
        self.dataset = dataset
        self.samples_path = samples_path
        self.names = names
        self.reduction_algorithm = reduction_algorithm
        self.use_latex = use_latex
        self.labels = labels
        self._encoder = LabelEncoder()
        self._encoder.fit(labels)
        self._experiment_name = experiment_name
        self._base_predictions_file = "./statistics/%s_models_predictions.txt"
        self._base_accuracies_file = "./statistics/%s_models_accuracies.txt"
        self._preloaded_predictions = None

        self._path_dir = os.path.join(
            "./reports/latex/", self._experiment_name
        )
        if not os.path.isdir(self._path_dir):
            os.mkdir(self._path_dir)

    def run_experiment(self):
        pass

    def to_label(self, predictions):
        return self._encoder.transform(predictions)

    def reverse_label(self, predictions):
        return self._encoder.inverse_transform(predictions)

    def get_output_directory(self):
        return self._path_dir

    def _get_saved_model_prediction(self, prefix, name, i):
        predictions_file_path = os.path.join(
            self._base_predictions_file % (prefix)
        )

        if (
            os.path.isfile(predictions_file_path)
            and self._preloaded_predictions is None
        ):
            predictions_file = open(predictions_file_path, "r")
            self._preloaded_predictions = json.load(predictions_file)

        return self._preloaded_predictions[name][i]

    def get_metrics(self, y, prediction):
        if (
            prediction.dtype == np.int32
            or prediction.dtype == np.int64
            or y.dtype == np.int32
            or y.dtype == np.int64
        ):
            prediction = self.reverse_label(prediction)
            y = self.reverse_label(y)

        labels = np.unique(y)

        a = accuracy_score(y, prediction, sample_weight=None)
        p, r, f1, s = precision_recall_fscore_support(
            y, prediction, sample_weight=None, labels=labels
        )

        report = {}
        for idx, l in enumerate(labels):
            report[l] = {
                "precision": p[idx],
                "recall": r[idx],
                "f1": f1[idx],
                "accuracy": a,
                "support": s[idx],
            }

        p, r, f1, s = precision_recall_fscore_support(
            y, prediction, average="weighted"
        )  # global values
        s = y.shape[0]
        report["global"] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "accuracy": a,
            "support": s,
        }

        return report, prediction

    def beautify_name(self, name):
        name = re.sub(r"CalibratedClassifierCV", "Perceptron", name)
        name = re.sub(r"SGD", "Perceptron", name)
        name = re.sub(r"ABCMeta", "Perceptron", name)
        name = re.sub(r"APriori", "A Priori", name)
        name = re.sub(r"APosteriori", "A Posteriori", name)
        name = re.sub(r"BaggingClassifier", "Bagging", name)
        name = re.sub(r"AdaBoostClassifier", "AdaBoost", name)
        name = re.sub(r"_Classifier", "", name)
        name = re.sub(r"_", " ", name)
        name = re.sub(r"Classifier", "", name)
        name = re.sub(r"SVC", "SVM", name)
        name = re.sub(r"DecisionTree", "Decision Tree", name)
        name = re.sub(r"KNeighbors", "KNN", name)
        name = re.sub(r"RandomForest", "Random Forest", name)
        name = re.sub(r"RotationForest", "Rotation Forest", name)
        name = re.sub(r"SingleBest", "Single Best", name)
        name = re.sub(r"StaticSelection", "Static Selection", name)
        name = re.sub(r"StackedClassifier", "Stacked Classifier", name)
        name = re.sub(r"NB", " NB", name)

        return name

    @staticmethod
    def get_recursive_directories(path):
        sub_dirs = [
            o for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))
        ]
        if len(sub_dirs) > 0:
            all_valid_paths = []
            for d in sub_dirs:
                sub_sub_dirs = [
                    o
                    for o in os.listdir(os.path.join(path, d))
                    if os.path.isdir(os.path.join(path, d, o))
                ]
                paths = [
                    {d + "_" + d1[:-11]: os.path.join(path, d, d1 + "/")}
                    for d1 in sub_sub_dirs
                ]
                all_valid_paths.append(paths)
            return all_valid_paths, True

        return [[{path: path}]], False


class ClassMetric(object):
    def __init__(self, class_label):
        self.class_label = class_label
        self.precisions = np.array([])
        self.recalls = np.array([])
        self.accuracies = np.array([])
        self.f1s = np.array([])
        self.supports = np.array([])

    def add_values(self, values):
        p = np.array([values["precision"]])
        r = np.array([values["recall"]])
        a = np.array([values["accuracy"]])
        f1 = np.array([values["f1"]])
        sup = np.array([values["support"]])

        self.precisions = np.append(self.precisions, p)
        self.recalls = np.append(self.recalls, r)
        self.accuracies = np.append(self.accuracies, a)
        self.f1s = np.append(self.f1s, f1)
        self.supports = np.append(self.supports, sup)

    def weighted_avg_and_std(self, values, weights):
        """
        Return the weighted average and standard deviation.

        values, weights -- Numpy ndarrays with the same shape.
        """
        average = np.average(values, weights=weights)
        # Fast and numerically precise:
        variance = np.average((values - average) ** 2, weights=weights)
        return (average, math.sqrt(variance))

    def report(self):
        mean_p, std_p = self.weighted_avg_and_std(
            self.precisions, self.supports
        )
        mean_r, std_r = self.weighted_avg_and_std(self.recalls, self.supports)
        mean_a, std_a = self.weighted_avg_and_std(
            self.accuracies, self.supports
        )
        mean_f1, std_f1 = self.weighted_avg_and_std(self.f1s, self.supports)

        self.__metric_list = [
            mean_p,
            std_p,
            mean_r,
            std_r,
            mean_f1,
            std_f1,
            mean_a,
            std_a,
        ]

        self.__metric = {
            "precision": {"mean": mean_p, "std": std_p},
            "recall": {"mean": mean_r, "std": std_r},
            "f1": {"mean": mean_f1, "std": std_f1},
            "accuracy": {"mean": mean_a, "std": std_a},
        }

        return self.__metric

    def to_list(self):
        self.report()
        return self.__metric_list
