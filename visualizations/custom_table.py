import os
import sys
import json
import argparse
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

sys.path.append("..")

from datasetutils.dataset import Dataset
from experiments.base_experiments import ClassMetric

from utils.latex import (
    print_latex_table_header,
    print_latex_table_entry,
    print_latex_table_footer,
)

labels = ["source", "sink", "neithernor"]
_encoder = LabelEncoder()
_encoder.fit(labels)


def get_report(y_pred, y):
    global _encoder

    if (
        y_pred.dtype == np.int32
        or y_pred.dtype == np.int64
        or y.dtype == np.int32
        or y.dtype == np.int64
    ):
        y_pred = _encoder.inverse_transform(y_pred)
        y = _encoder.inverse_transform(y)

    labels = np.unique(y)

    a = accuracy_score(y, y_pred, sample_weight=None)
    p, r, f1, s = precision_recall_fscore_support(
        y, y_pred, sample_weight=None, labels=labels
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
        y, y_pred, average="weighted"
    )  # global values
    s = y.shape[0]
    report["global"] = {
        "precision": p,
        "recall": r,
        "f1": f1,
        "accuracy": a,
        "support": s,
    }

    return report


def merge_predictions(data):
    model_predictions = {}
    for k in data:
        for m in data[k]:
            model_predictions[m] = data[k][m]
    return model_predictions


def main(name):
    dataset = Dataset("../datasetutils/db/methods.csv")
    dataset.load_samples("../datasetutils/sampled_db", recursive_reading=False)

    file = "../statistics/default_models_predictions.txt"
    data = json.load(open(file, "r"))

    # model_predictions = merge_predictions(data)
    model_predictions = data["MCS Bagging Perceptron"]

    output_dir = os.path.join("../reports/latex/")

    latex_table = open(
        os.path.join(output_dir, "%s_results.tex" % (name)),
        "w+",
    )
    print_latex_table_header(latex_table)

    for model_name in model_predictions:
        if "oracle" in model_name.lower():
            continue
        if "with" in model_name.lower():
            continue
        print(model_name)
        source = ClassMetric("source")
        sink = ClassMetric("sink")
        neithernor = ClassMetric("neithernor")
        global_metrics = ClassMetric("global")
        classes_metrics = [source, sink, neithernor]
        for i in range(30):
            _, _, _, y_test = dataset.get_sampled(i)
            y_pred = np.array(model_predictions[model_name][i])

            report = get_report(y_pred, y_test)

            for c in classes_metrics:
                c.add_values(report[c.class_label])
            global_metrics.add_values(report["global"])

        print_latex_table_entry(
            latex_table,
            model_name,
            global_metrics,
            new_mode=True,
        )

    print_latex_table_footer(
        latex_table, "the best classifiers", new_mode=True
    )
    latex_table.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n",
        "--name",
        action="store",
        dest="name",
        required=True,
        help="Name of the output table.",
    )

    args = parser.parse_args()
    main(args.name)
