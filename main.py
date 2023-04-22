import os

from datasetutils.dataset import Dataset
from preprocessing.dimm_reduction import create_embeddings_main
from experiments.evaluation import StratificationExperiment
from experiments.hardness_tracking import track_hardness, hardness_statistics
from experiments.ablation import ablation
from utils.utils import get_dimensions

import argparse
import models

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(action="ignore", category=ConvergenceWarning)


def main(
    classifiers,
    reduction_algorithms,
    experiment_name,
    group,
    train_classifiers,
):
    if os.path.isfile("./datasetutils/db/methods.csv"):
        dataset = Dataset(
            "./datasetutils/db/methods.csv"
        )  # use already preprocessed db
    else:
        dataset = Dataset(
            "./datasetutils/crude_db/", crude=True
        )  # get crude db
        dataset.make_samples("./datasetutils/sampled_db/")

    if reduction_algorithms:
        samples_path = "./preprocessing/reduced_dataset/"
    else:
        samples_path = "./datasetutils/sampled_db/"

    if group is not None:
        classifiers = {group: classifiers[group]}

    experiment = StratificationExperiment(
        classifiers,
        dataset,
        samples_path=samples_path,
        use_latex=True,
        experiment_name=experiment_name,
        train_classifiers=train_classifiers,
    )

    experiment.run_experiment()


def hardness_track(train):
    dataset = Dataset(
        "./datasetutils/db/methods.csv"
    )  # use already preprocessed db
    samples_path = "./datasetutils/sampled_db/"
    embedding_dimmensions = get_dimensions("./data/saved_models")

    hardness, pairwise_hardness = track_hardness(
        dataset,
        samples_path=samples_path,
        train=train,
        embedding_dimmensions=embedding_dimmensions,
        models=models.EMBEDDINGS,
    )
    model_a = ["triplet"]
    models_b = [k for k in hardness.keys() if k != "triplet"]

    print(hardness.keys())

    for name_a, name_b in zip(model_a * len(models_b), models_b):
        hardness_statistics(name_a, name_b, pairwise_hardness)


def create_embeddings():
    create_embeddings_main(Dataset.create())


def ablation_experiment(train, experiment_name):
    dataset = Dataset(
        "./datasetutils/db/methods.csv"
    )  # use already preprocessed db
    return ablation(dataset, train=train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        action="store",
        dest="mode",
        default="eval",
        help="Evaluation mode, choose from:\n-eval: Simple model evaluation.\n-track | hardness_track: Track hardness of missclassified instances.",
    )
    parser.add_argument(
        "-c",
        "--classifiers",
        action="store",
        dest="classifiers",
        default="default",
        help="Choose which classifiers to use:\n-default: Select classifiers with default parameters.\n-grid: Select grid optimized parameters.",
    )
    parser.add_argument(
        "-r",
        "--reduction",
        action="store",
        dest="reduction",
        default=0,
        help="Choose if use reduction algorithms. 1: True | 0: False.",
    )
    parser.add_argument(
        "-t_e",
        "--train_embeddings",
        action="store",
        dest="train_embeddings",
        default=1,
        help="This argument is used to change between train or load embedding algorithms.\n1- Train embeddings, this should be used when some modification\
        is used in the embeddings. 0- Load embeddings using the embeddings_path argument.",
    )
    parser.add_argument(
        "-n",
        "--name",
        action="store",
        dest="name",
        required=False,
        default="default",
        help="Use this argument to create a custom name for the experiment, this will create a directoriy to organize the experiments executed.",
    )
    parser.add_argument(
        "-g",
        "--group",
        action="store",
        dest="group",
        default=None,
        help="Control which classifiers are going to be used in the test.",
    )
    # IMPLEMENT EMBEDDING LOCATION SELECTION

    args = parser.parse_args()

    args.reduction = bool(int(args.reduction))
    __train_embeddings = bool(int(args.train_embeddings))

    if args.classifiers == "default":
        classifiers = models.DEFAULT_CLASSIFIERS
    elif args.classifiers == "grid":
        classifiers = models.GRID_CLASSIFIERS
    else:
        print("Invalid mode.")
        exit(1)

    if args.mode == "eval":
        main(
            classifiers,
            args.reduction,
            args.name,
            args.group,
            bool(__train_embeddings),
        )
    elif args.mode == "hardness_track" or args.mode == "track":
        hardness_track(bool(__train_embeddings))
    elif args.mode == "ablation":
        ablation_experiment(bool(__train_embeddings), args.name)
    elif args.mode == "create_embeddings":
        create_embeddings()
    else:
        print("Invalid mode.")
        exit(1)
