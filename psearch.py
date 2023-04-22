from datasetutils.dataset import Dataset
from evaluation import strat

from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import make_scorer
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    GridSearchCV,
    cross_val_score,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.base import clone

from xgboost.sklearn import XGBClassifier

from deslib.dcs.ola import OLA
from deslib.dcs.mcb import MCB
from deslib.des.des_p import DESP
from deslib.des.knora_u import KNORAU
from deslib.des.knora_e import KNORAE
from deslib.des.meta_des import METADES
from deslib.util.sgh import SGH

from deslib.static import SingleBest, StaticSelection, Oracle, StackedClassifier

from copy import deepcopy

import json, os, re

import numpy as np

from tqdm import tqdm

from joblib import Parallel, delayed


def gridRun(classifier, params, metric, X_train, y_train, X_test, y_test):
    workers = os.cpu_count()

    search = GridSearchCV(
        classifier,
        param_grid=params,
        n_jobs=workers,
        cv=5,
        iid=False,
        scoring="accuracy",
        verbose=True,
    )

    search.fit(X_train, y_train)

    best_params = search.best_params_
    print(best_params)

    prediction_t = []
    prediction = []

    try:
        prediction_t = search.predict(X_train)
        prediction = search.predict(X_test)
    except expression as identifier:
        prediction_t = search.predict(X_train, y_train)
        prediction = search.predict(X_test, y_test)

    labels = np.unique(prediction_t)

    p_t, r_t, f1_t, _ = precision_recall_fscore_support(
        y_train, prediction_t, average="weighted"
    )
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, prediction, average="weighted"
    )

    print("------ Grid run ------")
    print("TRAIN")
    print("precision: %.4f" % (p_t))
    print("recall: %.4f" % (r_t))
    print("f1: %.4f" % (f1_t))
    print("accuracy: %.4f" % (metric(y_train, search.predict(X_train))))
    print("\n\n")
    print("TEST")
    print("precision: %.4f" % (p))
    print("recall: %.4f" % (r))
    print("f1: %.4f" % (f1))
    print("accuracy (test): %.4f" % (metric(y_test, prediction)))
    print("----------------------")


def complete_search(
    dataset, classifiers, params, metric, X_train, y_train, X_test, y_test, p_rand=0.1
):

    if len(classifiers) != len(params):
        raise Exception("Classifiers and parameters size mismatch.")

    workers = os.cpu_count()  # number of cores to work on random search

    score = make_scorer(metric, greater_is_better=True)

    for i in range(len(classifiers)):

        c = classifiers[i]
        p = params[i]
        constraints = {}

        for k in p:
            if type(p[k]) != type(str()):
                if type(p[k]) == list and len(p[k]) > 1 and type(p[k][1]) == type(type):
                    constraints[k] = p[k][1]
                    p[k] = p[k][0]

        total_combinations = 1

        for k in p:
            total_combinations *= len(p[k])

        print(
            "Total combinations %d. Using %0.2f%% in random search."
            % (total_combinations, p_rand * 100)
        )

        search = RandomizedSearchCV(
            c,
            param_distributions=p,
            n_iter=(total_combinations * p_rand),
            n_jobs=workers,
            cv=5,
            random_state=199,
            scoring=score,
            verbose=True,
            iid=False,
        )

        search.fit(X_train, y_train)

        best_params = search.best_params_
        print(best_params)

        print("----- Random run -----")
        print(
            "%s (train): %.4f"
            % (metric.__name__, metric(y_train, search.predict(X_train)))
        )
        print(
            "%s (test): %.4f"
            % (metric.__name__, metric(y_test, search.predict(X_test)))
        )
        print("----------------------\n\n")

        new_params = {}

        # clear unused params
        if "kernel" in best_params.keys():
            if best_params["kernel"] != "poly" or best_params["kernel"] != "sigmoid":
                if "coef0" in best_params.keys():
                    del best_params["coef0"]
            if best_params["kernel"] == "linear":
                if "gamma" in best_params.keys():
                    del best_params["gamma"]

        # create new parameters range for numerical ones
        for k in best_params.keys():
            if type(best_params[k]) == type(str()) or type(best_params[k]) == type(
                BaggingClassifier()
            ):
                new_params[k] = [best_params[k]]

            else:
                param = best_params[k]
                delta = param / 10

                interval = (p[k][1] - p[k][0]) / 100

                if k in constraints:
                    if constraints[k] == int:
                        interval = int(round(interval))
                        delta = int(round(delta))
                        if interval == 0:
                            interval = 1

                new_params[k] = [
                    l for l in range(param - delta, param + delta, interval)
                ]
                if len(new_params[k]) == 0:
                    new_params[k] = [l for l in range(param, param + delta, interval)]
                if len(new_params[k]) == 0:
                    new_params[k] = [param]

        gridRun(c, p, metric, X_train, y_train, X_test, y_test)


def regulator(i):
    s = re.sub(r",|;|<|>|'|\[|\]|\\", "", i)
    return s


def monolithic_search(X_train, y_train, X_test, y_test):
    models = [
        DecisionTreeClassifier(random_state=199),
        # # MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True),
        # # BernoulliNB(binarize=None),
        # GaussianNB(var_smoothing=1e-9),
        # KNeighborsClassifier()
        # SVC()
        # MLPClassifier(max_iter=1000),
        # RandomForestClassifier(),
        # XGBClassifier()
    ]

    params = [
        {
            "criterion": ["gini", "entropy"],
            "splitter": ["best", "random"],
            "max_depth": [np.arange(2, 100, 5), int],
        },
        # # {
        # #     'alpha': np.arange(0, 10, 1)
        # # }, #multinomial
        # # {
        # #     'alpha': np.arange(0, 10, 1)
        # # },
        # {
        #     'var_smoothing': np.arange(1e-10, 1e-9, 1e-10)
        # }, #gaussian nb
        # {
        #     'n_neighbors': [np.arange(1, 200, 10), int],
        #     'weights': ['uniform', 'distance'],
        #     'algorithm': ['ball_tree', 'kd_tree', 'brute']
        # }
        # {
        #     'C': np.arange(1, 101, 1),
        #     'gamma': np.arange(0, 100, 0.5),
        #     'kernel': ['linear', 'rbf', 'sigmoid']
        # }
        # {
        #     'hidden_layer_sizes': [np.arange(1, 1000, 10), int],
        #     'activation': ['identity', 'logistic', 'tanh', 'relu'],
        #     'solver': ['lbfgs', 'sgd', 'adam']
        # }
        # {
        #     'n_estimators': [np.arange(1, 1000, 10), int],
        #     'criterion': ['gini', 'entropy'],
        # },
        # {
        #     'n_estimators': [np.arange(1, 1000, 10), int],
        #     'max_depth': [np.arange(1, 10, 1), int]
        # }
    ]

    complete_search(
        dataset,
        models,
        params,
        accuracy_score,
        X_train,
        y_train,
        X_test,
        y_test,
        p_rand=0.5,
    )


def ensemble_search(X_train, y_train, X_test, y_test):
    pool_dt = (
        [
            AdaBoostClassifier(
                DecisionTreeClassifier(random_state=199),
                n_estimators=i,
                random_state=199,
                algorithm="SAMME",
            )
            for i in range(10, 160, 10)
        ]
        + [
            BaggingClassifier(
                DecisionTreeClassifier(random_state=199),
                n_estimators=i,
                random_state=199,
            )
            for i in range(10, 160, 10)
        ]
        + [
            SGH(base_estimator=DecisionTreeClassifier(random_state=199), n_estimators=i)
            for i in range(10, 160, 10)
        ]
    )

    pool_pr = (
        [
            AdaBoostClassifier(
                Perceptron(max_iter=1000, tol=1e-3, random_state=199),
                n_estimators=100,
                random_state=199,
                algorithm="SAMME",
            )
            for i in range(10, 160, 10)
        ]
        + [
            BaggingClassifier(
                Perceptron(max_iter=1000, tol=1e-3, random_state=199),
                n_estimators=100,
                random_state=199,
            )
            for i in range(10, 160, 10)
        ]
        + [
            SGH(
                base_estimator=CalibratedClassifierCV(
                    Perceptron(max_iter=1, random_state=199), cv=3
                ),
                n_estimators=i,
            )
            for i in range(10, 160, 10)
        ]
    )

    pool_pr_prob = (
        [
            AdaBoostClassifier(
                CalibratedClassifierCV(
                    Perceptron(max_iter=1000, tol=1e-3, random_state=199), cv=3
                ),
                n_estimators=i,
                random_state=199,
            )
            for i in range(10, 160, 10)
        ]
        + [
            BaggingClassifier(
                CalibratedClassifierCV(
                    Perceptron(max_iter=1000, tol=1e-3, random_state=199), cv=3
                ),
                n_estimators=i,
                random_state=199,
            )
            for i in range(10, 160, 10)
        ]
        + [
            SGH(
                base_estimator=CalibratedClassifierCV(
                    Perceptron(max_iter=1, random_state=199), cv=3
                ),
                n_estimators=i,
            )
            for i in range(10, 160, 10)
        ]
    )

    print("Training pools... ")

    l = [pool_dt, pool_pr, pool_pr_prob]
    X_train, X_dsel, y_train, y_dsel = train_test_split(
        X_train, y_train, test_size=0.5, random_state=199
    )

    for i in tqdm(range(0, len(l))):
        baggings = l[i]
        for b in baggings:
            b.fit(X_dsel, y_dsel)

    pool_dt[0].fit(X_dsel, y_dsel)

    print("OK.")

    ensemble = [
        OLA(),
        KNORAU(),
        KNORAE(),
        METADES(),
        SingleBest(),
        StaticSelection(),
        StackedClassifier(),
        OLA(),
        KNORAU(),
        KNORAE(),
        METADES(),
        SingleBest(),
        StaticSelection(),
        StackedClassifier(),
    ]

    k = [i for i in range(2, 101, 1)]
    pct_classifiers = np.arange(0.01, 1.0, 0.01)

    ensemble_params = [
        {"pool_classifiers": pool_pr, "k": k, "random_state": [199]},
        {"pool_classifiers": pool_pr, "k": k, "random_state": [199]},
        {"pool_classifiers": pool_pr, "k": k, "random_state": [199]},
        {"pool_classifiers": pool_pr_prob, "k": k, "random_state": [199]},
        {"pool_classifiers": pool_pr, "random_state": [199]},
        {
            "pool_classifiers": pool_pr,
            "pct_classifiers": pct_classifiers,
            "random_state": [199],
        },
        {
            "pool_classifiers": pool_pr_prob,
            "meta_classifier": [
                KNeighborsClassifier(),
                MLPClassifier(),
                DecisionTreeClassifier(),
                SVC(),
            ],
            "random_state": [199],
        },
        {"pool_classifiers": pool_dt, "k": k, "random_state": [199]},
        {"pool_classifiers": pool_dt, "k": k, "random_state": [199]},
        {"pool_classifiers": pool_dt, "k": k, "random_state": [199]},
        {"pool_classifiers": pool_dt, "k": k, "random_state": [199]},
        {"pool_classifiers": pool_dt, "random_state": [199]},
        {
            "pool_classifiers": pool_dt,
            "pct_classifiers": pct_classifiers,
            "random_state": [199],
        },
        {
            "pool_classifiers": pool_dt,
            "meta_classifier": [
                KNeighborsClassifier(),
                MLPClassifier(),
                DecisionTreeClassifier(),
                SVC(),
            ],
            "random_state": [199],
        },
    ]

    for i in range(len(ensemble)):
        c = ensemble[i]
        p = ensemble_params[i]

        gridRun(c, p, accuracy_score, X_train, y_train, X_test, y_test)


def main():

    # dataset = Dataset('./datasetutils/crude_db/', crude=True) # get crude db
    dataset = Dataset("./datasetutils/db/methods.csv")  # use already preprocessed db
    # pca_search(dataset, graph=False)
    pca_values = [39, 63]
    # pca_values = [0]

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        stratify=dataset.target,
        random_state=199,
    )

    X_train = X_train.rename(regulator, axis="columns")
    X_test = X_test.rename(regulator, axis="columns")
    # dataset.make_samples('./datasetutils/sampled_db/')

    # monolithic_search(X_train, y_train, X_test, y_test)
    ensemble_search(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
