from re import A
from functools import partial
from ensemble_classifiers import Ensemble

from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    StackingClassifier,
    AdaBoostClassifier,
)

from deslib.dcs.ola import OLA
from deslib.dcs.a_posteriori import APosteriori
from deslib.dcs.a_priori import APriori
from deslib.dcs.lca import LCA
from deslib.dcs.mcb import MCB
from deslib.dcs.mla import MLA
from deslib.dcs.rank import Rank
from deslib.dcs.olp import OLP, ModedOLP
from deslib.des.des_p import DESP
from deslib.des.knop import KNOP
from deslib.des.knora_u import KNORAU
from deslib.des.knora_e import KNORAE
from deslib.des.meta_des import METADES
from deslib.static import (
    SingleBest,
    StaticSelection,
    Oracle,
    StackedClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from deslib.util.sgh import SGH

from xgboost.sklearn import XGBClassifier

from rotation_forest import RotationForestClassifier

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from preprocessing.dimm_reduction import (
    sklearnPCA,
    TripletEmbedding,
    logisticPCA,
    AutoEncoder,
)


GRID_CLASSIFIERS = [
    DecisionTreeClassifier(
        max_depth=22, criterion="entropy", splitter="best", random_state=199
    ),
    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True),
    BernoulliNB(alpha=0.05, binarize=None),
    # GaussianNB(var_smoothing=7.2e-10),
    KNeighborsClassifier(weights="uniform", algorithm="brute", n_neighbors=1),
    SVC(C=2, kernel="linear", random_state=199),
    MLPClassifier(
        activation="relu",
        hidden_layer_sizes=100,
        solver="adam",
        max_iter=10000,
        random_state=199,
    ),
    RandomForestClassifier(
        n_estimators=164, criterion="gini", random_state=199
    ),
    XGBClassifier(max_depth=7, n_estimators=601, random_state=199),
    Ensemble(
        BaggingClassifier(
            Perceptron(max_iter=1000, tol=1e-3, random_state=199),
            n_estimators=100,
            random_state=199,
        ),
        OLA(k=74, random_state=199),
        autotrain=False,
    ),
    Ensemble(
        BaggingClassifier(
            Perceptron(max_iter=1000, tol=1e-3, random_state=199),
            n_estimators=100,
            random_state=199,
        ),
        KNORAU(k=3, random_state=199),
        autotrain=False,
    ),
    Ensemble(
        BaggingClassifier(
            Perceptron(max_iter=1000, tol=1e-3, random_state=199),
            n_estimators=100,
            random_state=199,
        ),
        KNORAE(k=2, random_state=199),
        autotrain=False,
    ),
    Ensemble(
        AdaBoostClassifier(
            CalibratedClassifierCV(
                Perceptron(max_iter=1000, tol=1e-3, random_state=199), cv=3
            ),
            n_estimators=140,
            random_state=199,
        ),
        METADES(k=16, random_state=199),
        autotrain=False,
    ),
    Ensemble(
        AdaBoostClassifier(
            Perceptron(max_iter=1000, tol=1e-3, random_state=199),
            algorithm="SAMME",
            n_estimators=100,
            random_state=199,
        ),
        SingleBest(random_state=199),
        autotrain=False,
    ),
    Ensemble(
        BaggingClassifier(
            Perceptron(max_iter=1000, tol=1e-3, random_state=199),
            n_estimators=100,
            random_state=199,
        ),
        StaticSelection(pct_classifiers=0.42, random_state=199),
        autotrain=False,
    ),
    Ensemble(
        BaggingClassifier(
            CalibratedClassifierCV(
                Perceptron(max_iter=1000, tol=1e-3, random_state=199), cv=3
            ),
            n_estimators=20,
            random_state=199,
        ),
        StackedClassifier(
            meta_classifier=MLPClassifier(random_state=199, max_iter=10000),
            random_state=199,
        ),
        autotrain=False,
    ),
    Ensemble(
        BaggingClassifier(
            Perceptron(max_iter=1000, tol=1e-3, random_state=199),
            n_estimators=100,
            random_state=199,
        ),
        Oracle(random_state=199),
        autotrain=False,
    ),
    Ensemble(
        AdaBoostClassifier(
            DecisionTreeClassifier(random_state=199),
            algorithm="SAMME",
            n_estimators=130,
            random_state=199,
        ),
        OLA(k=14, random_state=199),
        autotrain=False,
    ),
    Ensemble(
        AdaBoostClassifier(
            DecisionTreeClassifier(random_state=199),
            algorithm="SAMME",
            n_estimators=140,
            random_state=199,
        ),
        KNORAU(k=2, random_state=199),
        autotrain=False,
    ),
    Ensemble(
        AdaBoostClassifier(
            DecisionTreeClassifier(random_state=199),
            algorithm="SAMME",
            n_estimators=130,
            random_state=199,
        ),
        KNORAE(k=10, random_state=199),
        autotrain=False,
    ),
    Ensemble(
        AdaBoostClassifier(
            DecisionTreeClassifier(random_state=199),
            algorithm="SAMME",
            n_estimators=110,
            random_state=199,
        ),
        METADES(k=69, random_state=199),
        autotrain=False,
    ),
    Ensemble(
        AdaBoostClassifier(
            DecisionTreeClassifier(random_state=199),
            algorithm="SAMME",
            n_estimators=10,
            random_state=199,
        ),
        SingleBest(random_state=199),
        autotrain=False,
    ),
    Ensemble(
        AdaBoostClassifier(
            DecisionTreeClassifier(random_state=199),
            algorithm="SAMME",
            n_estimators=150,
            random_state=199,
        ),
        StaticSelection(pct_classifiers=0.65, random_state=199),
        autotrain=False,
    ),
    Ensemble(
        AdaBoostClassifier(
            DecisionTreeClassifier(random_state=199),
            algorithm="SAMME",
            n_estimators=130,
            random_state=199,
        ),
        StackedClassifier(
            meta_classifier=MLPClassifier(random_state=199, max_iter=10000)
        ),
        autotrain=False,
    ),
    Ensemble(
        BaggingClassifier(
            DecisionTreeClassifier(random_state=199),
            n_estimators=100,
            random_state=199,
        ),
        Oracle(),
        autotrain=False,
    ),
]


# DEFAULT_MODELS = [
#     DecisionTreeClassifier(random_state=199),
#     MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True),
#     BernoulliNB(binarize=None),
#     # GaussianNB(var_smoothing=1e-9),
#     KNeighborsClassifier(n_neighbors=7),
#     SVC(kernel='linear', random_state=199),
#     MLPClassifier(100, max_iter=1000, random_state=199),
#     RandomForestClassifier(n_estimators=10, random_state=199),
#     XGBClassifier(max_depth=3, n_estimators=100),
#     RotationForestClassifier(n_estimators=10, random_state=199),
#     OLP(),
#     Ensemble(Perceptron(max_iter=1000, tol=1e-3, random_state=199), OLA),

#     Ensemble(Perceptron(max_iter=1000, tol=1e-3, random_state=199), KNORAU, test_size=0.25),
#     Ensemble(Perceptron(max_iter=1000, tol=1e-3, random_state=199), KNORAE, test_size=0.25),
#     Ensemble(Perceptron(max_iter=1000, tol=1e-3, random_state=199), KNORAU, test_size=0.25, distance_algorithm=TripletEmbedding),
#     Ensemble(Perceptron(max_iter=1000, tol=1e-3, random_state=199), KNORAE, test_size=0.25, distance_algorithm=TripletEmbedding),
#     Ensemble(Perceptron(max_iter=1000, tol=1e-3, random_state=199), KNORAU, test_size=0.5),
#     Ensemble(Perceptron(max_iter=1000, tol=1e-3, random_state=199), KNORAE, test_size=0.5),

#     Ensemble(CalibratedClassifierCV(Perceptron(max_iter=1000, tol=1e-3, random_state=199), cv=3), METADES),
#     Ensemble(Perceptron(max_iter=1000, tol=1e-3, random_state=199), SingleBest),
#     Ensemble(Perceptron(max_iter=1000, tol=1e-3, random_state=199), StaticSelection),
#     Ensemble(CalibratedClassifierCV(Perceptron(max_iter=1000, tol=1e-3, random_state=199), cv=3), StackedClassifier),
#     Ensemble(Perceptron(max_iter=1000, tol=1e-3, random_state=199), Oracle),
#     Ensemble(SGH(), OLA(random_state=199), autotrain=False),
#     Ensemble(DecisionTreeClassifier(random_state=199, max_depth=1), OLA),
#     Ensemble(DecisionTreeClassifier(random_state=199, max_depth=1), KNORAU),
#     Ensemble(DecisionTreeClassifier(random_state=199, max_depth=1), KNORAE),
#     Ensemble(DecisionTreeClassifier(random_state=199, max_depth=1), METADES),
#     Ensemble(DecisionTreeClassifier(random_state=199, max_depth=1), SingleBest),
#     Ensemble(DecisionTreeClassifier(random_state=199, max_depth=1), StaticSelection),
#     Ensemble(DecisionTreeClassifier(random_state=199, max_depth=1), StackedClassifier),
#     Ensemble(DecisionTreeClassifier(random_state=199, max_depth=1), Oracle)
# ]


DEFAULT_CLASSIFIERS = {
    "Monolithic": {
        "Decision Tree": DecisionTreeClassifier(random_state=199),
        # "Multinomial Naive Bayes": MultinomialNB(
        #     alpha=1.0, class_prior=None, fit_prior=True
        # ),
        "Naive Bayes": BernoulliNB(binarize=None),
        # "Gaussian Naive Bayes": GaussianNB(var_smoothing=1e-9),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVM": SVC(kernel="linear", random_state=199),
        "MLP": MLPClassifier(100, max_iter=1000, random_state=199),
        # "Rotation Forest": RotationForestClassifier(n_estimators=10, random_state=199),
        # "Stacked Perceptron": Ensemble(CalibratedClassifierCV(Perceptron(max_iter=1000, tol=1e-3, random_state=199)), StackedClassifier),
        # "Stacked Decision Tree": Ensemble(DecisionTreeClassifier(random_state=199, max_depth=1), StackedClassifier),
        # "OLA SGH": Ensemble(SGH(), OLA(random_state=199), autotrain=False),
    },
    # "MCS Bagging Perceptron": {
    #     "OLA Perceptron": Ensemble(
    #         Perceptron(max_iter=1000, tol=1e-3, random_state=199),
    #         OLA,
    #     ),
    #     "KNORA-U Perceptron": Ensemble(
    #         Perceptron(max_iter=1000, tol=1e-3, random_state=199),
    #         KNORAU,
    #     ),
    #     "KNORA-E Perceptron": Ensemble(
    #         Perceptron(max_iter=1000, tol=1e-3, random_state=199),
    #         KNORAE,
    #     ),
    #     "KNOP Perceptron": Ensemble(
    #         Perceptron(max_iter=1000, tol=1e-3, random_state=199),
    #         KNOP,
    #         calibrate=True,
    #     ),
    #     "META-DES Perceptron": Ensemble(
    #         Perceptron(max_iter=1000, tol=1e-3, random_state=199),
    #         METADES,
    #         calibrate=True,
    #     ),
    #     "Single Best Perceptron": Ensemble(
    #         Perceptron(max_iter=1000, tol=1e-3, random_state=199),
    #         SingleBest,
    #     ),
    #     "Static Selection Perceptron": Ensemble(
    #         Perceptron(max_iter=1000, tol=1e-3, random_state=199),
    #         StaticSelection,
    #     ),
    #     "Oracle Perceptron": Ensemble(
    #         Perceptron(max_iter=1000, tol=1e-3, random_state=199),
    #         Oracle,
    #     ),
    #     # "KNORA-U Perceptron": Ensemble(Perceptron(max_iter=1000, tol=1e-3, random_state=199), KNORAU, test_size=0.25),
    #     # "KNORA-E Perceptron": Ensemble(Perceptron(max_iter=1000, tol=1e-3, random_state=199), KNORAE, test_size=0.25),
    #     # "KNORA-U Perceptron with Triplet RoC": Ensemble(
    #     #     Perceptron(max_iter=1000, tol=1e-3, random_state=199),
    #     #     KNORAU,
    #     #     distance_algorithm=TripletEmbedding,
    #     # ),
    #     # "KNORA-E Perceptron with Triplet RoC": Ensemble(
    #     #     Perceptron(max_iter=1000, tol=1e-3, random_state=199),
    #     #     KNORAE,
    #     #     distance_algorithm=TripletEmbedding,
    #     # ),
    #     # "OLP": OLP(n_classifiers=100),
    #     # "ModedOLP": ModedOLP(n_classifiers=100),
    # },
    # "MCS Decision Tree": {
    #     "OLA Decision Tree": Ensemble(
    #         DecisionTreeClassifier(random_state=199, max_depth=2), OLA
    #     ),
    #     "KNORA-U Decision Tree": Ensemble(
    #         DecisionTreeClassifier(random_state=199, max_depth=2), KNORAU
    #     ),
    #     "KNORA-E Decision Tree": Ensemble(
    #         DecisionTreeClassifier(random_state=199, max_depth=2), KNORAE
    #     ),
    #     "KNOP Decision Tree": Ensemble(
    #         DecisionTreeClassifier(random_state=199, max_depth=2), KNOP
    #     ),
    #     # "KNORA-U Decision Tree with Triplet RoC": Ensemble(
    #     #     DecisionTreeClassifier(random_state=199, max_depth=2),
    #     #     KNORAU,
    #     #     distance_algorithm=TripletEmbedding,
    #     # ),
    #     # "KNORA-E Decision Tree with Triplet RoC": Ensemble(
    #     #     DecisionTreeClassifier(random_state=199, max_depth=2),
    #     #     KNORAE,
    #     #     distance_algorithm=TripletEmbedding,
    #     # ),
    #     "META-DES Decision Tree": Ensemble(
    #         DecisionTreeClassifier(random_state=199, max_depth=2), METADES
    #     ),
    #     "Single Best Decision Tree": Ensemble(
    #         DecisionTreeClassifier(random_state=199, max_depth=2), SingleBest
    #     ),
    #     "Static Selection Decision Tree": Ensemble(
    #         DecisionTreeClassifier(random_state=199, max_depth=2),
    #         StaticSelection,
    #     ),
    #     "Oracle Decision Tree": Ensemble(
    #         DecisionTreeClassifier(random_state=199, max_depth=2), Oracle
    #     ),
    # },
    # "MCS Boosting Perceptron": {
    #     "OLA Boost Perceptron": Ensemble(
    #         Perceptron(max_iter=1000, tol=1e-3, random_state=199),
    #         OLA,
    #         pool_training_scheme=partial(
    #             AdaBoostClassifier, algorithm="SAMME"
    #         ),
    #     ),
    #     "KNORA-U Boost Perceptron": Ensemble(
    #         Perceptron(max_iter=1000, tol=1e-3, random_state=199),
    #         KNORAU,
    #         pool_training_scheme=partial(
    #             AdaBoostClassifier, algorithm="SAMME"
    #         ),
    #     ),
    #     "KNORA-E Boost Perceptron": Ensemble(
    #         Perceptron(max_iter=1000, tol=1e-3, random_state=199),
    #         KNORAE,
    #         pool_training_scheme=partial(
    #             AdaBoostClassifier, algorithm="SAMME"
    #         ),
    #     ),
    #     "KNOP Boost Decision Tree": Ensemble(
    #         Perceptron(max_iter=1000, tol=1e-3, random_state=199),
    #         KNOP,
    #         pool_training_scheme=partial(
    #             AdaBoostClassifier, algorithm="SAMME"
    #         ),
    #         calibrate=True,
    #     ),
    #     "META-DES Boost Perceptron": Ensemble(
    #         Perceptron(max_iter=1000, tol=1e-3, random_state=199),
    #         METADES,
    #         pool_training_scheme=partial(
    #             AdaBoostClassifier, algorithm="SAMME"
    #         ),
    #         calibrate=True,
    #     ),
    #     "Single Best Boost Perceptron": Ensemble(
    #         Perceptron(max_iter=1000, tol=1e-3, random_state=199),
    #         SingleBest,
    #         pool_training_scheme=partial(
    #             AdaBoostClassifier, algorithm="SAMME"
    #         ),
    #     ),
    #     "Static Selection Boost Perceptron": Ensemble(
    #         Perceptron(max_iter=1000, tol=1e-3, random_state=199),
    #         StaticSelection,
    #         pool_training_scheme=partial(
    #             AdaBoostClassifier, algorithm="SAMME"
    #         ),
    #     ),
    #     "Oracle Boost Perceptron": Ensemble(
    #         Perceptron(max_iter=1000, tol=1e-3, random_state=199),
    #         Oracle,
    #         pool_training_scheme=partial(
    #             AdaBoostClassifier, algorithm="SAMME"
    #         ),
    #     ),
    # },
    # "MCS Boosting Decision Tree": {
    #     "OLA Boost Decision Tree": Ensemble(
    #         DecisionTreeClassifier(random_state=199, max_depth=2),
    #         OLA,
    #         pool_training_scheme=AdaBoostClassifier,
    #     ),
    #     "KNORA-U Boost Decision Tree": Ensemble(
    #         DecisionTreeClassifier(random_state=199, max_depth=2),
    #         KNORAU,
    #         pool_training_scheme=AdaBoostClassifier,
    #     ),
    #     "KNORA-E Boost Decision Tree": Ensemble(
    #         DecisionTreeClassifier(random_state=199, max_depth=2),
    #         KNORAE,
    #         pool_training_scheme=AdaBoostClassifier,
    #     ),
    #     "KNOP Boost Decision Tree": Ensemble(
    #         DecisionTreeClassifier(random_state=199, max_depth=2),
    #         KNOP,
    #         pool_training_scheme=AdaBoostClassifier,
    #     ),
    #     "META-DES Boost Decision Tree": Ensemble(
    #         DecisionTreeClassifier(random_state=199, max_depth=2),
    #         METADES,
    #         pool_training_scheme=AdaBoostClassifier,
    #     ),
    #     "Single Best Boost Decision Tree": Ensemble(
    #         DecisionTreeClassifier(random_state=199, max_depth=2),
    #         SingleBest,
    #         pool_training_scheme=AdaBoostClassifier,
    #     ),
    #     "Static Selection Boost Decision Tree": Ensemble(
    #         DecisionTreeClassifier(random_state=199, max_depth=2),
    #         StaticSelection,
    #         pool_training_scheme=AdaBoostClassifier,
    #     ),
    #     "Oracle Boost Decision Tree": Ensemble(
    #         DecisionTreeClassifier(random_state=199, max_depth=2),
    #         Oracle,
    #         pool_training_scheme=AdaBoostClassifier,
    #     ),
    # },
    "MCS Random Forest": {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=199,
        ),
        "Gradient Boosted Decision Trees": XGBClassifier(
            n_estimators=100,
            random_state=199,
            use_label_encoder=False,
            eval_metric="logloss",
        ),
        "Single Best": Ensemble(
            RandomForestClassifier(n_estimators=100, random_state=199),
            SingleBest,
            pool_training_scheme=None,
            test_size=None,
        ),
        "Static Selection": Ensemble(
            RandomForestClassifier(n_estimators=100, random_state=199),
            StaticSelection,
            pool_training_scheme=None,
            test_size=None,
        ),
        "OLA": Ensemble(
            RandomForestClassifier(n_estimators=100, random_state=199),
            OLA,
            pool_training_scheme=None,
            test_size=None,
        ),
        "KNORA-U": Ensemble(
            RandomForestClassifier(n_estimators=100, random_state=199),
            KNORAU,
            pool_training_scheme=None,
            test_size=None,
        ),
        "KNORA-E": Ensemble(
            RandomForestClassifier(n_estimators=100, random_state=199),
            KNORAE,
            pool_training_scheme=None,
            test_size=None,
        ),
        "KNOP": Ensemble(
            RandomForestClassifier(n_estimators=100, random_state=199),
            KNOP,
            pool_training_scheme=None,
            test_size=None,
        ),
        "META-DES": Ensemble(
            RandomForestClassifier(n_estimators=100, random_state=199),
            METADES,
            pool_training_scheme=None,
            test_size=None,
        ),
        "Oracle": Ensemble(
            RandomForestClassifier(n_estimators=100, random_state=199),
            Oracle,
            pool_training_scheme=None,
            test_size=None,
        ),
    },
    # "MCS Random Subspaces Decision Tree": {
    #     "OLA": Ensemble(
    #         DecisionTreeClassifier(random_state=199, max_depth=2),
    #         OLA,
    #         pool_training_scheme=partial(
    #             BaggingClassifier, max_samples=0.5, n_estimators=100, bootstrap=True, bootstrap_features=True
    #         ),
    #     ),
    #     "KNORA-U": Ensemble(
    #         DecisionTreeClassifier(random_state=199, max_depth=2),
    #         KNORAU,
    #         pool_training_scheme=partial(
    #             BaggingClassifier, max_samples=0.5, n_estimators=100, bootstrap=True, bootstrap_features=True
    #         ),
    #     ),
    #     "KNORA-E": Ensemble(
    #         DecisionTreeClassifier(random_state=199, max_depth=2),
    #         KNORAE,
    #         pool_training_scheme=partial(
    #             BaggingClassifier, max_samples=0.5, n_estimators=100, bootstrap=True, bootstrap_features=True
    #         ),
    #     ),
    #     "KNOP Random Forest": Ensemble(
    #         RandomForestClassifier(n_estimators=100, random_state=199, max_depth=2),
    #         KNOP,
    #         pool_training_scheme=None,
    #     ),
    #     "META-DES": Ensemble(
    #         DecisionTreeClassifier(random_state=199, max_depth=2),
    #         METADES,
    #         pool_training_scheme=partial(
    #             BaggingClassifier, max_samples=0.5, n_estimators=100, bootstrap=True, bootstrap_features=True
    #         ),
    #     ),
    #     "Single Best": Ensemble(
    #         DecisionTreeClassifier(random_state=199, max_depth=2),
    #         SingleBest,
    #         pool_training_scheme=partial(
    #             BaggingClassifier, max_samples=0.5, n_estimators=100, bootstrap=True, bootstrap_features=True
    #         ),
    #     ),
    #     "Static Selection": Ensemble(
    #         DecisionTreeClassifier(random_state=199, max_depth=2),
    #         StaticSelection,
    #         pool_training_scheme=partial(
    #             BaggingClassifier, max_samples=0.5, n_estimators=100, bootstrap=True, bootstrap_features=True
    #         ),
    #     ),
    #     "Oracle": Ensemble(
    #         DecisionTreeClassifier(random_state=199, max_depth=2),
    #         Oracle,
    #         pool_training_scheme=partial(
    #             BaggingClassifier, max_samples=0.5, n_estimators=100, bootstrap=True, bootstrap_features=True
    #         ),
    #     ),
    # },
}

POOL_COMPARISON = [
    Ensemble(
        BaggingClassifier(
            Perceptron(max_iter=1000, tol=1e-3, random_state=199),
            n_estimators=100,
        ),
        OLA(random_state=199),
        autotrain=False,
    ),
    Ensemble(
        BaggingClassifier(
            CalibratedClassifierCV(
                Perceptron(max_iter=1000, tol=1e-3, random_state=199), cv=3
            ),
            n_estimators=100,
        ),
        APriori(random_state=199),
        autotrain=False,
    ),
    Ensemble(
        BaggingClassifier(
            CalibratedClassifierCV(
                Perceptron(max_iter=1000, tol=1e-3, random_state=199), cv=3
            ),
            n_estimators=100,
        ),
        APosteriori(random_state=199),
        autotrain=False,
    ),
    Ensemble(
        AdaBoostClassifier(
            Perceptron(max_iter=1000, tol=1e-3, random_state=199),
            n_estimators=50,
            algorithm="SAMME",
            random_state=199,
        ),
        OLA(random_state=199),
        autotrain=False,
    ),
    Ensemble(
        AdaBoostClassifier(
            CalibratedClassifierCV(
                Perceptron(max_iter=1000, tol=1e-3, random_state=199), cv=3
            ),
            n_estimators=100,
        ),
        APriori(random_state=199),
        autotrain=False,
    ),
    Ensemble(
        AdaBoostClassifier(
            CalibratedClassifierCV(
                Perceptron(max_iter=1000, tol=1e-3, random_state=199), cv=3
            ),
            n_estimators=100,
        ),
        APosteriori(random_state=199),
        autotrain=False,
    ),
    Ensemble(SGH(), OLA(random_state=199), autotrain=False),
    Ensemble(
        SGH(
            base_estimator=CalibratedClassifierCV(
                Perceptron(max_iter=1000, tol=1e-3, random_state=199), cv=3
            )
        ),
        APriori(random_state=199),
        autotrain=False,
    ),
    Ensemble(
        SGH(
            base_estimator=CalibratedClassifierCV(
                Perceptron(max_iter=1000, tol=1e-3, random_state=199), cv=3
            )
        ),
        APosteriori(random_state=199),
        autotrain=False,
    ),
    OLP(),
]

EMBEDDINGS = {
    "triplet": TripletEmbedding,
    "pca": sklearnPCA,
    "autoencodertorch": AutoEncoder,
    "logpca": logisticPCA,
}
