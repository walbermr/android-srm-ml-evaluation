from ensemble_classifiers import Ensemble

from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from deslib.dcs.ola import OLA
from deslib.des.knop import KNOP
from deslib.des.knora_u import KNORAU
from deslib.des.knora_e import KNORAE
from deslib.des.meta_des import METADES
from deslib.static import (
    SingleBest,
    StaticSelection,
    Oracle,
)
from autosklearn.classification import AutoSklearnClassifier

from xgboost.sklearn import XGBClassifier

from preprocessing.dimm_reduction import (
    sklearnPCA,
    TripletEmbedding,
    logisticPCA,
    AutoEncoder,
)

DEFAULT_CLASSIFIERS = {
    "Monolithic": {
        "Decision Tree": DecisionTreeClassifier(random_state=199),
        "Naive Bayes": BernoulliNB(binarize=None),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVM": SVC(kernel="linear", random_state=199),
        "MLP": MLPClassifier(100, max_iter=1000, random_state=199),
        "Auto-sklearn": AutoSklearnClassifier(),
    },
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
}

EMBEDDINGS = {
    "triplet": TripletEmbedding,
    "pca": sklearnPCA,
    "autoencodertorch": AutoEncoder,
    "logpca": logisticPCA,
}
