from inspect import isclass
import numpy as np
from numpy.core.fromnumeric import ndim
from sklearn import neighbors
from sklearn.utils.validation import has_fit_parameter
from sklearn.calibration import CalibratedClassifierCV

from sklearn.ensemble import BaggingClassifier

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from copy import deepcopy


class SingletonPoolMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class PoolManager(metaclass=SingletonPoolMeta):
    pool = {}
    saved_div = {}
    fit_control = {}
    pool_type = {}
    distance_metric = {}
    pool_index = {}

    def get_pool_len(self, dimms, test_size):
        return len(self.distance_metric[dimms][test_size])

    def next_idx(self, dimms, test_size):
        self.pool_index[dimms][test_size] += 1
        if self.pool_index[dimms][test_size] == 30:
            self.pool_index[dimms][test_size] = 0

    def new_dimm(self, dimms):
        if dimms not in self.pool:
            self.pool[dimms] = {}
            self.saved_div[dimms] = {}
            self.fit_control[dimms] = {}
            self.pool_type[dimms] = {}
            self.distance_metric[dimms] = {}
            self.pool_index[dimms] = {}

    def new_test_sample(self, dimms, test_size):
        if test_size not in self.pool[dimms]:
            self.pool[dimms][test_size] = []
            self.saved_div[dimms][test_size] = []
            self.fit_control[dimms][test_size] = []
            self.pool_type[dimms][test_size] = []
            self.distance_metric[dimms][test_size] = []
            self.pool_index[dimms][test_size] = 0

    def clear(self):
        self.pool = {}
        self.saved_div = {}
        self.fit_control = {}
        self.pool_type = {}
        self.distance_metric = {}
        self.pool_index = {}


class Ensemble(BaseEstimator):
    def __init__(
        self,
        base,
        technique,
        n_estimators=100,
        pool_training_scheme=BaggingClassifier,
        autotrain=True,
        distance_algorithm=None,
        calibrate=False,
        test_size=0.5,
    ):
        self.base = base
        self._distance_algorithm = distance_algorithm
        self.n_estimators = n_estimators
        self._autotrain = autotrain
        self.test_size = test_size
        self.pool_training_scheme = pool_training_scheme
        self.calibrate = calibrate
        self.singleton_pool = PoolManager()

        self._technique = technique
        if self._autotrain:
            self.__name__ = "%s_%s_%s" % (
                technique().__class__.__name__,
                BaggingClassifier.__name__,
                base.__class__.__name__,
            )
        else:
            self.__name__ = "%s_%s_%s" % (
                technique.__class__.__name__,
                base.__class__.__name__,
                base.base_estimator.__class__.__name__,
            )

    def fit(self, X, y):
        dimms = X.shape[1]

        if self._autotrain:
            # if test size is none, DSEL == X_train
            if self.test_size is not None:
                X_train, X_dsel, y_train, y_dsel = train_test_split(
                    X,
                    y,
                    test_size=self.test_size,
                    random_state=199,
                    stratify=y,
                )
            else:
                X_train, X_dsel, y_train, y_dsel = X, X, y, y
                self.test_size = 0

            if isinstance(self.base, BaggingClassifier):
                raise TypeError("Base classifier is a bagging when autotrain.")

            self.singleton_pool.new_dimm(dimms)
            self.singleton_pool.new_test_sample(dimms, self.test_size)

            est_idx = self.singleton_pool.pool_index[dimms][self.test_size]

            if (
                self._distance_algorithm is not None
                and self.singleton_pool.get_pool_len(dimms, self.test_size)
                < 30
            ):
                self.singleton_pool.distance_metric[dimms][
                    self.test_size
                ].append(self._distance_algorithm())
                self.singleton_pool.distance_metric[dimms][self.test_size][
                    est_idx
                ].fit(X, y)

            # if the size of the pool is less than 30, train a new pool and add to the pool manager
            if len(self.singleton_pool.pool[dimms][self.test_size]) < 30:
                # if pool traning scheme is None, then it must use the base classifier as pool
                if self.n_estimators == 1 or self.pool_training_scheme is None:
                    pool = deepcopy(self.base)
                else:
                    pool = self.pool_training_scheme(
                        self.base,
                        n_estimators=self.n_estimators,
                        random_state=199,
                    )

                self.singleton_pool.pool[dimms][self.test_size].append(pool)
                self.singleton_pool.pool_type[dimms][self.test_size].append(
                    type(self.base)
                )
                self.singleton_pool.fit_control[dimms][self.test_size].append(
                    False
                )
                self.singleton_pool.saved_div[dimms][self.test_size].append(
                    [X_train, X_dsel, y_train, y_dsel]
                )

            else:
                if self.singleton_pool.pool_type[dimms][self.test_size][
                    est_idx
                ] != type(self.base):
                    if (
                        self.n_estimators == 1
                        or self.pool_training_scheme is None
                    ):
                        pool = deepcopy(self.base)
                    else:
                        pool = self.pool_training_scheme(
                            self.base,
                            n_estimators=self.n_estimators,
                            random_state=199,
                        )

                    self.singleton_pool.pool[dimms][self.test_size][
                        est_idx
                    ] = deepcopy(pool)
                    self.singleton_pool.fit_control[dimms][self.test_size][
                        est_idx
                    ] = False
                    self.singleton_pool.pool_type[dimms][self.test_size][
                        est_idx
                    ] = type(self.base)
                    (
                        X_train,
                        X_dsel,
                        y_train,
                        y_dsel,
                    ) = self.singleton_pool.saved_div[dimms][self.test_size][
                        est_idx
                    ]
                else:
                    (
                        X_train,
                        X_dsel,
                        y_train,
                        y_dsel,
                    ) = self.singleton_pool.saved_div[dimms][self.test_size][
                        est_idx
                    ]

            self.pool = self.singleton_pool.pool[dimms][self.test_size][
                est_idx
            ]

            # check if already fitted
            if not self.singleton_pool.fit_control[dimms][self.test_size][
                est_idx
            ]:
                control = False
                control_seed = 199
                # while pool os classifiers is len(1), change seed
                while True:
                    self.pool.fit(X_train, y_train)
                    if len(self.pool) != 1:
                        self.singleton_pool.fit_control[dimms][self.test_size][
                            est_idx
                        ] = True
                        if control:
                            self.singleton_pool.pool[dimms][self.test_size][
                                est_idx
                            ] = self.pool
                        break
                    else:
                        if control_seed == 10000:
                            raise TimeoutError(
                                "Too much seed searched during ensemble fit training."
                            )
                        control_seed += 1
                        self.pool = deepcopy(pool)
                        self.pool.random_state = control_seed
                        control = True

            if self.calibrate:
                calibrated_pool = []
                for clf in self.pool:
                    calibrated = CalibratedClassifierCV(
                        base_estimator=clf, cv="prefit"
                    )
                    calibrated.fit(X_dsel, y_dsel)
                    calibrated_pool.append(calibrated)

                self.pool = calibrated_pool

            if self._distance_algorithm != None:
                self.technique = self._technique(
                    pool_classifiers=self.pool,
                    random_state=199,
                    neighbor_metric=self.singleton_pool.distance_metric[dimms][
                        self.test_size
                    ][est_idx].compare,
                )
            else:
                self.technique = self._technique(self.pool, random_state=199)

            self.technique.fit(X_dsel, y_dsel)
            self.singleton_pool.next_idx(dimms, self.test_size)

        else:
            self.pool = deepcopy(self.base)
            X_train, X_dsel, y_train, y_dsel = train_test_split(
                X, y, test_size=0.5, random_state=np.random.RandomState(199)
            )
            self.pool.fit(X_dsel, y_dsel)

            if self._distance_algorithm is not None:
                distance_algorithm = deepcopy(self.distance_algorithm).fit(
                    X, y
                )
                self.technique = self._technique(
                    pool_classifiers=self.pool,
                    random_state=199,
                    neighbor_metric=distance_algorithm,
                )
            else:
                self.technique = self._technique(
                    pool_classifiers=self.pool, random_state=199
                )

            self.technique.fit(X_train, y_train)

    def predict(self, X, y=None):
        if type(y) != type(None):
            return self.technique.predict(X, y)
        else:
            return self.technique.predict(X)
