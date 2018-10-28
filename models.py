import math

import numpy as np
import sklearn.linear_model
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.gaussian_process import GaussianProcessRegressor


def concat(ftrs, prediction):
    return np.hstack((ftrs, prediction.reshape(-1, 1)))


def normpdf(x, mean, stddev):
    var = stddev ** 2
    denom = (2 * math.pi * var) ** 0.5
    num = np.exp(-((x - mean) ** 2 / (2 * var)))
    return num / denom


def log_prob(x, mean):
    return -0.5 * (np.log(2 * math.pi) + 2 * np.log(1e-5 + np.abs(x - mean)) +
                   1)


class RegressionConfidenceScorer():

    def __init__(self):
        pass

    def __call__(self, estimator, X, y):
        prediction = estimator.predict(X)
        if isinstance(estimator, GridSearchCV):
            stddev = estimator.best_estimator_.predict_stddev(
                X, prediction)
        else:
            stddev = estimator.predict_stddev(X, prediction)

        return log_prob(prediction, y).mean()


class ConfidenceRegressor():
    def __init__(self, **params):
        self.set_params(**params)

    def _extract_params(self, prefix):
        return {k[len(prefix):]: v for k, v in self.params.items()
                if k.startswith(prefix)}

    def fit(self, X, y):
        params = self.get_params()

        X_reg, X_conf, y_reg, y_conf = train_test_split(
            X, y, test_size=params["reg_conf_split"])

        self.regression = params["regression_cls"](
            **self._extract_params("regression__"))
        self.regression.fit(X_reg, y_reg)

        self.stddev = params["stddev_cls"](
            **self._extract_params("stddev__"))

        # Fit stddev.
        regression_pred = self.regression.predict(X_conf)
        ftrs = concat(X_conf, regression_pred)
        self.stddev.fit(ftrs, np.abs(regression_pred - y_conf))

    def predict(self, X):
        return self.regression.predict(X)

    def predict_stddev(self, X, prediction):
        ftrs = concat(X, prediction)
        return self.stddev.predict(ftrs)

    def set_params(self, **params):
        self.params = params
        return self

    def get_params(self, deep=True):
        return self.params


class GaussianProcessEnsemble():

    def __init__(self, **params):
        self.count = 10
        self.gps = [GaussianProcessRegressor() for _ in range(self.count)]
        self.set_params(**params)

    def fit(self, X, y):
        fold = KFold(self.count)
        for (_, fold), gp in zip(fold.split(X), self.gps):
            gp.fit(X[fold], y[fold])

    def predict(self, X):
        return np.mean([gp.predict(X) for gp in self.gps], axis=0)

    def predict_stddev(self, X, _):
        predictions = [gp.predict(X, return_std=True) for gp in self.gps]
        mean, stddev = zip(*predictions)

        # Convert means and stds into (N_samples, ensemble_size)
        # and weight them uniformly
        mean = np.vstack(mean).T / self.count
        stddev = np.vstack(stddev).T / self.count

        return stddev.mean(axis=1) + np.power(mean, 2).mean(axis=1) - \
            np.power(mean.mean(axis=1), 2)

    def set_params(self, **params):
        self.params = params
        for gp in self.gps:
            gp.set_params(**params)
        return self

    def get_params(self, deep=True):
        return self.params
