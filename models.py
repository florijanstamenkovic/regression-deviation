import math
import logging

import numpy as np
import sklearn.linear_model
import sklearn.ensemble
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


EPS = 1e-7


def log_norm_pdf(truth, predicted_mean, predicted_stddev):
    """
    Returns the logarithm of the probability density function
    of the normal distribution. Each argument can be either a
    numeric value or a numpy array.
    Args:
        truth: the point(s) at which the log-pdf should be calculated.
        predicted_mean: predicted mean(s) of the pdf.
        predicted_mean: predicted standard deviation(s) of the pdf.
    Return:
        Logarithm(s) of the probability densit(y/ies).
    """
    var = predicted_stddev ** 2 + EPS
    distance = (truth - predicted_mean) ** 2
    return -0.5 * (np.log(2 * math.pi * var) + distance / var)


class LogNormPdfScorer():
    """
    A scorer that can be used with sklearn's optimization hyper-param
    optimization functions and which takes the predicted mean and
    deviation into account and returns the mean-log-prob-density.
    """
    def __call__(self, estimator, X, y):
        mean = estimator.predict(X)
        if isinstance(estimator, GridSearchCV):
            stddev = estimator.best_estimator_.predict_stddev(X)
        else:
            stddev = estimator.predict_stddev(X)

        return log_norm_pdf(y, mean, stddev).mean()


class DeviationRegressor():
    """
    A class that models the standard deviation using a separate regression
    model then the one for estimating the mean. It's trained on the biased
    single-point deviation estimates (absolute distance) calculated from
    the mean-model predictions.
    """

    def __init__(self, **params):
        self.set_params(**params)

    def _regressor(self):
        regression_params = {
            k.split("__")[1]: v for k, v in self.params.items() if "__" in k}
        return self.params["regression_cls"](**regression_params)

    def fit(self, X, y):
        params = self.get_params()

        reg_conf_split = params["reg_conf_split"]
        if reg_conf_split is None:
            X_reg, X_conf = X, X
            y_reg, y_conf = y, y
        else:
            X_reg, X_conf, y_reg, y_conf = train_test_split(
                X, y, test_size=params["reg_conf_split"])

        self.regression = self._regressor()
        self.regression.fit(X_reg, y_reg)
        self.stddev = self._regressor()
        self.stddev.fit(
            X_conf, np.abs(self.regression.predict(X_conf) - y_conf))

    def predict(self, X):
        return self.regression.predict(X)

    def predict_stddev(self, X):
        return self.stddev.predict(X)

    def set_params(self, **params):
        self.params = params
        return self

    def get_params(self, deep=True):
        return self.params


class BaggingRegressor(sklearn.ensemble.BaggingRegressor):
    """ Extends sklearn's bagging to predict standard deviation. """

    def predict_stddev(self, X):
        std = np.std([e.predict(X) for e in self.estimators_], axis=0, ddof=1)
        return std


class KNeighborsRegressor(sklearn.neighbors.KNeighborsRegressor):
    """ Extends sklearn's k-neighbors-regressor to predict standard deviation. """

    def predict_stddev(self, X):
        distances, indices = self.kneighbors(X, return_distance=True)

        weight_param = self.get_params()["weights"]
        if weight_param == "uniform":
            weight = 1 / indices.shape[1]
        elif weight_param == "distance":
            weight = 1 / (distances + EPS)
            weight /= weight.sum(axis=1).reshape(-1, 1)
        else:
            raise Exception("Can't handle weighting param")

        neighbor_y = self._y[indices]
        means = (neighbor_y * weight).sum(axis=1)
        return (((neighbor_y - means.reshape(-1, 1)) ** 2)
                * weight).sum(axis=1) ** 0.5


class LogDensityGradientRegressor(nn.Module):
    """
    Estimates mean and standard deviation as linear functions of the input.
    Learns the function parameters by back-propagating the gradient of the
    logarithm of the normal distribution probability density function.
    """

    def __init__(self, **params):
        super(LogDensityGradientRegressor, self).__init__()
        self.set_params(**params)

    def forward_mean(self, x):
        return self.mean(x)

    def forward_std(self, x):
        return torch.max(torch.tensor(EPS), self.stddev(x))

    def log_prob(self, X, y):
        mean = self.forward_mean(X)
        stddev = self.forward_std(X)
        return -((2 * math.pi * stddev).log() +
                 ((y.reshape(-1, 1) - mean) ** 2) / stddev) / 2

    def fit(self, X, y):
        self.mean = nn.Linear(16, 1)
        self.stddev = nn.Linear(16, 1)
        with torch.no_grad():
            self.mean.weight.fill_(0)
            self.mean.bias.fill_(y.mean())
            self.stddev.weight.fill_(0)
            self.stddev.bias.fill_(y.std())

        optimizer = optim.LBFGS(self.parameters(), max_iter=100)

        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)

        def closure():
            optimizer.zero_grad()
            loss = -self.log_prob(X, y).mean()
            loss += (self.mean.weight ** 2).mean() * self.params["alpha"]
            loss += (self.stddev.weight ** 2).mean() * self.params["alpha"]
            loss.backward()
            return loss

        optimizer.step(closure)
        return self

    def set_params(self, **params):
        self.params = params
        return self

    def get_params(self, deep=True):
        return self.params

    def predict(self, X):
        with torch.no_grad():
            return self.forward_mean(
                torch.FloatTensor(X)).cpu().numpy().flatten()

    def predict_stddev(self, X):
        with torch.no_grad():
            return self.forward_std(
                torch.FloatTensor(X)).cpu().numpy().flatten()
