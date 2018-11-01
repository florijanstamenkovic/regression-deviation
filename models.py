import math
import logging

import numpy as np
import sklearn.linear_model
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def concat(ftrs, prediction):
    return np.hstack((ftrs, prediction.reshape(-1, 1)))


def normpdf(mean, stddev, prediction, log):
    var = stddev ** 2
    denom = (2 * math.pi * var) ** 0.5
    num = np.exp(-((mean - prediction) ** 2 / (2 * var)))
    prob = num / denom
    return np.log(prob + 0.000001) if log else prob


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

        return normpdf(y, stddev, prediction, True).mean()


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


class BayesianRidge():
    def __init__(self, **params):
        self.ridge = sklearn.linear_model.BayesianRidge(**params)
        self.params = params

    def fit(self, X,  y):
        self.ridge.fit(X, y)

    def predict(self, X):
        return self.ridge.predict(X)

    def predict_stddev(self, X, _):
        return self.ridge.predict(X, return_std=True)[1]

    def set_params(self, **params):
        self.params = params
        self.ridge.set_params(**params)
        return self

    def get_params(self, deep=True):
        return self.params


class GaussianProcessEnsemble():

    @staticmethod
    def _make_gp():
        kernel = kernels.ConstantKernel() * kernels.RBF() + kernels.WhiteKernel()
        return GaussianProcessRegressor(kernel, normalize_y=True)

    def __init__(self, **params):
        self.count = 20
        self.gps = [GaussianProcessEnsemble._make_gp()
                    for _ in range(self.count)]
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


class TorchRegressor(nn.Module):

    def __init__(self, **params):
        super(TorchRegressor, self).__init__()
        self.set_params(**params)
        self.mnb_size = 512
        self.epochs = 20

        self.mean_hid = nn.Linear(17, 512)
        self.mean_do = nn.Dropout(0.2)
        self.mean = nn.Linear(17, 1)
        self.stddev = nn.Linear(17, 1)
        self.optimizer = optim.RMSprop(self.parameters(), lr=0.001)

    def prob(self, x, mean, stddev):
        var = stddev ** 2
        denom = (2 * math.pi * var) ** 0.5
        num = (-((x - mean) ** 2 / (2 * var))).exp()
        return num / denom + 0.000001

    def fit(self, X, y):
        logging.info("Target mean: %.2f, stddev: %.2f",
                     y.mean(), y.std())

        self.mean.bias.data.fill_(y.mean())
        self.mean.weight.data.uniform_(0)
        self.stddev.bias.data.fill_(y.std())
        self.stddev.weight.data.fill_(0)

        def batches():
            inds = np.random.shuffle(np.arange(len(X)))

            for batch_ind in range(0, len(X) // self.mnb_size):
                start = self.mnb_size * batch_ind
                end = start + self.mnb_size
                yield torch.FloatTensor(X[start:end]), torch.FloatTensor(y[start:end])

        for epoch_ind in range(self.epochs):
            epoch_losses = []
            for X_mnb, y_mnb in batches():
                # mean = self.mean(self.mean_do(self.mean_hid(X_mnb)))
                mean = self.mean(X_mnb)
                stddev = self.stddev(X_mnb)
                # logging.info("%.3f, %.3f", mean.mean().item(), stddev.mean().item())
                prob = self.prob(y_mnb, mean, stddev)
                loss = -prob.log().mean()
                epoch_losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            logging.info("TorchRegressor epoch %d, loss %.5f",
                         epoch_ind, np.mean(epoch_losses))

    def set_params(self, **params):
        self.params = params
        return self

    def get_params(self, deep=True):
        return self.params

    def predict(self, X):
        with torch.no_grad():
            return self.mean(torch.FloatTensor(X)).cpu().numpy().flatten()

    def predict_stddev(self, X, _):
        with torch.no_grad():
            return self.stddev(torch.FloatTensor(X)).cpu().numpy().flatten()
