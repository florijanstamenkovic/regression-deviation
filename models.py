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


def normpdf(mean, stddev, prediction, log):
    var = stddev ** 2
    denom = (2 * math.pi * var) ** 0.5
    num = np.exp(-((mean - prediction) ** 2 / (2 * var)))
    prob = num / denom
    return np.log(prob + 0.000001) if log else prob


class RegressionDeviationScorer():

    def __init__(self):
        pass

    def __call__(self, estimator, X, y):
        prediction = estimator.predict(X)
        if isinstance(estimator, GridSearchCV):
            stddev = estimator.best_estimator_.predict_stddev(X)
        else:
            stddev = estimator.predict_stddev(X)

        return normpdf(y, stddev, prediction, True).mean()


class DeviationRegressor():

    def __init__(self, **params):
        self.set_params(**params)

    def _extract_params(self, prefix):
        return {k[len(prefix):]: v for k, v in self.params.items()
                if k.startswith(prefix)}

    def fit(self, X, y):
        params = self.get_params()

        reg_conf_split = params["reg_conf_split"]
        if reg_conf_split is None:
            X_reg, X_conf = X, X
            y_reg, y_conf = y, y
        else:
            X_reg, X_conf, y_reg, y_conf = train_test_split(
                X, y, test_size=params["reg_conf_split"])

        self.regression = params["regression_cls"](
            **self._extract_params("regression__"))
        self.regression.fit(X_reg, y_reg)

        self.stddev = params["stddev_cls"](
            **self._extract_params("stddev__"))

        self.stddev.fit(X_conf,
                        np.abs(self.regression.predict(X_conf) - y_conf))

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
    def predict_stddev(self, X):
        std = np.std([e.predict(X) for e in self.estimators_], axis=0, ddof=1)
        return std + 0.0000001


class TorchRegressor(nn.Module):

    def __init__(self, **params):
        super(TorchRegressor, self).__init__()
        self.set_params(**params)
        self.mnb_size = 5000
        self.epochs = 20

    def init_params(self, X, y):
        self.mean = nn.Linear(X.shape[1], 1)
        self.stddev = nn.Linear(X.shape[1], 1)
        self.optimizer_sgd = optim.SGD(self.parameters(), lr=0.05, momentum=0.1)
        self.optimizer_lbfgs = optim.LBFGS(self.parameters(), lr=0.8)

        if True:
            r = sklearn.linear_model.Ridge(alpha=10)
            r.fit(X, y)
            self.mean.bias.data.fill_(float(r.intercept_))
            self.mean.weight.data = torch.FloatTensor(r.coef_.reshape(1, -1))
        else:
            self.mean.bias.data.fill_(y.mean())
            self.mean.weight.data.uniform_(-0.1, 0.1)

        self.stddev.bias.data.fill_(y.std())
        self.stddev.weight.data.uniform_(-0.1, 0.1)

    def forward_mean(self, x):
        return self.mean(x)

    def forward_std(self, x):
        return torch.max(torch.tensor(0.001), self.stddev(x))

    def log_prob(self, x, mean, stddev):
        return -((2 * math.pi * stddev).log() + ((x - mean) ** 2) / stddev) / 2

    def fit_lbfgs(self, X, y):
        self.init_params(X, y)
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)

        def closure():
            self.optimizer_lbfgs.zero_grad()
            mean = self.forward_mean(X)
            stddev = self.forward_std(X)
            abs_dist = (y - mean).abs()
            log_prob = self.log_prob(y, mean, stddev)
            loss = -log_prob.mean()
            print('loss:', loss.item())
            return loss

        self.optimizer_lbfgs.step(closure)
        return self


    def fit_sgd(self, X, y):
        self.init_params(X, y)

        def batches():
            inds = np.random.shuffle(np.arange(len(X)))

            batch_count = (len(X) - 1) // self.mnb_size + 1
            for batch_ind in range(batch_count):
                start = self.mnb_size * batch_ind
                end = start + self.mnb_size
                yield (torch.FloatTensor(X[start:end]),
                       torch.FloatTensor(y[start:end]))

        def quantiles(t):
            t = t.cpu().detach().numpy()
            if t.ndim == 1 and len(t) == 1:
                return "%.4f" % float(t)
            return ", ".join("%.2f" % q for q in
                             np.quantile(t, np.linspace(0, 1, 5)))

        def param_quantiles(t):
            return "%s, grad: %s" % (quantiles(t), quantiles(t.grad))

        for epoch_ind in range(self.epochs):
            epoch_losses = []
            epoch_maes = []
            for batch_ind, (X_mnb, y_mnb) in enumerate(batches()):
                self.optimizer.zero_grad()
                mean = self.forward_mean(X_mnb)
                stddev = self.forward_std(X_mnb)
                abs_dist = (y_mnb - mean).abs()
                log_prob = self.log_prob(y_mnb, mean, stddev)
                loss = -log_prob.mean()
                epoch_losses.append(loss.item())
                epoch_maes.append(abs_dist.mean().item())

                loss.backward()
                nn.utils.clip_grad_value_(self.parameters(), 1)
                if batch_ind == 0:
                    logging.debug("\tMean: %s", quantiles(mean))
                    logging.debug("\tStddev: %s", quantiles(stddev))
                    logging.debug("\tAbs-dist: %s", quantiles(abs_dist))
                    logging.debug("\tLog-prob: %s", quantiles(log_prob))
                    logging.debug("\tMean weight: %s",
                                  param_quantiles(self.mean.weight))
                    logging.debug("\tMean bias: %s",
                                  param_quantiles(self.mean.bias))
                    logging.debug("\tStddev weight: %s",
                                  param_quantiles(self.stddev.weight))
                    logging.debug("\tStddev bias: %s",
                                  param_quantiles(self.stddev.bias))
                self.optimizer.step()

            logging.info(
                "TorchRegressor epoch %d, MAE %.2f, mean loss %.5f, "
                "first batch loss %.5f", epoch_ind, np.mean(epoch_maes),
                np.mean(epoch_losses), epoch_losses[0])

        return self

    def fit(self, X, y):
        self.fit_lbfgs(X, y)
        error = np.abs((self.predict(X) - y)).mean()
        print("train mae: ", error)
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
