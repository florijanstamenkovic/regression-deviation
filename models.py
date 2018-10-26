import numpy as np
import sklearn.linear_model
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.gaussian_process import GaussianProcessRegressor


def concat(ftrs, prediction):
    return np.hstack((ftrs, prediction.reshape(-1, 1)))


def error(prediction, y):
    return np.abs(prediction - y)


# TODO see about bin medians or whatever
def error_at_retrieval(error, confidence, reduced=True, points=51):
    assert(confidence.min() >= 0 and confidence.max() <= 1)

    space = np.linspace(0, 1, points)
    retrieval = [(confidence >= c).mean() for c in space]

    # Flip coordinates as retrieval must be increasing for np.interp
    space = space[::-1]
    retrieval = retrieval[::-1]

    errors = []
    for point in space:
        conf_thresh = np.interp(point, retrieval, space)
        mask = confidence >= conf_thresh
        errors.append(error.max() if mask.sum() == 0 else error[mask].mean())

    if not reduced:
        return np.array(errors), space

    return sum(e * r for e, r in zip(errors, space)) / len(space)


class RegressionConfidenceScorer():

    def __init__(self):
        pass

    def __call__(self, estimator, X,  y):
        prediction = estimator.predict(X)
        if isinstance(estimator, GridSearchCV):
            confidence = estimator.best_estimator_.predict_confidence(X, prediction)
        else:
            confidence = estimator.predict_confidence(X, prediction)

        return -error_at_retrieval(error(prediction, y), confidence)


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

        self.confidence = params["confidence_cls"](
            **self._extract_params("confidence__"))

        # Fit confidence.
        regression_pred = self.regression.predict(X_conf)
        ftrs = concat(X_conf, regression_pred)
        # TODO make error a hyperparameter
        self.confidence.fit(ftrs, error(regression_pred, y_conf))
        prediction = self.confidence.predict(ftrs)
        self.min_conf = prediction.min()
        self.max_conf = prediction.max()

    def predict(self, X):
        return self.regression.predict(X)

    def predict_confidence(self, X, prediction):
        ftrs = concat(X, prediction)
        prediction = self.confidence.predict(ftrs)
        prediction = (prediction - self.min_conf) / self.max_conf
        return np.maximum(0, np.minimum(1, prediction))

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
        for (X_fold, y_fold), gp in zip(fold.split(X, y), self.gps):
            gp.fit(X_fold, y_fold)

    def predict(self, X):
        return np.mean(gp.predict(X) for gp in self.gps)

    def predict_confidence(self, X, _):
        predictions = [gp.predict(X, return_std=True) for gp in self.gps]
        means, std = zip(*predictions)
        raise Exception("Not yet implemented")

        return self.gp.predict(X, return_std=True)[1]

    def set_params(self, **params):
        self.params = params
        for gp in self.gps:
            gp.set_params(**params)
        return self

    def get_params(self, deep=True):
        return self.params
