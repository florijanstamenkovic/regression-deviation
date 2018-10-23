import numpy as np
import sklearn.linear_model
from sklearn.model_selection import train_test_split, GridSearchCV


def concat(ftrs, prediction):
    return np.hstack((ftrs, prediction.reshape(-1, 1)))


def error(prediction, y):
    return (prediction - y).abs()


# TODO see about bin medians or whatever
def error_at_retrieval(error, confidence, points=21):
    space = np.linspace(0, 1, points)
    retrieval = [(confidence >= c).mean() for c in space]

    errors = []
    for point in space:
        conf_thresh = np.interp(point, space, retrieval)
        mask = confidence >= conf_thresh
        errors.append(0 if mask.sum() == 0 else error[mask].mean())

    return np.array(errors), space


class RegressionConfidenceScorer():

    def __init__(self):
        pass

    def __call__(self, estimator, X,  y):
        prediction = estimator.predict(X)
        if isinstance(estimator, GridSearchCV):
            confidence = estimator.best_estimator_.predict_confidence(X, prediction)
        else:
            confidence = estimator.predict_confidence(X, prediction)

        errors, space = error_at_retrieval(error(prediction, y), confidence)
        result = sum(e * r for e, r in zip(errors, space)) / len(space)
        return -result


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
