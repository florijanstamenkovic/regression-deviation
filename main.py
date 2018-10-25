#!/usr/bin/env python3


import logging

import numpy as np
import sklearn
import sklearn.datasets
import sklearn.dummy
import sklearn.ensemble
import sklearn.linear_model
from sklearn.model_selection import train_test_split, GridSearchCV

import data
# TODO replace
from models import *
import models

from sklearn.model_selection import (GridSearchCV, train_test_split, KFold)


def main():
    logging.basicConfig(level=logging.INFO)

    X, y = data.preprocess(data.load())

    mods = (
        (ConfidenceRegressor, {
            "regression_cls": [sklearn.dummy.DummyRegressor],
            "reg_conf_split": [0.5],
            "confidence_cls": [sklearn.dummy.DummyRegressor],
        }),
        # (ConfidenceRegressor, {
        #     "regression_cls": [sklearn.linear_model.Ridge],
        #     "regression__alpha": [0, 1, 10, 100],
        #     "reg_conf_split": [0.5],
        #     "confidence_cls": [sklearn.linear_model.Ridge],
        #     "confidence__alpha": [0, 1, 10, 100],
        # }),
        (ConfidenceRegressor, {
            "regression_cls": [sklearn.ensemble.RandomForestRegressor],
            "reg_conf_split": [0.5],
            "confidence_cls": [sklearn.ensemble.RandomForestRegressor],
        }),
    )

    for model_cls, params in mods:
        logging.info("Fitting model %s", model_cls.__name__)

        grid_search = GridSearchCV(
            model_cls(), params,
            scoring=RegressionConfidenceScorer(), cv=5, n_jobs=-1)

        # Calc error again as it's more stable with more points.
        predictions, confidences = [], []
        for train_index, test_index in KFold(5).split(X):
            grid_search.fit(X[train_index], y[train_index])
            prediction = grid_search.predict(X[test_index])
            predictions.append(prediction)
            confidences.append(grid_search.best_estimator_.predict_confidence(
                X[test_index], prediction))

        predictions = np.concatenate(predictions)
        confidences = np.concatenate(confidences)

        score = error_at_retrieval(error(predictions, y), confidences, 51)
        logging.info("Model: %s, Score: %.2f", model_cls.__name__,
                     error_at_retrieval(error(predictions, y), confidences))


if __name__ == "__main__":
    main()
