#!/usr/bin/env python3


from argparse import ArgumentParser
import logging
import os

import numpy as np
import sklearn
import sklearn.datasets
import sklearn.dummy
import sklearn.ensemble
import sklearn.linear_model
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     train_test_split, KFold)

import matplotlib
matplotlib.use('Agg')  # ensure headless operation
from matplotlib import pyplot as plt

import data
# TODO replace
from models import *



OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_error_at_retrieval(errors, confidences, model_name):
    errors, space = error_at_retrieval(
        errors, confidences, reduced=False)
    plt.figure(figsize=(12, 9))
    plt.title("Error at retrieval: " + model_name)
    plt.plot(space, errors)
    plt.ylabel("Error")
    plt.xlabel("Retrieval")
    plt.grid(alpha=0.5, linestyle=':')
    plt.savefig(os.path.join(OUTPUT_DIR, "error_at_retrieval_%s.pdf" %
                             model_name), bbox_inches='tight')
    plt.close()

MODELS = (
    (ConfidenceRegressor, "dummy",
     {
        "regression_cls": [sklearn.dummy.DummyRegressor],
        "reg_conf_split": [0.5],
        "confidence_cls": [sklearn.dummy.DummyRegressor],
    }),
    (ConfidenceRegressor, "ridge_ridge",
     {
        "regression_cls": [sklearn.linear_model.Ridge],
        "regression__alpha": [0, 1, 10, 100],
        "reg_conf_split": [0.5],
        "confidence_cls": [sklearn.linear_model.Ridge],
        "confidence__alpha": [0, 1, 10, 100],
    }),
    (ConfidenceRegressor, "rf_rf",
     {
        "regression_cls": [sklearn.ensemble.RandomForestRegressor],
        "regression__n_estimators": [100],
        "reg_conf_split": [0.5],
        "confidence_cls": [sklearn.ensemble.RandomForestRegressor],
        "confidence__n_estimators": [100],
    }),
    (GaussianProcessEnsemble, "gaussian_process",
     {
    }),
)


def parse_args():
    argp = ArgumentParser()
    argp.add_argument(
        "--model", default=None, choices=tuple(m[1] for m in MODELS),
        nargs="+",
        help="Force which model is evaluated. Default evaluates all.")
    return argp.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    X, y = data.preprocess(data.load())

    if args.model is None:
        models = MODELS
    else:
        models = [m for m in MODELS if m[1] in args.model]

    for model_cls, model_name, params in models:
        logging.info("Fitting model %s", model_name)

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

        errors = error(predictions, y)
        logging.info("Model: %s, Score: %.2f", model_name,
                     error_at_retrieval(errors, confidences))
        plot_error_at_retrieval(errors, confidences, model_name)


if __name__ == "__main__":
    main()
