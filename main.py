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

import data
import models
import plot


MODELS = (
    (models.ConfidenceRegressor, "dummy",
     {
         "regression_cls": [sklearn.dummy.DummyRegressor],
         "reg_conf_split": [0.5],
         "stddev_cls": [sklearn.dummy.DummyRegressor],
     }),
    (models.ConfidenceRegressor, "ridge_ridge",
     {
         "regression_cls": [sklearn.linear_model.Ridge],
         "regression__alpha": [0, 1, 10, 100],
         "reg_conf_split": [0.5],
         "stddev_cls": [sklearn.linear_model.Ridge],
         "stddev__alpha": [0, 1, 10, 100],
     }),
    (models.ConfidenceRegressor, "rf_rf",
     {
         "regression_cls": [sklearn.ensemble.RandomForestRegressor],
         "regression__n_estimators": [100],
         "reg_conf_split": [0.5],
         "stddev_cls": [sklearn.ensemble.RandomForestRegressor],
         "stddev__n_estimators": [100],
     }),
    (models.GaussianProcessEnsemble, "gaussian_process",
     {
     }),
)


def parse_args():
    argp = ArgumentParser()
    argp.add_argument(
        "--model", default=None, choices=tuple(m[1] for m in MODELS),
        nargs="+",
        help="Force which model is evaluated. Default evaluates all.")
    argp.add_argument("--n-jobs", type=int, default=-1,
                      help="Number of jobs in cross validation. Default all "
                      "CPUs.")
    argp.add_argument("--limit-dataset", action="store_true",
                      help="If the dataset should be reduced for debugging.")
    return argp.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    X, y = data.preprocess(data.load())
    if args.limit_dataset:
        X = X[:500]
        y = y[:500]

    if args.model is None:
        used_models = models.MODELS
    else:
        used_models = [m for m in MODELS if m[1] in args.model]

    for model_cls, model_name, params in used_models:
        logging.info("Fitting model %s", model_name)

        grid_search = GridSearchCV(
            model_cls(), params, cv=5, n_jobs=args.n_jobs,
            scoring=models.RegressionConfidenceScorer())

        # Calc error again as it's more stable with more points.
        predictions, stddevs = [], []
        for train_index, test_index in KFold(5).split(X):
            # Transform the data
            scaler = sklearn.preprocessing.StandardScaler()
            X_train = scaler.fit_transform(X[train_index])
            grid_search.fit(X_train, y[train_index])

            prediction = grid_search.predict(scaler.transform(X[test_index]))
            predictions.append(prediction)
            stddevs.append(grid_search.best_estimator_.predict_stddev(
                X[test_index], prediction))

        predictions = np.concatenate(predictions)
        log_probs = models.log_prob(predictions, y)
        stddevs = np.concatenate(stddevs)

        logging.info("Model: %s, MAE: %.2f, mean log-prob: %.2f",
                     model_name, np.abs(predictions - y).mean(), log_probs.mean())
        plot.plot_error_at_retrieval(log_probs, stddevs, model_name)


if __name__ == "__main__":
    main()
