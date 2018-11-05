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
    (models.ConfidenceRegressor, "dummy", False,
     {
         "regression_cls": [sklearn.dummy.DummyRegressor],
         "reg_conf_split": [0.5],
         "stddev_cls": [sklearn.dummy.DummyRegressor],
     }),
    (models.ConfidenceRegressor, "ridge", False,
     {
         "regression_cls": [sklearn.linear_model.Ridge],
         "regression__alpha": [0, 1, 10, 1000],
         "reg_conf_split": [0.5, None],
         "stddev_cls": [sklearn.linear_model.Ridge],
         "stddev__alpha": [0, 1, 10, 1000],
     }),
    (models.ConfidenceRegressor, "linear", False,
     {
         "regression_cls": [sklearn.linear_model.LinearRegression],
         "reg_conf_split": [0.5, None],
         "stddev_cls": [sklearn.linear_model.LinearRegression],
     }),
    (models.ConfidenceRegressor, "rf", False,
     {
         "regression_cls": [sklearn.ensemble.RandomForestRegressor],
         "regression__n_estimators": [100],
         "reg_conf_split": [0.5, None],
         "stddev_cls": [sklearn.ensemble.RandomForestRegressor],
         "stddev__n_estimators": [100],
     }),
    (models.GaussianProcessEnsemble, "gaussian_process", False,
     {
     }),
    (models.RandomForestStdRegressor, "random_forest", False,
     {
         "n_estimators": [100],
     }),
    (models.TorchRegressor, "torch", True,
     {
     }),
)


def parse_args():
    argp = ArgumentParser()
    argp.add_argument("--logging", default=logging.INFO,
                      choices=["INFO", "DEBUG"], help="Logging level")
    argp.add_argument(
        "--model", default=None, choices=tuple(m[1] for m in MODELS), nargs="+",
        help="Force which model is evaluated. Default evaluates all.")
    argp.add_argument("--n-jobs", type=int, default=-1,
                      help="Number of jobs in cross validation. Default all "
                      "CPUs.")
    argp.add_argument("--dataset", choices=["bike", "news", "boston"],
                      default="bike",
                      help="Which dataset to use")
    argp.add_argument("--limit-dataset", type=int, default=None,
                      help="If the dataset should be reduced for debugging.")
    argp.add_argument("--test-size", type=float, default=0.4,
                      help="Part of the dataset that's the test set")
    return argp.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(level=args.logging)

    X, y = getattr(data, "load_" + args.dataset)()

    if args.limit_dataset is not None:
        X = X[:args.limit_dataset]
        y = y[:args.limit_dataset]

    logging.info("Using the '%s' dataset, %d rows, %d features",
                 args.dataset, X.shape[0], X.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=2345)

    if args.model is None:
        used_models = models.MODELS
    else:
        used_models = [m for m in MODELS if m[1] in args.model]

    for model_cls, model_name, scale_target, params in used_models:
        logging.info("Fitting model %s", model_name)

        if scale_target:
            y_mean = y_train.mean()
            y_std = y_train.std()
            y_train = (y_train - y_mean) / y_std
            y_test = (y_test - y_mean) / y_std

        grid_search = GridSearchCV(
            model_cls(), params, cv=5, n_jobs=args.n_jobs,
            scoring=models.RegressionConfidenceScorer(), iid=True)

        # Transform the data
        scaler = sklearn.preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        grid_search.fit(X_train, y_train)

        prediction = grid_search.predict(scaler.transform(X_test))
        stddev = grid_search.best_estimator_.predict_stddev(X_test)

        log_probs = models.normpdf(y_test, stddev, prediction, True)

        if scale_target:
            y_test = y_test * y_std + y_mean
            prediction = prediction * y_std + y_mean

        mae = np.abs(prediction - y_test)
        rmse = (mae ** 2).mean() ** 0.5

        logging.info("Model: %s, MAE: %.2f, RMSE: %.2f, mean log-prob: %.2f",
                     model_name, mae.mean(), rmse, log_probs.mean())
        plot.plot_error_at_retrieval(mae, stddev, model_name)


if __name__ == "__main__":
    main()
