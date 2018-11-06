#!/usr/bin/env python3


from argparse import ArgumentParser
import logging
import os

import numpy as np
import sklearn
import sklearn.datasets
import sklearn.dummy
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
import sklearn.linear_model
from sklearn.model_selection import GridSearchCV, train_test_split, KFold

import data
import models
import plot


MODELS = (
    (models.DeviationRegressor, "deviation_dummy",
     {
         "regression_cls": [sklearn.dummy.DummyRegressor],
         "reg_conf_split": [0.5],
         "stddev_cls": [sklearn.dummy.DummyRegressor],
     }),
    (models.DeviationRegressor, "deviation_ridge",
     {
         "regression_cls": [Ridge],
         "regression__alpha": [0, 1, 10, 1000],
         "reg_conf_split": [0.5, None],
         "stddev_cls": [Ridge],
         "stddev__alpha": [0, 1, 10, 1000],
     }),
    (models.DeviationRegressor, "deviation_linear",
     {
         "regression_cls": [LinearRegression],
         "reg_conf_split": [0.5, None],
         "stddev_cls": [LinearRegression],
     }),
    (models.DeviationRegressor, "deviation_random_forest",
     {
         "regression_cls": [RandomForestRegressor],
         "regression__n_estimators": [100],
         "reg_conf_split": [0.5, None],
         "stddev_cls": [RandomForestRegressor],
         "stddev__n_estimators": [100],
     }),
    (models.BaggingRegressor, "bagging",
     {"base_estimator": [None, Ridge(alpha=10)], "n_estimators": [100]}),
    (models.TorchRegressor, "torch", {}),
    (models.KNeighborsRegressor, "knn", {}),
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

    # Results containing (abs_error, stddev, model_name)
    results = []

    for model_cls, model_name, params in used_models:
        logging.info("Fitting model %s", model_name)

        grid_search = GridSearchCV(
            model_cls(), params, cv=5, n_jobs=args.n_jobs,
            scoring=models.RegressionDeviationScorer(), iid=True)

        # Transform the input to zero-mean unit-var
        scaler = sklearn.preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        grid_search.fit(X_train, y_train)

        prediction = grid_search.predict(scaler.transform(X_test))
        stddev = grid_search.best_estimator_.predict_stddev(X_test)

        results.append((np.abs(prediction - y_test), stddev, model_name))

    for abs_error, stddev, model_name in results:
        rmse = (abs_error ** 2).mean() ** 0.5
        logging.info("Model: %s, MAE: %.2f, RMSE: %.2f",
                     model_name, abs_error.mean(), rmse)
        plot.plot_stddev_error_scatter(abs_error, stddev, model_name)

    plot.plot_error_at_retrieval(*zip(*results))


if __name__ == "__main__":
    main()
