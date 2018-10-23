#!/usr/bin/env python3


import logging

import numpy as np
import sklearn
import sklearn.datasets
import sklearn.dummy
import sklearn.ensemble
import sklearn.linear_model
from sklearn.model_selection import train_test_split

import data
# TODO replace
from models import *
import models

from sklearn.model_selection import (GridSearchCV, train_test_split, KFold)


def main():
    logging.basicConfig(level=logging.INFO)

    X, y = data.preprocess(data.load())

    mods = (
        (ConfidenceRegressor, {"regression_cls": [sklearn.linear_model.Ridge],
                               "regression__alpha": [0, 1, 10, 100],
                               "reg_conf_split": [0.5, 0.75],
                               "confidence_cls": [sklearn.linear_model.Ridge],
                               "confidence__alpha": [0, 1, 10, 100],
         }),
    )

    for model_cls, params in mods:
        grid_search = sklearn.model_selection.GridSearchCV(
            model_cls(), params,
            scoring="neg_mean_squared_error", cv=5, n_jobs=-1)

        logging.info("Fitting")
        score = sklearn.model_selection.cross_val_score(
            grid_search, X, y, scoring="neg_mean_squared_error", cv=5)

        logging.info("Score: %.2f", np.sqrt(-score.mean()))


if __name__ == "__main__":
    main()
