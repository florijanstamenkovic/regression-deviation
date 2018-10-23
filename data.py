import itertools
import logging
import os
import shutil
import tempfile
from zipfile import ZipFile

from urllib.request import urlretrieve

import numpy as np
import pandas as pd

URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
HOUR_DATA_PATH = "hour.csv"


def load():
    if not os.path.exists(HOUR_DATA_PATH):
        logging.info("Downloading file from: %s", URL)
        filename, _ = urlretrieve(URL)
        tmpdir = tempfile.mkdtemp()
        with ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        shutil.move(os.path.join(tmpdir, "hour.csv"), HOUR_DATA_PATH)

    return pd.read_csv(HOUR_DATA_PATH, index_col=0)


def preprocess(df):
    # Delete columns we won't use.
    to_delete = ["dteday", "yr", "mnth", "weekday", "atemp", "casual",
                 "registered"]
    df = df.drop(to_delete, axis=1)

    # Convert hours to part-of-day
    df["part_of_day"] = df["hr"].astype(np.int64) / 6
    del df["hr"]

    # Convert categoric columns to one-hot.
    for col in ["season", "part_of_day", "weathersit"]:
        col_int = df[col].astype(np.int32)
        for value in col_int.unique():
            df[col + "_" + str(value)] = col_int == value
        del df[col]

    # Extract the target column.
    target = df["cnt"]
    del df["cnt"]

    return df, target


def cross_cor(df):
    logging.info("Feature cross correlation:")
    for ftr_a, ftr_b in itertools.combinations(df.columns, 2):
        r = (np.corrcoef(df[ftr_a], df[ftr_b], rowvar=True))[0, 1]
        logging.info("%s, %s - %.2f", ftr_a, ftr_b, r)
