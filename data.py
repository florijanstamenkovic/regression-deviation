import itertools
import logging
import os
import shutil
import tempfile
from zipfile import ZipFile

from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn import datasets

URL_BIKE= "http://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
FILENAME_BIKE = "hour.csv"

URL_NEWS = "http://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip"
FILENAME_NEWS = "OnlineNewsPopularity.csv"


def download_and_extract(url, source, target):
        logging.info("Downloading file from: %s", url)
        archive_filename, _ = urlretrieve(url)
        tmpdir = tempfile.mkdtemp()
        with ZipFile(archive_filename, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        shutil.move(os.path.join(tmpdir, source), target)


def load_bike():
    if not os.path.exists(FILENAME_BIKE):
        download_and_extract(URL_BIKE, FILENAME_BIKE, FILENAME_BIKE)

    df = pd.read_csv(FILENAME_BIKE, index_col=0)

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

    # The last weather situation has only 3 samples set. We can't use that.
    del df["weathersit_4"]

    return df.values.astype(np.float32), target.values


def load_news():
    if not os.path.exists(FILENAME_NEWS):
        download_and_extract(URL_NEWS,
                             os.path.join("OnlineNewsPopularity", FILENAME_NEWS),
                             FILENAME_NEWS)
    df = pd.read_csv(FILENAME_NEWS, skipinitialspace=True)
    df = df.drop(["url", "timedelta"], axis=1)

    # Extract the target column.
    target = df["shares"]
    del df["shares"]

    return df.values.astype(np.float32), target.values


def load_boston():
    return datasets.load_boston(True)
