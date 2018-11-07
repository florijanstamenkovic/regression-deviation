#!/usr/bin/env python3

import math
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')  # ensure headless operation
from matplotlib import pyplot as plt


FIGSIZE = (8, 6)


def savefig(name):
    out_dir = "output"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "%s.pdf" % name), bbox_inches='tight')


def plot_error_at_retrieval(abs_errors, stddevs, model_names, rmse, points=41):

    error_name = "RMSE" if rmse else "MAE"

    plt.figure(figsize=FIGSIZE)
    plt.title("%s at retrieval" % error_name)

    for abs_error, stddev, model_name in zip(abs_errors, stddevs, model_names):
        stddev_space = np.linspace(stddev.min(), stddev.max(), points)
        retrieval = [(stddev <= s).mean() for s in stddev_space]

        retrieval_space = np.linspace(0, 1, points)

        errors = []
        for point in retrieval_space:
            stddev_thresh = np.interp(point, retrieval, stddev_space)
            mask = stddev <= stddev_thresh
            if mask.sum() == 0:
                error = 0
            else:
                error = abs_error[mask]
                error = (error ** 2).mean() ** 0.5 if rmse else error.mean()
            errors.append(error)

        # Don't plot zero retrievals
        mask = retrieval_space >= 0.1
        retrieval_space = retrieval_space[mask]
        errors = np.array(errors)[mask]

        plt.plot(retrieval_space, errors, label=model_name)

    plt.ylabel(error_name)
    plt.xlabel("Retrieval")
    plt.grid(alpha=0.5, linestyle=':')
    plt.legend()
    savefig("%s_at_retrieval" % error_name)
    plt.close()


def plot_stddev_error_scatter(abs_error, stddev, model_name):
    plt.figure(figsize=FIGSIZE)
    plt.title("Deviation vs error scatter")
    plt.scatter(abs_error, stddev, s=2.0, alpha=0.2)
    plt.xlabel("Absolute error")
    plt.ylabel("Estimated standard deviation")
    plt.grid(alpha=0.5, linestyle=':')
    savefig("std_scatter_%s" % model_name)
    plt.close()



def plot_pdfs():
    m = 0
    s = 1

    def gexp():
        return np.exp(-((x - m) ** 2)) / (2 * s)

    def gauss():
        return gexp() / (2 * math.pi * s) ** 0.5

    def gauss_dm():
        return gexp() * (x - m) / (2 * math.pi * s ** 3) ** 0.5

    def gauss_ds():
        return -gexp() * (s - (x - m) ** 2) / (2 * math.pi * s ** 5) ** 0.5

    def plot_gauss(name, plot_points):
        plt.figure(figsize=FIGSIZE)
        plt.title("Gaussian probability density function, mean=0, stddev=1")
        plt.plot(x, gauss(), label="N(m, s)")
        plt.plot(x, gauss_dm(), label="d/dm N(m, s)")
        plt.plot(x, gauss_ds(), label="d/ds N(m, s)")
        if plot_points:
            def plot_point(x, linestyle, label):
                plt.plot([x, x], [-0.2, 0.2], color="r", linestyle=linestyle,
                         label=label)
            plot_point(0.1, ":", "x0")
            plot_point(1, "-.", "x1")
            plot_point(2, "--", "x2")
        plt.grid(alpha=0.5, linestyle=':')
        plt.legend()
        savefig(name)
        plt.close()

    x = np.linspace(-3, 3, 300)
    plot_gauss("gauss", False)

    x = np.linspace(0, 3, 300)
    plot_gauss("gauss_points", True)

    x = np.linspace(-2, 2, 300)
    plt.figure(figsize=FIGSIZE)
    plt.title("Log of Gaussian probability density function, mean=0, stddev=1")
    plt.plot(x, np.log(gauss()), label="ln(N(m, s))")
    plt.plot(x, (x - m) / s, label="d/dm ln(N(m, s))")
    plt.plot(x, ((x - m) ** 2 - s) / (2 * s ** 2), label="d/ds ln(N(m, s))")
    plt.grid(alpha=0.5, linestyle=':')
    plt.legend()
    savefig("ln_gauss")
    plt.close()


if __name__ == "__main__":
    plot_pdfs()
