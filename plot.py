import os

import numpy as np

import matplotlib
matplotlib.use('Agg')  # ensure headless operation
from matplotlib import pyplot as plt


OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_error_at_retrieval(abs_errors, stddevs, model_names, rmse, points=41):

    error_name = "RMSE" if rmse else "MAE"

    plt.figure(figsize=(12, 9))
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
    plt.savefig(os.path.join(OUTPUT_DIR, "%s_at_retrieval.pdf" % error_name),
                bbox_inches='tight')
    plt.close()


def plot_stddev_error_scatter(abs_error, stddev, model_name):
    plt.figure(figsize=(12, 9))
    plt.title("Deviation vs error scatter")
    plt.scatter(abs_error, stddev, s=2.0, alpha=0.2)
    plt.xlabel("Absolute error")
    plt.ylabel("Estimated standard deviation")
    plt.grid(alpha=0.5, linestyle=':')
    plt.savefig(os.path.join(OUTPUT_DIR, "std_scatter_%s.pdf" % model_name),
                bbox_inches='tight')
    plt.close()
