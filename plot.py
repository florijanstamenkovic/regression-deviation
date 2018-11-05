import os

import numpy as np

import matplotlib
matplotlib.use('Agg')  # ensure headless operation
from matplotlib import pyplot as plt


OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_error_at_retrieval(abs_errors, stddevs, model_names, points=41):

    plt.figure(figsize=(12, 9))
    plt.title("MAE at retrieval")

    for abs_error, stddev, model_name in zip(abs_errors, stddevs, model_names):
        stddev_space = np.linspace(stddev.min(), stddev.max(), points)
        retrieval = [(stddev <= s).mean() for s in stddev_space]

        retrieval_space = np.linspace(0, 1, points)

        errors = []
        for point in retrieval_space:
            stddev_thresh = np.interp(point, retrieval, stddev_space)
            mask = stddev <= stddev_thresh
            errors.append(0 if mask.sum() == 0 else abs_error[mask].mean())

        # Don't plot zero retrievals
        mask = retrieval_space >= 0.1
        retrieval_space = retrieval_space[mask]
        errors = np.array(errors)[mask]

        plt.plot(retrieval_space, errors, label=model_name)

    plt.ylabel("MAE")
    plt.xlabel("Retrieval")
    plt.grid(alpha=0.5, linestyle=':')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "mae_at_retrieval.pdf"),
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
