import os

import numpy as np

import matplotlib
matplotlib.use('Agg')  # ensure headless operation
from matplotlib import pyplot as plt


OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_error_at_retrieval(log_prob, stddev, model_name, points=41):
    confidence = stddev.max() - stddev
    confidence -= confidence.min()
    confidence /= confidence.max()

    space = np.linspace(0, 1, points)
    retrieval = [(confidence >= c).mean() for c in space]

    # Flip coordinates as retrieval must be increasing for np.interp
    space = space[::-1]
    retrieval = retrieval[::-1]

    errors = []
    for point in space:
        conf_thresh = np.interp(point, retrieval, space)
        mask = confidence >= conf_thresh
        errors.append(0 if mask.sum() == 0 else log_prob[mask].mean())

    plt.figure(figsize=(12, 9))
    plt.title("Log-prob at retrieval: " + model_name)
    plt.plot(space, errors)
    plt.ylabel("Log-prob")
    plt.xlabel("Retrieval")
    plt.grid(alpha=0.5, linestyle=':')
    plt.savefig(os.path.join(OUTPUT_DIR, "log_prob_at_retrieval_%s.pdf" %
                             model_name), bbox_inches='tight')
    plt.close()
