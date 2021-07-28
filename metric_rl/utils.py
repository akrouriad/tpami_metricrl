import os
import pickle
import numpy as np
import scipy.stats as st


def save_parameters(log_name, params):
    file_name = os.path.join(log_name, 'parameters.pkl')
    with open(file_name, 'wb') as f:
        pickle.dump(params, f)


def get_mean_and_confidence(data):
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)
    interval, _ = st.t.interval(0.95, n-1, scale=se)
    return mean, interval


def plot_data(ax, x, use_median=False):
    if use_median:
        median = np.median(x, axis=0).squeeze()
        epochs = range(len(median))
        ax.plot(epochs, median)
        ax.fill_between(epochs, np.quantile(x, axis=0, q=.25), np.quantile(x, axis=0, q=.75), alpha=.5)
    else:
        mean, interval = get_mean_and_confidence(x)
        epochs = range(len(mean))
        ax.plot(epochs, mean)
        ax.fill_between(epochs, mean - interval, mean + interval, alpha=.5)
