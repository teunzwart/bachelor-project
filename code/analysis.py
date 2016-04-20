"""Tools for (statistical) analysis of Monte Carlo simulation."""

import matplotlib.pyplot as plt
import numpy as np


def bootstrap_method(data, no_of_resamples, temperature, operation):
    """Calculate error using the bootstrap method."""
    resamples = np.empty(no_of_resamples)
    for k in range(no_of_resamples):
        random_picks = np.random.choice(data, len(data))
        resamples.put(k, operation(random_picks, temperature))

    #TODO: Divide by correct number.
    error = np.sqrt((np.mean(resamples**2) - np.mean(resamples)**2))
    return error


def calculate_error(data):
    """Calculate the error on a data set."""
    return np.std(data) / np.sqrt(len(data))


def binning_method(data, halfings, quantity, show_plot=False):
    """Calculate autocorrelation time, mean and error for a quantity using the binning method."""
    original_length = len(data)
    errors = []
    errors.append((original_length, calculate_error(data)))
    for n in range(halfings):
        if len(data) < 64:
            break
        data = np.asarray([(a + b) / 2 for a, b in zip(data[::2], data[1::2])])
        errors.append((len(data), calculate_error(data)))

    if show_plot:
        plt.title("Binning Method {0} Error, Log Scale".format(quantity))
        plt.xlabel("Data Points")
        plt.ylabel("Error")
        plt.xlim(original_length, 1)
        plt.ylim(ymin=0, ymax=max(errors, key=lambda x: x[1])[1] * 1.15)
        plt.semilogx([e[0] for e in errors], [e[1] for e in errors[::1]], basex=2)
        plt.show()

        # plt.title("Binning Method {0} Error".format(quantity))
        # plt.xlabel("Data Points")
        # plt.ylabel("Error")
        # plt.xlim(original_length, 1)
        # plt.ylim(ymin=0, ymax=max(errors, key=lambda x: x[1])[1] * 1.15)
        # plt.plot([e[0] for e in errors], [e[1] for e in errors[::1]])
        # plt.show()

    autocorrelation_time = 0.5 * ((max(errors, key=lambda x: x[1])[1] / errors[0][1])**2 - 1)
    if np.isnan(autocorrelation_time):
        autocorrelation_time = 1

    return autocorrelation_time, np.mean(data), max(errors, key=lambda x: x[1])[1]
