"""Tools for (statistical) analysis of Monte Carlo simulation."""

import copy

import matplotlib.pyplot as plt
import numpy as np
import scipy


def bootstrap(data, no_of_resamples, operation, **kwargs):
    """Calculate error using the bootstrap method."""
    resamples = np.empty(no_of_resamples)
    for k in range(no_of_resamples):
        random_picks = np.random.choice(data, len(data))
        resamples.put(k, operation(random_picks, kwargs))

    # For bootstrap, the standard error == the standard deviation.
    # TODO: Check this with literature.
    error = np.sqrt((np.mean(resamples**2) - np.mean(resamples)**2))
    return np.mean(resamples), error


def calculate_error(data):
    """Calculate the error on a data set."""
    return np.std(data) / np.sqrt(len(data))


def binning(data, halfings, quantity, show_plot=False):
    """Calculate autocorrelation time, mean and error for a quantity using the binning method."""
    original_length = len(data)
    errors = []
    data_sets = []
    errors.append((original_length, calculate_error(data)))
    data_sets.append(data)
    for n in range(halfings):
        if len(data) < 64:
            break
        data = np.asarray([(a + b) / 2 for a, b in zip(data[::2], data[1::2])])
        errors.append((len(data), calculate_error(data)))
        data_sets.append(data)
    max_error = max(errors, key=lambda x: x[1])
    max_error_index = errors.index(max_error)

    autocorrelation_time = 0.5 * ((max_error[1] / errors[0][1])**2 - 1)
    if np.isnan(autocorrelation_time):
        autocorrelation_time = 1

    if show_plot:
        plt.title("Binning Method {0} Error, Log Scale".format(quantity))
        plt.xlabel("Data Points")
        plt.ylabel("Error")
        plt.xlim(original_length, 1)
        plt.ylim(ymin=0, ymax=max_error[1] * 1.15)
        plt.semilogx([e[0] for e in errors], [e[1] for e in errors], basex=2)
        plt.show()

    return np.mean(data), max_error[1], autocorrelation_time, data_sets[max_error_index]


def jackknife(data, no_of_bins, operation, **kwargs):
    """
    Calculate errors using jackknife resampling.

    no_of_bins should divide len(data)
    """
    data_length = len(data)
    all_bin_estimate = operation(data, kwargs)
    calculated_values = []
    split_data = np.split(data, no_of_bins)
    # From https://stackoverflow.com/questions/28056195/python-leave-one-out-estimation
    mask = np.arange(1, no_of_bins) - np.tri(no_of_bins, no_of_bins - 1, k=-1, dtype=bool)
    leave_one_out = np.asarray(split_data)[mask]
    for m in leave_one_out:
        value = operation(np.concatenate(m), kwargs)
        calculated_values.append(value)
    mean = np.sum(calculated_values) / no_of_bins
    standard_error = np.sqrt((1 - 1 / data_length) * (np.sum(np.asarray(calculated_values)**2 - mean**2)))
    bias = (no_of_bins - 1) * (mean - all_bin_estimate)
    if bias >= 0.5 * standard_error and bias != 0:
        print("Bias is large for {0}: error is {1}, bias is {2} ".format(operation, standard_error, bias))
    return all_bin_estimate, standard_error, bias


def heat_capacity(energy_data, kwargs):
    """
    Calculate the heat capacity for a given energy data set and temperature.

    Multiply by the number of sites, because the data has been normalised to the number of sites.
    """
    return kwargs['no_of_sites'] / kwargs['temperature']**2 * (np.mean(energy_data**2) - np.mean(energy_data)**2)


def magnetizability(magnetization_data, kwargs):
    """Calculate the magnetizability."""
    return kwargs['no_of_sites'] / kwargs['temperature'] * (np.mean(magnetization_data**2) - np.mean(np.absolute(magnetization_data))**2)


def binder_cumulant(magnetization_data, kwargs):
    """Calculate the binder cumulant."""
    return 1 - np.mean(magnetization_data**4) / (3 * np.mean(magnetization_data**2)**2)


def find_binder_intersection(data):
    """Find the intersection of Binder cumulant data series for different lattice sizes."""
    # Adapted from http://stackoverflow.com/questions/8094374/
    intersections = []
    # We do not want to operate on the original list.
    data = copy.deepcopy(data)
    while len(data) >= 2:
        data_set = data.pop(0)[1]
        x1, y1, _ = zip(*data_set)
        x1, y1 = np.asarray(x1), np.asarray(y1)
        p1 = scipy.interpolate.PiecewisePolynomial(x1, y1[:, np.newaxis])
        for p in data:
            x2, y2, _ = zip(*p[1])
            x2, y2 = np.asarray(x2), np.asarray(y2)
            p2 = scipy.interpolate.PiecewisePolynomial(x2, y2[:, np.newaxis])
            intersections.append(scipy.optimize.fsolve(lambda x: p1(x) - p2(x), 2)[0])

    intersection = np.mean(intersections)
    intersection_error = calculate_error(intersections)
    return intersection, intersection_error
