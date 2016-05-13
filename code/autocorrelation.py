"""
Calculate autocorrelation functions.

See the following links for more implementation ideas:
- https://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation
- https://stackoverflow.com/questions/12269834/is-there-any-numpy-autocorrellation-function-with-standardized-output
"""

import numpy as np
import matplotlib.pyplot as plt


def autocorrelation(data):
    """
    Naive (SLOW!) way to calculate the autocorrelation function.

    Results are very similair to fast numpy version.
    """
    correlations = []
    for t in range(len(data)):
        tmax = len(data)
        upper_bound = tmax - t
        first_sum = 0
        second_sum = 0
        third_sum = 0

        for n in np.arange(upper_bound):
            first_sum += data[n] * data[n + t]
            second_sum += data[n]
            third_sum += data[n + t]

        correlation = (1 / upper_bound) * (first_sum - (1 / upper_bound) * second_sum * third_sum)
        correlations.append(correlation)

    normalized_acf = correlations / max(correlations)

    plt.title("Autocorrelation Function")
    plt.xlabel("Monte Carlo Sweeps")
    plt.plot(range(len(normalized_acf)), normalized_acf)
    plt.show()
    plt.title("Autocorrelation Function")
    plt.xlabel("Monte Carlo Sweeps")
    plt.plot(range(2500), normalized_acf[:2500])
    plt.show()

    correlation_time = np.ceil(np.trapz(normalized_acf[:4000]))

    return correlation_time, normalized_acf


def numpy_autocorrelation(data, show_plot=False):
    """Quick autocorrelation calculation."""
    data = np.asarray([d - np.mean(data) for d in data])
    acf = np.correlate(data, data, mode="full")[(len(data) - 1):]  # Only keep the usefull data (correlation is symmetric around index len(data)).
    normalized_acf = acf / acf.max()
    if show_plot:
        plt.title("Autocorrelation Function")
        plt.xlabel("Monte Carlo Sweeps")
        plt.plot(range(len(normalized_acf)), normalized_acf)
        plt.show()

        plt.title("Autocorrelation Function")
        plt.xlabel("Monte Carlo Sweeps")
        plt.plot(range(10), normalized_acf[:10])
        plt.show()

    correlation_time = np.trapz(normalized_acf)

    return correlation_time, normalized_acf


def calculate_error(data):
    """Calculate the error on a data set."""
    return np.std(data) / np.sqrt(len(data))


def sample_every_two_correlation_times(energy_data, magnetization_data, correlation_time, no_of_sites):
    """Sample the given data every 2 correlation times and determine value and error."""
    magnet_samples = []
    energy_samples = []

    for t in np.arange(0, len(energy_data), 2 * int(np.ceil(correlation_time))):
        magnet_samples.append(magnetization_data[t])
        energy_samples.append(energy_data[t])

    magnet_samples = np.asarray(magnet_samples)
    energy_samples = np.asarray(energy_samples)

    abs_magnetization = np.mean(np.absolute(magnet_samples))
    abs_magnetization_error = calculate_error(magnet_samples)
    print("<m> (<|M|/N>) = {0} +/- {1}".format(abs_magnetization, abs_magnetization_error))

    magnetization = np.mean(magnet_samples)
    magnetization_error = calculate_error(magnet_samples)
    print("<M/N> = {0} +/- {1}".format(magnetization, magnetization_error))

    energy = np.mean(energy_samples)
    energy_error = calculate_error(energy_samples)
    print("<E/N> = {0} +/- {1}".format(energy, energy_error))

    magnetization_squared = np.mean((magnet_samples * no_of_sites)**2)
    magnetization_squared_error = calculate_error((magnet_samples * no_of_sites)**2)
    print("<M^2> = {0} +/- {1}".format(magnetization_squared, magnetization_squared_error))
