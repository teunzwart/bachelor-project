"""Tools for (statistical) analysis of Monte Carlo simulation."""

import itertools
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import exact_ising_model as exact
import plotting

SIMULATION_FOLDER = "./simulation_runs"


def open_simulation_files(data_files):
    """Unpickle simulation data."""
    data = []
    for d in data_files:
        with open("{0}/{1}".format(SIMULATION_FOLDER, d), 'rb') as f:
            data.append(pickle.load(f))

    return data


def data_analysis(data_files, save=False, show_plots=True, exact_ising=True):
    """If save=True, save plots."""
    data = open_simulation_files(data_files)
    energies = []
    energy_correlations = []
    magnetizations = []
    magnetization_correlations = []
    cluster_fractions = []
    heat_capacities = []
    magnetizabilities = []
    binder_cumulants = []
    for d in data:
        energy_at_size = []
        energy_correlations_at_size = []
        cluster_fraction_at_size = []
        heat_capacity_at_size = []
        magnetization_at_size = []
        magnetization_correlations_at_size = []
        magnetizability_at_size = []
        binder_cumulant_at_size = []

        for t in d:
            (lattice_size, bond_energy, initial_temperature, thermalization_sweeps, measurement_sweeps,
             temperature, correction_factor, correction_factor_error) = t[0]
            measurements = t[1]
            heat_cap_jack, heat_cap_error_jack, _ = jackknife(measurements['energy bins'], measurements['energy sq bins'], 8, heat_capacity, temperature=temperature, no_of_sites=lattice_size**2)
            heat_cap_boot, heat_cap_error_boot = bootstrap(measurements['energy bins'], measurements['energy sq bins'], 1000, heat_capacity, temperature=temperature, no_of_sites=lattice_size**2)
            heat_cap = np.mean([heat_cap_jack, heat_cap_boot])
            heat_cap_error = max(heat_cap_error_jack, heat_cap_error_boot)

            # Magnetizability.
            chi_jack, chi_error_jack, _ = jackknife(measurements['m bins'], measurements['mag sq bins'], 8, magnetizability, temperature=temperature, no_of_sites=lattice_size**2)
            chi_boot, chi_error_boot = bootstrap(measurements['m bins'], measurements['mag sq bins'], 1000, magnetizability, temperature=temperature, no_of_sites=lattice_size**2)
            chi = np.mean([chi_jack, chi_boot])
            chi_error = max(chi_error_jack, chi_error_boot)

            binder_jack, binder_error_jack, _ = jackknife(measurements['mag sq bins'], measurements['mag 4th bins'], 8, binder_cumulant)
            binder_boot, binder_error_boot = bootstrap(measurements['mag sq bins'], measurements['mag 4th bins'], 1000, binder_cumulant)
            binder = np.mean([binder_jack, binder_boot])
            binder_error = max(binder_error_jack, binder_error_boot)

            energy_at_size.append((temperature, measurements['energy'], measurements['energy error']))
            energy_correlations_at_size.append((temperature, measurements['energy correlation']))
            magnetization_at_size.append((temperature, measurements['m'], measurements['m error']))
            cluster_fraction_at_size.append((temperature, correction_factor, correction_factor_error))
            heat_capacity_at_size.append((temperature, heat_cap, heat_cap_error))
            magnetizability_at_size.append((temperature, chi, chi_error))
            binder_cumulant_at_size.append((temperature, binder, binder_error))
            magnetization_correlations_at_size.append((temperature, measurements['m correlation']))

        energies.append([lattice_size, energy_at_size])
        energy_correlations.append([lattice_size, energy_correlations_at_size])
        magnetization_correlations.append([lattice_size, magnetization_correlations_at_size])
        magnetizations.append([lattice_size, magnetization_at_size])
        cluster_fractions.append([lattice_size, cluster_fraction_at_size])
        heat_capacities.append([lattice_size, heat_capacity_at_size])
        magnetizabilities.append([lattice_size, magnetizability_at_size])
        binder_cumulants.append([lattice_size, binder_cumulant_at_size])

    # Find the critical temperature.
    critical_temperature, critical_temperature_error = find_binder_intersection(binder_cumulants)

    bond_energy = data[0][0][0][1]
    if exact_ising:
        exact_heat = exact.heat_capacity(bond_energy, 0, 10 * np.absolute(bond_energy))
        exact_energy = exact.internal_energy(bond_energy, 0, 10 * np.absolute(bond_energy))
        exact_magnetization = exact.magnetization(bond_energy, 0, 10 * np.absolute(bond_energy))
    else:
        exact_heat = None
        exact_energy = None
        exact_magnetization = None

    if show_plots:
        plotting.plot_quantity_range(energies, "Energy per Site", exact=exact_energy, save=save)
        plotting.plot_quantity_range(cluster_fractions, "Mean Cluster Size as fraction of Lattice", save=save)
        plotting.plot_quantity_range(magnetizations, "Absolute Magnetization per Site", exact=exact_magnetization, save=save)
        plotting.plot_quantity_range(heat_capacities, "Heat Capacity per Site", exact=exact_heat, save=save)
        plotting.plot_quantity_range(magnetizabilities, "Magnetizability per Site", save=save)
        plotting.plot_quantity_range(binder_cumulants, "Binder Cumulant", save=save)
        plotting.plot_correlation_time_range(energy_correlations, "Energy per Site", save=save)
        plotting.plot_correlation_time_range(magnetization_correlations, "Absolute Magnetization", save=save)

    return critical_temperature, critical_temperature_error, magnetizabilities, magnetizations, heat_capacities


def find_critical_exponents(critical_temperature, critical_temperature_error, magnetizabilities, magnetizations, heat_capacities, alpha, beta, gamma, nu):
    # Sanity check.
    if not 0 <= alpha < 1:
        raise ValueError("Alpha should be in the interval [0, 1), alpha is {0}".format(alpha))
    if critical_temperature and critical_temperature_error:
        data_collapse(magnetizabilities, "Magnetizability", critical_temperature, gamma, nu, "Gamma", "Nu")
        data_collapse(magnetizations, "Magnetization", critical_temperature, beta, nu, "Beta", "Nu")
        data_collapse(heat_capacities, "Heat Capacity", critical_temperature, alpha, nu, "Alpha", "Nu")

        critical_exponent_consistency(gamma, alpha, beta, nu)

    else:
        print("No data collapse could be performed.")


def bootstrap(data1, data2, no_of_resamples, operation, **kwargs):
    """Calculate error using the bootstrap method."""
    resamples = np.empty(no_of_resamples)
    for k in range(no_of_resamples):
        random_picks1 = np.random.choice(data1, len(data1))
        random_picks2 = np.random.choice(data2, len(data2))
        resamples.put(k, operation(random_picks1, random_picks2, kwargs))

    error = calculate_error(resamples)
    return np.mean(resamples), error


def calculate_error(data):
    """Calculate the error on a data set."""
    return np.std(data) / np.sqrt(len(data))


def binning(data, quantity, show_plot=False):
    """
    Calculate autocorrelation time, mean and error for a quantity using the binning method.

    The bins become uncorrelated when the error approaches a constant.
    These uncorrelated bins can be used for jackknife resampling.
    """
    original_length = len(data)
    errors = []
    errors.append((original_length, calculate_error(data)))
    while len(data) > 128:
        data = np.asarray([(a + b) / 2 for a, b in zip(data[::2], data[1::2])])
        errors.append((len(data), calculate_error(data)))
    autocorrelation_time = 0.5 * ((errors[-1][1] / errors[0][1])**2 - 1)
    if np.isnan(autocorrelation_time) or autocorrelation_time <= 0:
        autocorrelation_time = 1

    if show_plot:
        plt.title("Binning Method {0} Error, Log Scale".format(quantity))
        plt.xlabel("Data Points")
        plt.ylabel("Error")
        plt.xlim(original_length, 1)
        plt.ylim(ymin=0, ymax=max(errors, key=lambda x: x[1])[1] * 1.15)
        plt.semilogx([e[0] for e in errors], [e[1] for e in errors], basex=2)
        plt.show()

    return np.mean(data), errors[-1][1], autocorrelation_time, data


def jackknife(data1, data2, no_of_bins, operation, **kwargs):
    """
    Calculate errors using jackknife resampling.

    no_of_bins should divide len(data)
    """
    data_length = len(data1)
    all_bin_estimate = operation(data1, data2, kwargs)
    calculated_values = []
    split_data1 = np.split(data1, no_of_bins)
    split_data2 = np.split(data2, no_of_bins)
    # From https://stackoverflow.com/questions/28056195/
    mask = np.arange(1, no_of_bins) - np.tri(no_of_bins, no_of_bins - 1, k=-1, dtype=bool)
    leave_one_out1 = np.asarray(split_data1)[mask]
    leave_one_out2 = np.asarray(split_data2)[mask]
    for m1, m2 in zip(leave_one_out1, leave_one_out2):
        value = operation(np.concatenate(m1), np.concatenate(m2), kwargs)
        calculated_values.append(value)
    mean = np.sum(calculated_values) / no_of_bins
    standard_error = np.sqrt((1 - 1 / data_length) * (np.sum(np.asarray(calculated_values)**2 - mean**2)))
    bias = (no_of_bins - 1) * (mean - all_bin_estimate)
    if bias >= 0.5 * standard_error and bias != 0:
        print("Bias is large for {0}: error is {1}, bias is {2} ".format(operation, standard_error, bias))
    return all_bin_estimate, standard_error, bias


def heat_capacity(energy_data, energy_sq_data, kwargs):
    """
    Calculate the heat capacity for a given energy data set and temperature.

    Multiply by the number of sites, because the data has been normalised to the number of sites.
    """
    return kwargs['no_of_sites'] / kwargs['temperature']**2 * (np.mean(energy_sq_data) - np.mean(energy_data)**2)


def magnetizability(magnetization_data, magnetization_sq_data, kwargs):
    """Calculate the magnetizability."""
    return kwargs['no_of_sites'] / kwargs['temperature'] * (np.mean(magnetization_sq_data) - np.mean(magnetization_data)**2)


def binder_cumulant(magnetization_sq_data, magnetization_4th_data, kwargs):
    """Calculate the binder cumulant."""
    return 1 - np.mean(magnetization_4th_data) / (3 * np.mean(magnetization_sq_data)**2)


def find_binder_intersection(data):
    """Find the intersection of Binder cumulant data series for different lattice sizes."""
    # We use Cramers rule, adapted from http://stackoverflow.com/questions/20677795/
    intersections = []
    # Inerate over datasets.
    for k in range(len(data) - 1):
        data1 = data[k][1]
        data2 = data[k + 1][1]
        # Iterate over temperatures.
        for z in range(len(data1) - 1):
            intersection_error = []
            # Also calculate the intersection when the error has been added or subtracted.
            for e in [-1, 0, 1]:
                p1a = data1[z]
                p1b = data1[z + 1]
                dx1 = p1b[0] - p1a[0]
                dy1 = p1b[1] + e * p1b[2] - (p1a[1] + e * p1a[2])
                dydx1 = dy1 / dx1
                c1 = -dydx1 * p1a[0] + p1a[1]

                p2a = data2[z]
                p2b = data2[z + 1]
                dx2 = p2b[0] - p2a[0]
                dy2 = p2b[1] + e * p2b[2] - (p2a[1] + e * p2a[2])
                dydx2 = dy2 / dx2
                c2 = -dydx2 * p2a[0] + p2a[1]

                det = -dydx1 + dydx2
                if det == 0:
                    continue
                intersection_x = (c1 - c2) / det
                intersection_y = (dydx2 * c1 - dydx1 * c2) / det
                # The intersection should lie within the area of interest.
                if p1a[0] <= intersection_x <= p1b[0]:
                    intersection_error.append((intersection_x, intersection_y))

            if intersection_error:
                x_intersection = np.mean([i[0] for i in intersection_error])
                x_intersection_error = calculate_error([i[0] for i in intersection_error])
                y_intersection = np.mean([i[1] for i in intersection_error])
                y_intersection_error = calculate_error([i[1] for i in intersection_error])
                intersections.append(((x_intersection, x_intersection_error), (y_intersection, y_intersection_error)))
    if intersections:
        critical_temperature = np.mean([p[0][0] for p in intersections])
        critical_temperature_error = calculate_error([p[0][0] for p in intersections])#(1 / len(intersections)) * np.sqrt(sum([p[0][1]**2 for p in intersections]))
    else:
        critical_temperature = None
        critical_temperature_error = None
    print([i[0] for i in intersections])
    print("Critical temperature is {0} +/- {1}". format(critical_temperature, critical_temperature_error))

    return critical_temperature, critical_temperature_error


def chi_squared_data_collapse(data, critical_temperature,
                              critical_temperature_error):
    """Find the critical exponents for the magnetizability."""
    lowest_residual = np.inf
    best_nu = 0
    best_gamma = 0
    # for gamma in np.linspace(0.01, 2, 200):
    for nu in np.linspace(0.01, 1, 100):
        gamma = 1.8 * nu
        scaling_functions = []
        for d in data:
            scaling_function_at_size = []
            lattice_size = d[0]
            values = d[1]
            for v in values:
                t = (v[0] - critical_temperature) / critical_temperature
                scaling_variable = lattice_size**(1 / nu) * t
                v_tilde = lattice_size**(-gamma / nu) * v[1]
                scaling_function_at_size.append((scaling_variable, v_tilde))
            scaling_functions.append([lattice_size, scaling_function_at_size])

        at_interval = [a for a in list(itertools.chain(*[s[1] for s in scaling_functions])) if -1 <= a[0] <= 1]
        if len(at_interval) >= 15:

            at_interval_x, at_interval_y = zip(*at_interval)
            polynomial, residuals, a, b, c = np.polyfit(at_interval_x, at_interval_y, 5, full=True)

            if residuals / (len(at_interval) - 5) < lowest_residual:
                lowest_residual = residuals / (len(at_interval) - 6)
                best_gamma = gamma
                best_nu = nu
                print("gamma={0}, nu={1}, residual={2}".format(best_gamma, best_nu, residuals))
                f = np.poly1d(polynomial)

                x_new = np.linspace(min(at_interval_x), max(at_interval_x), 50)
                y_new = f(x_new)
                plt.figure()
                plt.plot(at_interval_x, at_interval_y, 'o', x_new, y_new)
                plt.show()
    print("best gamma = {0}, best nu = {1}".format(best_gamma, best_nu))


def data_collapse(data, quantity, critical_temperature, critical_exponent1, critical_exponent2, name1, name2):
    scaling_functions = []
    for d in data:
        scaling_function_at_size = []
        lattice_size = d[0]
        values = d[1]
        for v in values:
            t = (v[0] - critical_temperature) / critical_temperature
            scaling_variable = lattice_size**(1 / critical_exponent2) * t
            v_tilde = lattice_size**(-critical_exponent1 / critical_exponent2) * v[1]
            scaling_function_at_size.append((scaling_variable, v_tilde))
        scaling_functions.append([lattice_size, scaling_function_at_size])
    for p in scaling_functions:
        plt.xlabel("L^(1/nu)t")
        plt.ylabel("{0} Scaling Function".format(quantity))
        plt.plot([k[0] for k in p[1]], [k[1] for k in p[1]], linestyle='None', marker='o', label="{0} by {0} Lattice".format(p[0]))
    print("{0} = {1}, {2} = {3}".format(name1, critical_exponent1, name2, critical_exponent2))
    plt.legend(loc='best')
    plt.show()


def loglog_exponent_finding(data, quantity):
    lattice_sizes_log = []
    magnetizations_log = []
    for k in data:
        lattice_size_log = np.log(k[0])
        magnetization_log = np.log(k[1][0][1])
        lattice_sizes_log.append(lattice_size_log)
        magnetizations_log.append(magnetization_log)

    lattice_sizes = np.asarray(lattice_sizes_log)
    magnetizations_log = np.asarray(magnetizations_log)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(lattice_sizes, magnetizations_log)
    plt.xlabel("Log(L)")
    plt.ylabel("Log({0})".format(quantity))
    plt.plot(lattice_sizes, magnetizations_log, linestyle='None', marker='o')
    plt.show()
    return slope, std_err


def critical_exponent_consistency(gamma, alpha, beta, nu):
    delta = (2 - alpha) / beta - 1
    eta = 2 - gamma / nu
    print("Eta = {0}, Delta = {1}".format(eta, delta))

    print("2 nu + alpha = {0}, should be 2".format(2 * nu + alpha))
    print("alpha + 2 beta + gamma = {0}, should be 2".format(alpha + 2 * beta + gamma))
