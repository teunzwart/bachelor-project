"""Tools for (statistical) analysis of Monte Carlo simulation."""

import itertools
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns

import exact_ising_model as exact
import plotting

plt.rc('text', usetex=True)
sns.set_style("ticks")
sns.set_palette('colorblind')  # Options: deep, muted, pastel, bright, dark, colorblind

SIMULATION_FOLDER = "./simulation_runs"
SAVE_LOCATION = "./analysis_images"


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
    energies = {}
    energy_correlations = {}
    magnetizations = {}
    magnetization_correlations = {}
    cluster_fractions = {}
    heat_capacities = {}
    magnetizabilities = {}
    binder_cumulants = {}
    for d in data:
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

            energies.setdefault(lattice_size, []).append((temperature, measurements['energy'], measurements['energy error']))
            energy_correlations.setdefault(lattice_size, []).append((temperature, measurements['energy correlation']))
            magnetizations.setdefault(lattice_size, []).append((temperature, measurements['m'], measurements['m error']))
            cluster_fractions.setdefault(lattice_size, []).append((temperature, correction_factor, correction_factor_error))
            heat_capacities.setdefault(lattice_size, []).append((temperature, heat_cap, heat_cap_error))
            magnetizabilities.setdefault(lattice_size, []).append((temperature, chi, chi_error))
            binder_cumulants.setdefault(lattice_size, []).append((temperature, binder, binder_error))
            magnetization_correlations.setdefault(lattice_size, []).append((temperature, measurements['m correlation']))

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
        plotting.plot_quantity_range(cluster_fractions, "Mean Cluster Size as Fraction of Lattice", save=save)
        plotting.plot_quantity_range(magnetizations, "Absolute Magnetization per Site", exact=exact_magnetization, save=save)
        plotting.plot_quantity_range(heat_capacities, "Heat Capacity per Site", exact=exact_heat, save=save)
        plotting.plot_quantity_range(magnetizabilities, "Susceptibility per Site", save=save)
        plotting.plot_quantity_range(binder_cumulants, "Binder Cumulant", save=save)
        plotting.plot_correlation_time_range(energy_correlations, "Energy per Site", save=save)
        plotting.plot_correlation_time_range(magnetization_correlations, "Absolute Magnetization", save=save)

    return critical_temperature, critical_temperature_error, magnetizabilities, magnetizations, heat_capacities


def find_critical_exponents(critical_temperature, critical_temperature_error, magnetizabilities, magnetizations, heat_capacities, alpha, beta, gamma, nu, save=False, heat_capacity_correction=0):
    """Perform a data collapse for the different quantities."""
    # Sanity check.
    # if not 0 <= alpha < 1:
    #     raise ValueError("Alpha should be in the interval [0, 1), alpha is {0}".format(alpha))
    if critical_temperature and critical_temperature_error:
        data_collapse(magnetizabilities, "Susceptibility", critical_temperature, -gamma, nu, "Gamma", save=save)
        data_collapse(magnetizations, "Magnetization", critical_temperature, beta, nu, "Beta", save=save)
        data_collapse(heat_capacities, "Heat Capacity", critical_temperature, -alpha, nu, "Alpha", save=save, heat_capacity_correction=heat_capacity_correction)

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


def binning(data, quantity, show_plot=False, save_plot=False):
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
        # plt.title(r'${0}$'.format('\mathrm{' + quantity.replace(' ', '\ ') + '\ Error}'))
        plt.xlabel(r'$\mathrm{Number\ of\ Data\ Points}$')
        plt.ylabel(r'$\mathrm{Error}$')
        plt.xlim(original_length, 1)
        plt.ylim(ymin=0, ymax=max(errors, key=lambda x: x[1])[1] * 1.15)
        plt.semilogx([e[0] for e in errors], [e[1] for e in errors], basex=2)
        sns.despine()
        if save_plot:
            plt.savefig("{0}/{1}_{2}_binning_error.pdf".format(SAVE_LOCATION, time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())), quantity.replace(" ", "_"), bbox_inches='tight'))
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
    keys = sorted(data.keys())
    for key in keys[:-1]:
        data1 = data[key]
        data2 = data[keys[keys.index(key) + 1]]
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
        critical_temperature_error = calculate_error([p[0][0] for p in intersections])
    else:
        critical_temperature = None
        critical_temperature_error = None
    print([i[0] for i in intersections])
    print("Critical temperature is {0} +/- {1}". format(critical_temperature, critical_temperature_error))

    return critical_temperature, critical_temperature_error


def chi_squared_data_collapse(data_set, critical_temperature,
                              critical_temperature_error, ratio, ratio_error, second_exponent_name, show_plots=False, save_plot=False, heat_capacity_correction=0, collapse_limit=1):
    """Find critical exponents through an iterative data fit."""
    best_nus = []
    best_second_exponents = []
    # Easier to overwrite the same file a number of times than to figure out when the last figure was made.
    current_time = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    for k in [-1, 0, 1]:  # For subtraction and addition of error on ratio.
        for p in [-1, 0, 1]:  # Same for error on critical temperature.
            lowest_residual = np.inf
            best_nu = 0
            best_second_exponent = 0
            for nu in np.linspace(0.01, 1.1, 110):
                second_exponent = (ratio + k * ratio_error) * nu
                scaling_functions = {}
                for lattice_size, data in sorted(data_set.items()):
                    # if lattice_size == 20:
                    #     continue
                    for v in data:
                        TC = critical_temperature + p * critical_temperature_error
                        t = (v[0] - TC) / TC
                        scaling_variable = lattice_size**(1 / nu) * t
                        if abs(second_exponent) < 0.0001:
                            v_tilde = np.log(lattice_size) * (v[1] + heat_capacity_correction)
                        else:
                            v_tilde = lattice_size**(-second_exponent / nu) * (v[1] + heat_capacity_correction)
                        v_tilde_error = lattice_size**(-second_exponent / nu) * v[2]
                        scaling_functions.setdefault(lattice_size, []).append((scaling_variable, v_tilde, v_tilde_error))

                # Only consider the temperature range where the most data points lie.
                at_interval = sorted([a for a in list(itertools.chain(*list(scaling_functions.values()))) if abs(a[0]) <= collapse_limit])
                if len(at_interval) >= 15:  # We need enough data points to be able to do a collapse.
                    at_interval_x, at_interval_y, at_interval_y_error = zip(*at_interval)  # Seperate x and y values.
                    polynomial, residuals, _, _, _ = np.polyfit(at_interval_x, at_interval_y, 5, full=True)
                    degrees_of_freedom = len(at_interval) - 6
                    if (residuals / degrees_of_freedom) < lowest_residual:  # Determine whether this fit is better than previous ones.
                        lowest_residual = residuals / degrees_of_freedom
                        best_second_exponent = second_exponent
                        best_nu = nu
                        f = np.poly1d(polynomial)
                        x_new = np.linspace(min(at_interval_x), max(at_interval_x), 50)
                        y_new = f(x_new)
                        if show_plots and k == 0 and p == 0:
                            plt.xlabel(r'$L^{(1/\nu)}t$')
                            plt.ylabel(r'$\mathrm{Scaling\ Function}$')
                            for lattice_size, values in sorted(scaling_functions.items()):
                                plt.errorbar([k[0] for k in values], [k[1] for k in values], [k[2] for k in values], marker='o', linestyle='None', label=r'${0}$'.format(str(lattice_size) + '\mathrm{\ by\ }' + str(lattice_size) + "\mathrm{\ Lattice}"))
                            plt.xlim(-collapse_limit, collapse_limit)
                            sns.despine()
                            plt.legend(loc='best')
                            plt.plot(x_new, y_new)
                            if save_plot:
                                plt.savefig("{0}/{1}_{2}_data_collapse.pdf".format(SAVE_LOCATION, current_time, '{0}_over_nu'.format(second_exponent_name), bbox_inches='tight'))
                            plt.show()

                            for lattice_size, values in sorted(scaling_functions.items()):
                                plt.errorbar([k[0] for k in values], [k[1] - f(k[0]) for k in values], [k[2] for k in values], marker='o', linestyle='None', label=r'${0}$'.format(str(lattice_size) + '\mathrm{\ by\ }' + str(lattice_size) + "\mathrm{\ Lattice}"))
                            plt.axhline(y=0)
                            plt.xlim(-collapse_limit, collapse_limit)
                            plt.ylim(-0.01, 0.01)
                            plt.xlabel(r'$L^{(1/\nu)}t$')
                            plt.ylabel(r'$\mathrm{Scaling\ Function\ -\ polynomial}$')
                            if save_plot:
                                plt.savefig("{0}/{1}_{2}_data_collapse_residual.pdf".format(SAVE_LOCATION, current_time, '{0}_over_nu'.format(second_exponent_name), bbox_inches='tight'))
                            plt.show()

            best_nus.append(best_nu)
            best_second_exponents.append(best_second_exponent)
    print("best {0} = {1} +/- {2}\nbest nu = {3} +/- {4}\n".format(second_exponent_name, abs(np.mean(best_second_exponents)), calculate_error(best_second_exponents), np.mean(best_nus), calculate_error(best_nus)))

    return np.mean(best_second_exponents), calculate_error(best_second_exponents), np.mean(best_nus), calculate_error(best_nus)


def data_collapse(data_set, quantity, critical_temperature, critical_exponent1, nu, name1, save=False, heat_capacity_correction=0):
    """Perform a data collapse with a given set of critical exponents. Used to see how the exact result collapses."""
    scaling_functions = {}
    for lattice_size, data in sorted(data_set.items()):
        for v in data:
            t = (v[0] - critical_temperature) / critical_temperature
            scaling_variable = lattice_size**(1 / nu) * t
            if critical_exponent1 == 0:
                v_tilde = np.log(lattice_size) * (v[1] + heat_capacity_correction)
            else:
                v_tilde = lattice_size**(critical_exponent1 / nu) * (v[1] + heat_capacity_correction)
            v_tilde_error = lattice_size**(critical_exponent1 / nu) * v[2]
            scaling_functions.setdefault(lattice_size, []).append((scaling_variable, v_tilde, v_tilde_error))
    for lattice_size, data in sorted(scaling_functions.items()):
        plt.xlabel(r'$L^{(1 / \nu)}t$')
        plt.ylabel(r'${0}$'.format('\mathrm{' + quantity.replace(' ', '\ ') + '\ Scaling\ Function}'))
        sns.despine()
        plt.errorbar([k[0] for k in data], [k[1] for k in data], [k[2] for k in data], linestyle='None', marker='o', label=r"${0}$".format(str(lattice_size) + '\mathrm{\ by\ }' + str(lattice_size) + "\mathrm{\ Lattice}"))
    print("{0} = {1}, nu = {2}".format(name1, critical_exponent1, nu))
    plt.legend(loc='best')
    sns.despine()
    if save:
        plt.savefig("{0}/{1}_{2}_data_collapse.pdf".format(SAVE_LOCATION, time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())), quantity.replace(" ", "_"), bbox_inches='tight'))

    plt.show()


def loglog_exponent_finding(data_set, quantity, save=False, heat_capacity_correction=0):
    """Find ratios of critical exponents by finding a linear relation between logarithms of lattice sizes and quantity values."""
    lattice_sizes_log = []
    magnetizations_log = []
    magnetizations_log_error = []

    for lattice_size, data in data_set.items():
        magnetization_log = np.log(data[0][1] + heat_capacity_correction)
        magnetization_log_error = data[0][2] / data[0][1]
        lattice_sizes_log.append(np.log(lattice_size))
        magnetizations_log.append(magnetization_log)
        magnetizations_log_error.append(magnetization_log_error)

    lattice_sizes = np.asarray(lattice_sizes_log)
    magnetizations_log = np.asarray(magnetizations_log)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(lattice_sizes, magnetizations_log)

    fitted_line = slope * lattice_sizes + intercept

    plt.xlabel(r'$\log(L)$')
    plt.ylabel(r'$\log({0})$'.format('\mathrm{' + quantity.replace(" ", "\ ") + '}'))
    plt.errorbar(lattice_sizes, magnetizations_log, magnetizations_log_error, linestyle='None', marker='o')
    plt.plot(lattice_sizes, fitted_line)
    sns.despine()
    if save:
        plt.savefig("{0}/{1}_{2}_loglog_plot.pdf".format(SAVE_LOCATION, time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())), quantity.lower().replace(" ", "_"), bbox_inches='tight'))
    plt.show()
    return slope, std_err


def critical_exponent_consistency(gamma, alpha, beta, nu):
    """Use scaling laws to determine consistency of critical exponents."""
    delta = (2 - alpha) / beta - 1
    eta = 2 - gamma / nu
    print("Eta = {0}, Delta = {1}".format(eta, delta))

    print("2 nu + alpha = {0}, should be 2".format(2 * nu + alpha))
    print("alpha + 2 beta + gamma = {0}, should be 2".format(alpha + 2 * beta + gamma))
