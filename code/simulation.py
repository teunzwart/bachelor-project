"""Run Monte Carlo simulation over a range of lattice sizes and temperature."""

import pickle
import time

import numpy as np

import analysis
import ising_model
import potts_model

SIMULATION_FOLDER = "./simulation_runs"


def single_temperature_simulation(model, algorithm, lattice_size, bond_energy, temperature, initial_temperature,
                                  thermalization_sweeps, measurement_sweeps, show_plots=False):
    """Only perform the simulation. No analysis."""
    total_no_of_sweeps = thermalization_sweeps + measurement_sweeps

    if model == "ising":
        model = ising_model.IsingModel
    elif model == "potts":
        model = potts_model.PottsModel
    else:
        raise Exception("{0} is an invalid model choice.".format(model))

    print("Lattice Size {0}, Temperature {1}".format(lattice_size, temperature))
    simulation = model(lattice_size, bond_energy, temperature, initial_temperature, total_no_of_sweeps)
    if algorithm == "metropolis":
        simulation.metropolis()
        correlation_correction = 1
        correlation_correction_error = 0
    elif algorithm == "wolff":
        cluster_sizes = simulation.wolff()
        # We need to correct for the different way in which correlation time is defined for Wolff.
        correlation_correction = np.mean(cluster_sizes) / simulation.no_of_sites
        correlation_correction_error = analysis.calculate_error(cluster_sizes) / simulation.no_of_sites
    else:
        raise Exception("Invalid algorithm.")

    # Normalized energy and magnetization in equilbrium.
    equi_energy = simulation.energy_history[thermalization_sweeps:] / simulation.no_of_sites
    equi_magnetization = simulation.magnetization_history[thermalization_sweeps:] / simulation.no_of_sites

    # E and E**2 have very similair binning error profiles. So don't plot both.
    energy, energy_error, energy_correlation, energy_bins = analysis.binning(equi_energy, "Energy per Site", show_plots)
    energy_sq, energy_sq_error, _, energy_sq_bins = analysis.binning(equi_energy**2, "Energy Squared per Site", False)
    energy_4th, energy_4th_error, _, energy_4th_bins = analysis.binning(equi_energy**4, "Energy^4 per Site", False)

    m, m_error, m_correlation, m_bins = analysis.binning(np.absolute(equi_magnetization), "<|M|>/N", show_plots)
    # Magnetization is always so noisy that plots don't make sense.
    mag, mag_error, _, mag_bins = analysis.binning(equi_magnetization, "Magnetization per Site", False)
    mag_sq, mag_sq_error, _, mag_sq_bins = analysis.binning(equi_magnetization**2, "Magnetization Squared per site", False)
    mag_4th, mag_4th_error, _, mag_4th_bins = analysis.binning(equi_magnetization**4, "Magnetization^4 per Site", False)

    data = ((lattice_size, bond_energy, initial_temperature, total_no_of_sweeps,
             temperature, correlation_correction, correlation_correction_error),
            {"energy": energy, "energy error": energy_error,
             "energy bins": energy_bins, "energy correlation": energy_correlation * correlation_correction,
             "energy sq": energy_sq, "energy sq error": energy_sq_error, "energy sq bins": energy_sq_bins,
             "energy 4th": energy_4th, "energy 4th error": energy_4th_error, "energy 4th bins": energy_4th_bins,
             "m": m, "m error": m_error, "m correlation": m_correlation, "m bins": m_bins,
             "mag": mag, "mag error": mag_error, "mag bins": mag_bins,
             "mag sq": mag_sq, "mag sq error": mag_sq_error, "mag sq bins": mag_sq_bins,
             "mag fourth": mag_4th, "mag 4th error": mag_4th_error, "mag 4th bins": mag_4th_bins})

    return data


def simulation_range(model, algorithm, lattice_sizes, bond_energy, initial_temperature,
                     thermalization_sweeps, measurement_sweeps, lower, upper, step=0.2, show_plots=False, save=True):
    """Run a given model over a range of temperature."""
    for k in lattice_sizes:
        simulations = []
        num_of_samples = round(((upper - lower) / step) + 1)
        for t in np.linspace(lower, upper, num_of_samples):
            data = single_temperature_simulation(model, algorithm, k, bond_energy, t, initial_temperature,
                                                 thermalization_sweeps, measurement_sweeps, show_plots=False)
            simulations.append(data)

        with open("{0}/{1}_{2}_{3}_{4}_{5}_[{6}-{7}]_{8}.pickle".format(SIMULATION_FOLDER, time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time())), model, algorithm, k, thermalization_sweeps + measurement_sweeps, lower, upper, step), 'wb+') as f:
            pickle.dump(simulations, f, pickle.HIGHEST_PROTOCOL)
