"""Run Monte Carlo simulation over a range of lattice sizes and temperature."""

import argparse
import pickle
import time

import numpy as np

import analysis
import ising_model
import potts_model

SIMULATION_FOLDER = "./simulation_runs"


def single_temperature_simulation(model, algorithm, lattice_size, bond_energy, temperature, initial_temperature,
                                  thermalization_sweeps, measurement_sweeps, show_plots=False, save_plots=False, show_values=True):
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
    energy, energy_error, energy_correlation, energy_bins = analysis.binning(equi_energy, "Energy per Site", show_plots, save_plots)
    energy_sq, energy_sq_error, _, energy_sq_bins = analysis.binning(equi_energy**2, "Energy Squared per Site", False)
    energy_4th, energy_4th_error, _, energy_4th_bins = analysis.binning(equi_energy**4, "Energy^4 per Site", False)

    m, m_error, m_correlation, m_bins = analysis.binning(np.absolute(equi_magnetization), "<|M|>/N", show_plots)
    # Magnetization is always so noisy that plots don't make sense.
    mag, mag_error, _, mag_bins = analysis.binning(equi_magnetization, "Magnetization per Site", False)
    mag_sq, mag_sq_error, _, mag_sq_bins = analysis.binning(equi_magnetization**2, "Magnetization Squared per site", False)
    mag_4th, mag_4th_error, _, mag_4th_bins = analysis.binning(equi_magnetization**4, "Magnetization^4 per Site", False)

    data = ((lattice_size, bond_energy, initial_temperature, thermalization_sweeps, measurement_sweeps,
             temperature, correlation_correction, correlation_correction_error),
            {"energy": energy, "energy error": energy_error,
             "energy bins": energy_bins, "energy correlation": energy_correlation * correlation_correction,
             "energy sq": energy_sq, "energy sq error": energy_sq_error, "energy sq bins": energy_sq_bins,
             "energy 4th": energy_4th, "energy 4th error": energy_4th_error, "energy 4th bins": energy_4th_bins,
             "m": m, "m error": m_error, "m correlation": m_correlation * correlation_correction, "m bins": m_bins,
             "mag": mag, "mag error": mag_error, "mag bins": mag_bins,
             "mag sq": mag_sq, "mag sq error": mag_sq_error, "mag sq bins": mag_sq_bins,
             "mag fourth": mag_4th, "mag 4th error": mag_4th_error, "mag 4th bins": mag_4th_bins})

    if show_values:
        print("Energy per site: {0} +/- {1}".format(energy, energy_error))
        print("\n")

    return data


def simulation_range(model, algorithm, lattice_sizes, bond_energy, initial_temperature,
                     thermalization_sweeps, measurement_sweeps, lower, upper, step=0.2, show_plots=False, save=True):
    """Run a given model over a range of temperature."""
    for k in lattice_sizes:
        simulations = []
        num_of_samples = round(((upper - lower) / step) + 1)
        for t in np.linspace(lower, upper, num_of_samples):
            data = single_temperature_simulation(model, algorithm, k, bond_energy, t, initial_temperature,
                                                 thermalization_sweeps, measurement_sweeps, show_plots)
            simulations.append(data)

        if save:
            with open("{0}/{1}_{2}_{3}_{4}_{5}_[{6}-{7}]_{8}.pickle".format(SIMULATION_FOLDER, time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time())), model, algorithm, k, measurement_sweeps, lower, upper, step), 'wb+') as f:
                pickle.dump(simulations, f, pickle.HIGHEST_PROTOCOL)

    print("Done.")


def argument_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Simulate the 2- and 3-state Potts model in 2D.")
    parser.add_argument(
        "model",
        help="Either Ising (q=2) or Potts (q=3)",
        choices=["ising", "potts"],
        type=str)
    parser.add_argument(
        "algorithm",
        help="Algorithm to use for simulation",
        choices=["metropolis", "wolff"],
        type=str)
    parser.add_argument(
        "lattice_sizes",
        help="Lattice sizes to simulate",
        nargs='+',
        type=int
    )
    parser.add_argument(
        "bond_energy",
        help="Specify the bond energy",
        type=int
    )
    parser.add_argument(
        "init_temperature",
        choices=["hi", "lo"],
        help=('specify initial temperature of simulation ("hi" is infinite, '
              '"lo" is 0)'))
    parser.add_argument(
        "thermalization_sweeps",
        help="Number of sweeps to perform before measurements start",
        type=int
    )
    parser.add_argument(
        "measurement_sweeps",
        help="Number of sweeps to measure",
        type=int
    )
    parser.add_argument(
        "lower",
        help="Lower temperature bound",
        type=float
    )
    parser.add_argument(
        "upper",
        help="Upper temperature bound",
        type=float
    )
    parser.add_argument(
        "--step",
        help="Temperature step, default is 0.2",
        type=float,
        default=0.2
    )
    parser.add_argument(
        "--show_plots",
        help="Show plots on error calculations, default is false because this is a blocking operation",
        action="store_true"
    )
    parser.add_argument(
        "--nosave",
        help="Do not save output in a binary pickle file, default behaviour is that it is saved",
        action="store_false"
    )
    arguments = parser.parse_args()
    return arguments

if __name__ == '__main__':
    args = argument_parser()
    simulation_range(
        args.model,
        args.algorithm,
        args.lattice_sizes,
        args.bond_energy,
        args.init_temperature,
        args.thermalization_sweeps,
        args.measurement_sweeps,
        args.lower,
        args.upper,
        args.step,
        args.show_plots,
        args.nosave)
