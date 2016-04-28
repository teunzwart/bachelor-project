"""Run Monte Carlo simulation over a range of lattice sizes and temperature."""

import numpy as np

import analysis
import exact_ising_model as exact
import plotting


def temperature_range(model, algorithm, lattice_size, bond_energy, initial_temperature,
                      thermalization_sweeps, sweeps, lower, upper, step=0.2, show_values=False):
    """Run a given model over a range of temperature."""
    total_no_of_sweeps = thermalization_sweeps + sweeps
    energy_range = []
    energy_correlation_range = []

    energy_sq_range = []

    m_range = []
    m_correlation_range = []

    mag_range = []
    mag_sq_range = []

    heat_cap_range = []

    for t in np.arange(lower, upper, step):
        print("Lattice Size {0}, Temperature {1}".format(lattice_size, round(t, 3)))
        simulation = model(lattice_size, bond_energy, t, initial_temperature, total_no_of_sweeps)
        if algorithm == "metropolis":
            simulation.metropolis()
        elif algorithm == "wolff":
            cluster_sizes = simulation.wolff()
            mean_cluster = np.mean(cluster_sizes)
            mean_cluster_error = analysis.calculate_error(cluster_sizes)
            #TODO: Fix Wolff correlation times.
        else:
            raise Exception("Invalid algorithm.")
        equilibrium_energy = simulation.energy_history[thermalization_sweeps:] / simulation.no_of_sites
        equilibrium_magnetization = simulation.magnetization_history[thermalization_sweeps:] / simulation.no_of_sites

        energy, energy_error, energy_correlation, _ = analysis.binning(equilibrium_energy, 10, "Energy per Site")
        energy_sq, energy_sq_error, energy_sq_correlation, _ = analysis.binning((simulation.no_of_sites * equilibrium_energy)**2, 10, "Energy Squared")

        m, m_error, m_correlation, _ = analysis.binning(np.absolute(equilibrium_magnetization), 10, "<|M|>/N")
        mag, mag_error, mag_correlation, _ = analysis.binning(equilibrium_magnetization, 10, "Magnetization per Site")
        mag_sq, mag_sq_error, mag_sq_correlation, _ = analysis.binning((equilibrium_magnetization * simulation.no_of_sites)**2, 10, "Magnetization Squared")

        heat_cap_jack, heat_cap_error_jack, _ = analysis.jackknife(equilibrium_energy, 8, analysis.heat_capacity, temperature=t, no_of_sites=simulation.no_of_sites)
        heat_cap_boot, heat_cap_error_boot = analysis.bootstrap(equilibrium_energy, 8, analysis.heat_capacity, temperature=t, no_of_sites=simulation.no_of_sites)
        heat_cap = np.mean([heat_cap_jack, heat_cap_boot])
        heat_cap_error = max(heat_cap_error_jack, heat_cap_error_jack)

        energy_range.append((t, energy, energy_error))
        energy_correlation_range.append((t, energy_correlation))

        energy_sq_range.append((t, energy_sq, energy_sq_error))

        m_range.append((t, m, m_error))
        m_correlation_range.append((t, m_correlation))

        mag_range.append((t, mag, mag_error))
        mag_sq_range.append((t, mag_sq, mag_sq_error))

        heat_cap_range.append((t, heat_cap, heat_cap_error))

        if show_values:
            print("Energy per Site: {0} +/- {1}".format(energy, energy_error))
            print("Magnetization per Site: {0} +/- {1}".format(mag, mag_error))
            print("m (<|M|>/N): {0} +/- {1}".format(m, m_error))
            print("Energy Squared: {0} +/- {1}".format(energy_sq, energy_sq_error))
            print("Magnetization Squared: {0} +/- {1}".format(mag_sq, mag_sq_error))
            print("Heat Capacity per Site: {0} +/- {1}".format(heat_cap, heat_cap_error))
            print("\n")

    return (energy_range, energy_correlation_range,
            energy_sq_range,
            m_range, m_correlation_range,
            mag_range,
            mag_sq_range,
            heat_cap_range)


def lattice_size_range(lattice_sizes, model, algorithm, bond_energy,
                       initial_temperature, thermalization_sweeps, sweeps, lower,
                       upper, step=0.2, save_plots=False, show_values=False):
    energy_lattice_range = []
    energy_correlation_lattice_range = []

    energy_sq_lattice_range = []

    m_lattice_range = []
    m_correlation_lattice_range = []

    mag_lattice_range = []

    mag_sq_lattice_range = []

    heat_cap_lattice_range = []

    for p in range(len(lattice_sizes)):
        (energy_range, energy_correlation_range, energy_sq_range,
         m_range, m_correlation_range,
         mag_range, mag_sq_range,
         heat_cap_range) = temperature_range(model, algorithm, lattice_sizes[p],
                                             bond_energy, initial_temperature,
                                             thermalization_sweeps, sweeps,
                                             lower, upper, step, show_values)

        energy_lattice_range.append(energy_range)
        energy_correlation_lattice_range.append(energy_correlation_range)

        energy_sq_lattice_range.append(energy_sq_range)

        m_lattice_range.append(m_range)
        m_correlation_lattice_range.append(m_correlation_range)

        mag_lattice_range.append(mag_range)
        mag_sq_lattice_range.append(mag_sq_range)

        heat_cap_lattice_range.append(heat_cap_range)

    exact_energy = exact.internal_energy(bond_energy, 0.01, upper)
    exact_heat_capacity = exact.heat_capacity(bond_energy, 0.01, upper)
    exact_magnetization = exact.magnetization(bond_energy, 0.01, upper)

    plotting.plot_quantity_range(energy_lattice_range, lattice_sizes, "Energy per Site", exact=exact_energy, save=save_plots)
    plotting.plot_correlation_time_range(energy_correlation_lattice_range, lattice_sizes, "Energy per Site", save=save_plots)
    plotting.plot_quantity_range(energy_sq_lattice_range, lattice_sizes, "Energy Squared", save=save_plots)

    plotting.plot_quantity_range(m_lattice_range, lattice_sizes, "Absolute Magnetization per Site", exact=exact_magnetization, save=save_plots)
    plotting.plot_quantity_range(mag_sq_lattice_range, lattice_sizes, "Squared Magnetization per Site", save=save_plots)

    plotting.plot_quantity_range(heat_cap_lattice_range, lattice_sizes, "Heat Capacity per Site", exact=exact_heat_capacity, save=save_plots)

    return (energy_lattice_range, energy_correlation_lattice_range,
            energy_sq_lattice_range,
            m_lattice_range, m_correlation_lattice_range,
            mag_lattice_range, mag_sq_lattice_range,
            heat_cap_lattice_range)
