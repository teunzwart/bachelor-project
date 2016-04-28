"""Run Monte Carlo simulation over a range of lattice sizes and temperature."""

import numpy as np

import analysis
import exact_ising_model as exact
import ising_model
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
    chi_range = []
    binder_range = []

    if model == "ising":
        model = ising_model.IsingModel
    else:
        raise Exception("{0} is an invalid model choice.".format(model))

    for t in np.arange(lower, upper, step):
        print("Lattice Size {0}, Temperature {1}".format(lattice_size, round(t, 5)))
        simulation = model(lattice_size, bond_energy, t, initial_temperature, total_no_of_sweeps)
        if algorithm == "metropolis":
            simulation.metropolis()
            correlation_correction_factor = 1
            correlation_error_correction = 0
        elif algorithm == "wolff":
            cluster_sizes = simulation.wolff()
            # We need to correct for the different way in which correlationd time is defined for Wolff.
            # TODO: Correct the error on the correlation times.
            correlation_correction_factor = np.mean(cluster_sizes) / simulation.no_of_sites
            correlation_error_correction = analysis.calculate_error(cluster_sizes) / simulation.no_of_sites
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

        # Magnetizability.
        chi_jack, chi_error_jack, _ = analysis.jackknife(equilibrium_magnetization, 8, analysis.magnetizability, temperature=t, no_of_sites=simulation.no_of_sites)
        chi_boot, chi_error_boot = analysis.bootstrap(equilibrium_magnetization, 8, analysis.magnetizability, temperature=t, no_of_sites=simulation.no_of_sites)
        chi = np.mean([chi_jack, chi_boot])
        chi_error = max(chi_error_jack, chi_error_boot)

        binder_jack, binder_error_jack, _ = analysis.jackknife(equilibrium_magnetization, 8, analysis.binder_cumulant)
        binder_boot, binder_error_boot = analysis.bootstrap(equilibrium_magnetization, 8, analysis.binder_cumulant)
        binder = np.mean([binder_jack, binder_boot])
        binder_error = max(binder_error_jack, binder_error_boot)

        energy_range.append((t, energy, energy_error))
        energy_correlation_range.append((t, energy_correlation * correlation_correction_factor))

        energy_sq_range.append((t, energy_sq, energy_sq_error))

        m_range.append((t, m, m_error))
        m_correlation_range.append((t, m_correlation * correlation_correction_factor))

        mag_range.append((t, mag, mag_error))
        mag_sq_range.append((t, mag_sq, mag_sq_error))

        heat_cap_range.append((t, heat_cap, heat_cap_error))
        chi_range.append((t, chi, chi_error))
        binder_range.append((t, binder, binder_error))

        if show_values:
            print("Energy per Site: {0} +/- {1}".format(energy, energy_error))
            print("Magnetization per Site: {0} +/- {1}".format(mag, mag_error))
            print("m (<|M|>/N): {0} +/- {1}".format(m, m_error))
            print("Energy Squared: {0} +/- {1}".format(energy_sq, energy_sq_error))
            print("Magnetization Squared: {0} +/- {1}".format(mag_sq, mag_sq_error))
            print("Heat Capacity per Site: {0} +/- {1}".format(heat_cap, heat_cap_error))
            print("Magnetizability per Site: {0} +/- {1}".format(chi, chi_error))
            print("Binder Cumulant: {0} +/- {1}".format(binder, binder_error))
            print("\n")

    return ((lattice_size, energy_range), (lattice_size, energy_correlation_range),
            (lattice_size, energy_sq_range), (lattice_size, m_range),
            (lattice_size, m_correlation_range), (lattice_size, mag_range),
            (lattice_size, mag_sq_range), (lattice_size, heat_cap_range),
            (lattice_size, chi_range),
            (lattice_size, binder_range))


def lattice_size_range(lattice_sizes, model, algorithm, bond_energy,
                       initial_temperature, thermalization_sweeps, sweeps, lower,
                       upper, step=0.2, save_plots=False, show_plots=True, show_values=False):
    """Caculate and plot values over a range of lattice sizes."""
    energy_lattice_range = []
    energy_correlation_lattice_range = []

    energy_sq_lattice_range = []

    m_lattice_range = []
    m_correlation_lattice_range = []

    mag_lattice_range = []

    mag_sq_lattice_range = []

    heat_cap_lattice_range = []
    chi_lattice_range = []
    binder_lattice_range = []

    for p in lattice_sizes:
        (energy_range, energy_correlation_range, energy_sq_range, m_range,
         m_correlation_range, mag_range, mag_sq_range, heat_cap_range,
         chi_range,
         binder_range) = temperature_range(model, algorithm, p, bond_energy,
                                           initial_temperature,
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
        chi_lattice_range.append(chi_range)
        binder_lattice_range.append(binder_range)

    critical_t, critical_t_error = analysis.find_binder_intersection(binder_lattice_range)
    if show_values:
        print("Critical Temperature: {0} +/- {1}".format(critical_t, critical_t_error))

    exact_energy = exact.internal_energy(bond_energy, 0.01, upper)
    exact_heat_capacity = exact.heat_capacity(bond_energy, 0.01, upper)
    exact_magnetization = exact.magnetization(bond_energy, 0.01, upper)

    if show_plots:
        plotting.plot_quantity_range(energy_lattice_range, "Energy per Site", exact=exact_energy, save=save_plots)
        plotting.plot_correlation_time_range(energy_correlation_lattice_range, "Energy per Site", save=save_plots)
        plotting.plot_quantity_range(energy_sq_lattice_range, "Energy Squared", save=save_plots)

        plotting.plot_quantity_range(m_lattice_range, "Absolute Magnetization per Site", exact=exact_magnetization, save=save_plots)
        plotting.plot_quantity_range(mag_sq_lattice_range, "Squared Magnetization per Site", save=save_plots)

        plotting.plot_quantity_range(heat_cap_lattice_range, "Heat Capacity per Site", exact=exact_heat_capacity, save=save_plots)

        plotting.plot_quantity_range(chi_lattice_range, "Magnetizability per Site", save=save_plots)

        plotting.plot_quantity_range(binder_lattice_range, "Binder Cumulant", save=save_plots)

    return {"energy": energy_lattice_range,
            "energy correlation": energy_correlation_lattice_range,
            "energy squared": energy_sq_lattice_range,
            "m": m_lattice_range,
            "m correlation": m_correlation_lattice_range,
            "magnetization": mag_lattice_range,
            "magnetization squared": mag_sq_lattice_range,
            "heat capacity": heat_cap_lattice_range,
            "binder": binder_lattice_range,
            "critical temperature": (critical_t, critical_t_error)}
