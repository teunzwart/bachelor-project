"""An implementation of the Metropolis algorithm for the Ising model."""

import time

import numpy as np
import matplotlib.pyplot as plt


class MetropolisIsing:
    """An Implementation of the Metropolis algorithm for the Ising model."""

    def __init__(self, lattice_size, bond_energy, temperature,
                 initial_temperature, sweeps):
        """Initialize variables and the lattice."""
        print("Temperature is {0}".format(round(temperature, 2)))
        self.lattice_size = lattice_size
        self.no_of_sites = lattice_size**2
        self.bond_energy = bond_energy
        self.temperature = temperature
        self.beta = 1 / self.temperature
        self.initial_temperature = initial_temperature
        self.sweeps = sweeps
        self.lattice = self.init_lattice()
        self.energy = self.calculate_lattice_energy()
        self.exponents = self.exponents_init()
        self.energy_history = np.empty(self.sweeps)
        self.magnet_history = np.empty(self.sweeps)
        self.rng_seed = int(self.lattice_size * self.temperature * 1000)
        np.random.seed(self.rng_seed)

    def init_lattice(self):
        """
        Initialize the lattice for the given initial temperature.

        Broken symmetry in the ferromagnetic case (J>0) is taken into account.
        Anti-ferromagnetic ground state has alternatly positive and negative orientation.
        "hi" corresponds to infinte temperature, "lo" to T=0.
        """
        if self.initial_temperature == "hi":
            lattice = np.random.choice([-1, 1], self.no_of_sites).reshape(self.lattice_size, self.lattice_size)

        elif self.initial_temperature == "lo":
            if self.bond_energy > 0:
                # Broken ground state energy.
                ground_state = np.random.choice([-1, 1])
                lattice = np.full((self.lattice_size, self.lattice_size), ground_state, dtype="int64")
            elif self.bond_energy < 0:
                # Set lattice to alternating pattern.
                row1 = np.hstack([1, -1] * self.lattice_size)
                row2 = np.hstack([-1, 1] * self.lattice_size)
                lattice = np.vstack([row1, row2] * self.lattice_size)
            else:
                raise Exception("Bond energy can not be 0.")

        return lattice

    def calculate_lattice_energy(self):
        """Calculate the energy of the lattice using the Ising model Hamiltonian in zero-field."""
        energy = 0
        for y in range(self.lattice_size):
            for x in range(self.lattice_size):
                center = self.lattice[y][x]
                # Toroidal boundary conditions, lattice wraps around
                neighbours = [
                    (y, (x - 1) % self.lattice_size),
                    (y, (x + 1) % self.lattice_size),
                    ((y - 1) % self.lattice_size, x),
                    ((y + 1) % self.lattice_size, x)]
                for n in neighbours:
                    if self.lattice[n] == center:
                        energy -= self.bond_energy
                    else:
                        energy += self.bond_energy

        return energy / 2  # Every bond has been counted twice.

    def exponents_init(self):
        """Calculate the exponents once since FPO are expensive."""
        exponents = {}
        for x in range(-4, 5, 2):
            exponents[2 * self.bond_energy * x] = np.exp(-self.beta * 2 * self.bond_energy * x)
        return exponents

    def metropolis(self, optimize=True):
        """
        Implentation of the Metropolis alogrithm.

        If optimize is False, the naive way to calculate the energy delta
        using the Hamiltonian is used.
        """
        for t in range(self.sweeps):
            # if t % (self.sweeps / 10) == 0:
            #     print("Sweep {0}".format(t))
            # Measurement every sweep.
            np.put(self.energy_history, t, self.energy)
            np.put(self.magnet_history, t, np.sum(self.lattice))
            for k in range(self.lattice_size ** 2):
                # Pick a random location on the lattice.
                rand_y = np.random.randint(0, self.lattice_size)
                rand_x = np.random.randint(0, self.lattice_size)

                spin = self.lattice[rand_y, rand_x]  # Get spin at the random location.

                # Determine the energy delta from flipping that spin.
                neighbours = [
                    (rand_y, (rand_x - 1) % self.lattice_size),
                    (rand_y, (rand_x + 1) % self.lattice_size),
                    ((rand_y - 1) % self.lattice_size, rand_x),
                    ((rand_y + 1) % self.lattice_size, rand_x)]
                spin_sum = 0
                for n in neighbours:
                    spin_sum += self.lattice[n]
                energy_delta = 2 * self.bond_energy * spin * spin_sum

                # Energy may always be lowered.
                if energy_delta <= 0:
                    acceptance_probability = 1
                # Energy is increased with probability proportional to Boltzmann distribution.
                else:
                    acceptance_probability = self.exponents[energy_delta]
                if np.random.random() <= acceptance_probability:
                    # Flip the spin and change the energy.
                    self.lattice[rand_y, rand_x] = -1 * spin
                    self.energy += energy_delta

    def exact_magnetization(self, lower_temperature, higher_temperature, step=0.1):
        """Calculate the exact magnetization. Boltzmann constant is set to 1."""
        exact_magnetization = []

        for t in np.arange(lower_temperature, higher_temperature, step):
            M = (1 - np.sinh((2 / t) * self.bond_energy)**(-4))**(1 / 8)
            if np.isnan(M):
                M = 0
            exact_magnetization.append((t, M))

        return exact_magnetization

    def autocorrelation(self, data):
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

        correlation_time = np.ceil(np.trapz(normalized_acf[:500]))

        return correlation_time, normalized_acf

    def numpy_autocorrelation(self, data, plotting=False):
        """Quick autocorrelation calculation."""
        data = np.asarray([d - np.mean(data) for d in data])
        acf = np.correlate(data, data, mode="full")[(len(data) - 1):]  # Only keep the usefull data (correlation is symmetric around index len(data)).
        normalized_acf = acf / acf.max()
        if plotting:
            plt.title("Autocorrelation Function")
            plt.xlabel("Monte Carlo Sweeps")
            plt.plot(range(len(normalized_acf)), normalized_acf)
            plt.show()

            plt.title("Autocorrelation Function")
            plt.xlabel("Monte Carlo Sweeps")
            plt.plot(range(2500), normalized_acf[:2500])
            plt.show()

        correlation_time = np.trapz(normalized_acf[:500])

        return correlation_time, normalized_acf

    def plot_energy(self, data=None):
        """Plot of the energy per spin."""
        plt.title("Energy per Spin")
        plt.xlabel("Monte Carlo Sweeps")
        plt.ylabel("Energy per Spin")
        if data is None:
            plt.plot(self.energy_history / self.no_of_sites)
        else:
            plt.plot(data)
        plt.show()

    def plot_magnetization(self, data=None):
        """Plot the magnetization per spin."""
        plt.title("Magnetization per Spin")
        plt.xlabel("Monte Carlo Sweeps")
        plt.ylabel("Magnetization per Spin")
        if data is None:
            plt.plot(self.magnet_history / self.no_of_sites)
        else:
            plt.plot(data)
        plt.show()

    def show_lattice(self, show_ticks=True):
        """Plot the lattice."""
        plt.xticks(range(0, self.lattice_size, 1))
        plt.yticks(range(0, self.lattice_size, 1))
        if not show_ticks:
            for tic in plt.gca().xaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False
            for tic in plt.gca().yaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False
            plt.gca().grid(False)
        plt.imshow(self.lattice, interpolation="nearest", extent=[0, self.lattice_size, self.lattice_size, 0])
        plt.show()

    def plot_correlation_time_range(self, data, lattice_size, quantity, critical_temp=False, show_plot=True):
        """Plot autocorrelation times for a range of temperatures."""
        plt.title("{0} Autocorrelation Time in Monte Carlo Sweeps".format(quantity))
        plt.xlabel("Temperature")
        plt.ylabel("Monte Carlo Sweeps")
        plt.plot([d[0] for d in data], [d[1] for d in data], marker='o', linestyle='None', label="{0} by {0} Lattice".format(lattice_size))
        if critical_temp:
            plt.axvline(2.269)
        if show_plot:
            plt.legend(loc='upper right')
            plt.show()

    def plot_quantity_range(self, data, errors, quantity, lattice_size, legend_loc="upper right", critical_temp=False, exact=None, show_plot=True, save=False):
        """Plot quantity over temperature range."""
        plt.title(quantity)
        plt.xlabel("Temperature")
        plt.ylabel(quantity)
        plt.plot([d[0] for d in data], [d[1] for d in data], label="{0} by {0} Lattice".format(lattice_size), linestyle='None', marker='o')
        plt.errorbar([d[0] for d in data], [d[1] for d in data], [e[1] for e in errors], linestyle='None')
        if critical_temp:
            plt.axvline(2.269)
        if exact is not None:
            plt.plot([e[0] for e in exact], [e[1] for e in exact], label="Exact Solution")

        plt.xlim(0, data[-1][0] + 0.2)
        ymin, ymax = plt.ylim()
        data_min = min(data, key=lambda x: x[1])[1]
        data_max = max(data, key=lambda x: x[1])[1]
        if data_max <= 0:
            if data_min <= ymin:
                plt.ylim(data_min * 1.15, 0)
        else:
            if data_max >= ymax:
                plt.ylim(0, data_max * 1.15)
        plt.legend(loc=legend_loc)
        if save:
            plt.savefig("../bachelor-thesis/images/{0}-{1}.pdf".format(quantity.replace(" ", "_"), int(np.rint(time.time()))), bbox_inches='tight')
        if show_plot:
            plt.show()

    def calculate_error(self, data):
        """Calculate the error on a data set."""
        return np.std(data) / np.sqrt(len(data))

    def heat_capacity(self, energy_data, temperature):
        """
        Calculate the heat capacity for a given energy data set and temperature.

        Multiply by the number of sites, because the data has been normalised to the number of sites.
        """
        return self.no_of_sites / temperature**2 * (np.mean(energy_data**2) - np.mean(energy_data)**2)

    def binning_method(self, data, L, quantity, plotting=False):
        """
        Calculate autocorrelation time, mean and error for a quantity using the binning method.

        L is the number of halfings of the data.
        """
        original_length = len(data)
        errors = []
        errors.append((original_length, self.calculate_error(data)))
        for n in range(L):
            if len(data) < 32:
                break
            data = np.asarray([(a + b) / 2 for a, b in zip(data[::2], data[1::2])])
            errors.append((len(data), self.calculate_error(data)))

        if plotting:
            plt.title("Binning Method {0} Error, Log Scale".format(quantity))
            plt.xlabel("Data Points")
            plt.ylabel("Error")
            plt.xlim(original_length, 1)
            plt.ylim(ymin=0, ymax=max(errors, key=lambda x: x[1])[1] * 1.15)
            plt.semilogx([e[0] for e in errors], [e[1] for e in errors[::1]], basex=2)
            plt.show()

            plt.title("Binning Method {0} Error".format(quantity))
            plt.xlabel("Data Points")
            plt.ylabel("Error")
            plt.xlim(original_length, 1)
            plt.ylim(ymin=0, ymax=max(errors, key=lambda x: x[1])[1] * 1.15)
            plt.plot([e[0] for e in errors], [e[1] for e in errors[::1]])
            plt.show()

        autocorrelation_time = 0.5 * ((max(errors, key=lambda x: x[1])[1] / errors[0][1])**2 - 1)
        if np.isnan(autocorrelation_time):
            autocorrelation_time = 1

        return autocorrelation_time, np.mean(data), max(errors, key=lambda x: x[1])[1]

    def bootstrap_method(self, data, no_of_resamples, temperature, operation):
        """Calculate error using the bootstrap method."""
        resamples = np.empty(no_of_resamples)
        for k in range(no_of_resamples):
            random_picks = np.random.choice(data, len(data))
            resamples.put(k, operation(random_picks, temperature))

        error = np.sqrt((np.mean(resamples**2) - np.mean(resamples)**2)) / no_of_resamples
        return error

    def sample_every_two_correlation_times(self, energy_data, magnetization_data, correlation_time):
        """Sample the given data every 2 correlation times and determine value and error."""
        magnet_samples = []
        energy_samples = []

        for t in np.arange(0, len(energy_data), 2 * int(correlation_time)):
            magnet_samples.append(magnetization_data[t])
            energy_samples.append(energy_data[t])

        magnet_samples = np.asarray(magnet_samples)
        energy_samples = np.asarray(energy_samples)

        abs_magnetization = np.mean(np.absolute(magnet_samples))
        abs_magnetization_error = self.calculate_error(magnet_samples)
        print("<m> (<|M|/N>) = {0} +/- {1}".format(abs_magnetization, abs_magnetization_error))

        magnetization = np.mean(magnet_samples)
        magnetization_error = self.calculate_error(magnet_samples)
        print("<M/N> = {0} +/- {1}".format(magnetization, magnetization_error))

        energy = np.mean(energy_samples)
        energy_error = self.calculate_error(energy_samples)
        print("<E/N> = {0} +/- {1}".format(energy, energy_error))

        magnetization_squared = np.mean((magnet_samples * self.no_of_sites)**2)
        magnetization_squared_error = self.calculate_error((magnet_samples * self.no_of_sites)**2)
        print("<M^2> = {0} +/- {1}".format(magnetization_squared, magnetization_squared_error))

if __name__ == "__main__":
    metropolis_ising = MetropolisIsing(4, 1, 2, "lo", 1000)
    metropolis_ising.metropolis()
    print(metropolis_ising.lattice)
    print(metropolis_ising.energy)
    print(metropolis_ising.magnet_history)
    print(metropolis_ising.energy_history)
