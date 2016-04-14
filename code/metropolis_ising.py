"""An implementation of the Metropolis algorithm for the Ising model."""

import numpy as np
import scipy.special
import matplotlib.pyplot as plt


class MetropolisIsing:
    """An Implementation of the Metropolis algorithm for the Ising model."""

    def __init__(self, lattice_size_L, bond_energy_J, temperature_T,
                 initial_temperature, sweeps):
        """Initialize variables and the lattice."""
        self.lattice_size_L = lattice_size_L
        self.no_of_sites = lattice_size_L**2
        self.bond_energy_J = bond_energy_J
        self.temperature_T = temperature_T
        self.beta = 1 / self.temperature_T
        self.initial_temperature = initial_temperature
        self.sweeps = sweeps
        self.lattice = self.init_lattice()
        self.energy = self.calculate_lattice_energy()
        self.exponents = self.exponents_init()
        self.energy_history = np.empty(self.sweeps)
        self.magnet_history = np.empty(self.sweeps)
        self.rng_seed = int(self.lattice_size_L * self.temperature_T * 1000)
        np.random.seed(self.rng_seed)


    def init_lattice(self):
        """
        Initialize the lattice for the given initial temperature.

        Broken symmetry in the ferromagnetic case (J>0) is taken into account.
        Anti-ferromagnetic ground state has alternatly positive and negative orientation.
        "hi" corresponds to infinte temperature, "lo" to T=0.
        """
        if self.initial_temperature == "hi":
            lattice = np.random.choice([-1, 1], self.no_of_sites).reshape(self.lattice_size_L, self.lattice_size_L)

        elif self.initial_temperature == "lo":
            if self.bond_energy_J > 0:
                # Broken ground state energy.
                ground_state = np.random.choice([-1, 1])
                lattice = np.full((self.lattice_size_L, self.lattice_size_L), ground_state, dtype="int64")
            elif self.bond_energy_J < 0:
                # Set lattice to alternating pattern.
                row1 = np.hstack([1, -1] * self.lattice_size_L)
                row2 = np.hstack([-1, 1] * self.lattice_size_L)
                lattice = np.vstack([row1, row2] * self.lattice_size_L)
            else:
                raise Exception("Bond energy can not be 0.")

        return lattice

    def calculate_lattice_energy(self):
        """Calculate the energy of the lattice using the Ising model Hamiltonian in zero-field."""
        energy = 0
        for y in range(self.lattice_size_L):
            for x in range(self.lattice_size_L):
                center = self.lattice[y][x]
                # Toroidal boundary conditions, lattice wraps around
                neighbours = [
                    (y, (x - 1) % self.lattice_size_L),
                    (y, (x + 1) % self.lattice_size_L),
                    ((y - 1) % self.lattice_size_L, x),
                    ((y + 1) % self.lattice_size_L, x)]
                for n in neighbours:
                    if self.lattice[n] == center:
                        energy -= self.bond_energy_J
                    else:
                        energy += self.bond_energy_J

        return energy / 2  # Every bond has been counted twice.

    def exponents_init(self):
        """Calculate the exponents once since FPO are expensive."""
        exponents = {}
        for x in range(-4, 5, 2):
            exponents[2 * self.bond_energy_J * x] = np.exp(-self.beta * 2 * self.bond_energy_J * x)
        return exponents

    def metropolis(self):
        """Implentation of the Metropolis alogrithm."""
        for t in range(self.sweeps):
            # if t % (self.sweeps / 10) == 0:
            #     print("Sweep {0}".format(t))
            # Measurement every sweep.
            np.put(self.energy_history, t, self.energy)
            np.put(self.magnet_history, t, np.sum(self.lattice))
            for k in range(self.lattice_size_L ** 2):
                # Pick a random location on the lattice.
                rand_y = np.random.randint(0, self.lattice_size_L)
                rand_x = np.random.randint(0, self.lattice_size_L)

                spin = self.lattice[rand_y, rand_x]  # Get spin at the random location.

            # Determine the energy delta from flipping that spin.
            neighbours = [
                (rand_y, (rand_x - 1) % self.lattice_size_L),
                (rand_y, (rand_x + 1) % self.lattice_size_L),
                ((rand_y - 1) % self.lattice_size_L, rand_x),
                ((rand_y + 1) % self.lattice_size_L, rand_x)]
            spin_sum = 0
            for n in neighbours:
                spin_sum += self.lattice[n]
            energy_delta = 2 * self.bond_energy_J * spin * spin_sum

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

    def exact_magnetization(self, temperature):
        """Calculate the exact magnetization. Boltzmann constant is set to 1."""
        M = (1 - np.sinh((2 / temperature) * self.bond_energy_J)**(-4))**(1 / 8)
        if np.isnan(M):
            return 0
        return M

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

        correlation_time = np.ceil(np.trapz(normalized_acf[:500]))

        return correlation_time, normalized_acf

    def plot_energy(self, data=None):
        """Plot of the energy per spin."""
        plt.title("Energy per Spin")
        plt.xlabel("Monte Carlo Sweeps")
        if data is None:
            plt.plot(self.energy_history / self.no_of_sites)
        else:
            plt.plot(data)
        plt.show()

    def plot_magnetization(self, data=None):
        """Plot the magnetization per spin."""
        plt.title("Magnetization per Spin")
        plt.xlabel("Monte Carlo Sweeps")
        if data is None:
            plt.plot(self.magnet_history / self.no_of_sites)
        else:
            plt.plot(data)
        plt.show()

    def show_lattice(self):
        """Plot the lattice."""
        plt.xticks(range(0, self.lattice_size_L, 1))
        plt.yticks(range(0, self.lattice_size_L, 1))
        plt.imshow(self.lattice, interpolation="nearest", extent=[0, self.lattice_size_L, self.lattice_size_L, 0])
        plt.show()

    def plot_correlation_time_range(self, data, lattice_size, critical_temp=False, show_plot=True):
        """Plot correlation times for a range of temperatures."""
        plt.title("Correlation Time in Monte Carlo Sweeps")
        plt.xlabel("Temperature")
        plt.ylabel("Monte Carlo Sweeps")
        plt.plot([d[0] for d in data], [d[1] for d in data], marker='o', linestyle='None', label=lattice_size)
        if critical_temp:
            plt.axvline(2.269)
        if show_plot:
            plt.legend(loc='upper right')
            plt.show()

    def plot_quantity_range(self, data, errors, quantity, lattice_size, legend_loc=None, critical_temp=False, exact=None, show_plot=True):
        """Plot quantity over temperature range."""
        plt.title(quantity)
        plt.xlabel("Temperature")
        plt.ylabel(quantity)
        plt.plot([d[0] for d in data], [d[1] for d in data], label=lattice_size, linestyle='None', marker='o')
        plt.errorbar([d[0] for d in data], [d[1] for d in data], [e[1] for e in errors], linestyle='None')
        if critical_temp:
            plt.axvline(2.269)
        if exact is not None:
            plt.plot([e[0] for e in exact], [e[1] for e in exact])
        plt.xlim(0, data[len(data) - 1][0] + 0.2)
        ymin, ymax = plt.ylim()
        print(ymin, ymax)
        data_min = min(data, key=lambda x: x[1])[1]
        data_max = max(data, key=lambda x: x[1])[1]

        if data_min < ymin and data_max > ymax:
            plt.ylim(data_min * 1.15, data_max * 1.15)
        if data_min < ymin:
            plt.ylim(data_min * 1.15, 0)
        elif data_max > ymax:
            plt.ylim(0, data_max * 1.15)
        if show_plot:
            plt.legend(loc=legend_loc)
            plt.show()


if __name__ == "__main__":
    metropolis_ising = MetropolisIsing(4, 1, 2, "lo", 1000)
    metropolis_ising.metropolis()
    print(metropolis_ising.lattice)
    print(metropolis_ising.energy)
    print(metropolis_ising.magnet_history)
    print(metropolis_ising.energy_history)
