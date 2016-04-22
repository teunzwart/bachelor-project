"""An implementation of the Metropolis algorithm for the Ising model."""

import numpy as np
import matplotlib.pyplot as plt


class MetropolisIsing:
    """An Implementation of the Metropolis algorithm for the Ising model."""

    def __init__(self, lattice_size, bond_energy, temperature,
                 initial_temperature, sweeps):
        """Initialize variables and the lattice."""
        print("\nTemperature is {0}".format(round(temperature, 2)))
        self.rng_seed = int(lattice_size * temperature * 1000)
        print("RNG Seed is {0}".format(self.rng_seed))
        np.random.seed(self.rng_seed)
        self.lattice_size = lattice_size
        self.no_of_sites = lattice_size**2
        self.bond_energy = bond_energy
        self.temperature = temperature
        self.beta = 1 / self.temperature
        self.initial_temperature = initial_temperature
        self.sweeps = sweeps
        self.lattice = self.init_lattice()
        self.energy = self.calculate_lattice_energy()
        self.energy_history = np.empty(self.sweeps)
        self.magnetization_history = np.empty(self.sweeps)

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

    def metropolis(self, show_progress=False):
        """Implentation of the Metropolis alogrithm."""
        # Precalculate the exponenents because floating point operations are expensive.
        exponents = {2 * self.bond_energy * x: np.exp(-self.beta * 2 * self.bond_energy * x) for x in range(-4, 5, 2)}
        for t in range(self.sweeps):
            if t % 100 == 0 and show_progress:
                print("Sweep {0}".format(t))
            # Measurement every sweep.
            np.put(self.energy_history, t, self.energy)
            np.put(self.magnetization_history, t, np.sum(self.lattice))
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
                    acceptance_probability = exponents[energy_delta]
                if np.random.random() <= acceptance_probability:
                    # Flip the spin and change the energy.
                    self.lattice[rand_y, rand_x] = -1 * spin
                    self.energy += energy_delta

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

        correlation_time = np.ceil(np.trapz(normalized_acf[:4000]))

        return correlation_time, normalized_acf

    def numpy_autocorrelation(self, data, show_plot=False):
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

    def calculate_error(self, data):
        """Calculate the error on a data set."""
        return np.std(data) / np.sqrt(len(data))



    def sample_every_two_correlation_times(self, energy_data, magnetization_data, correlation_time):
        """Sample the given data every 2 correlation times and determine value and error."""
        magnet_samples = []
        energy_samples = []

        for t in np.arange(0, len(energy_data), 2 * int(np.ceil(correlation_time))):
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

    def temperature_range():
        """Run simulation over a range of temperatures."""
        pass

if __name__ == "__main__":
    metropolis_ising = MetropolisIsing(4, 1, 2, "lo", 1000)
    metropolis_ising.metropolis()
    print(metropolis_ising.lattice)
    print(metropolis_ising.energy)
    print(metropolis_ising.magnet_history)
    print(metropolis_ising.energy_history)
