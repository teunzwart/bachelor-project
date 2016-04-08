"""An implementation of the Metropolis algorithm for the Ising model."""

import numpy as np
import scipy.special


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

        return energy

    def exponents_init(self):
        """Calculate the exponents once since FPO are expensive."""
        exponents = {}
        for x in range(-4, 5, 2):
            exponents[2 * self.bond_energy_J * x] = np.exp(-self.beta * 2 * self.bond_energy_J * x)
        return exponents

    def metropolis(self):
        """Implentation of the Metropolis alogrithm."""
        for t in range(self.sweeps):
            if t % (self.sweeps / 10) == 0:
                print("Sweep {0}".format(t))
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

    def exact_energy(self, temperature):
        """Calculate the exact energy with Boltzmann constant set to 1."""
        k = 2 * np.sinh((2 / temperature) * self.bond_energy_J) / (np.cosh((2 / temperature) * self.bond_energy_J))**2
        energy = (- self.bond_energy_J * (1 / np.tanh((2 / temperature) * self.bond_energy_J)) *
                  (1 + (2 / np.pi) * (2 * np.tanh((2 / temperature) * self.bond_energy_J)**2 - 1) *
                   scipy.special.ellipk(k)))

        return energy

if __name__ == "__main__":
    metropolis_ising = MetropolisIsing(4, 1, 2, "lo", 1000)
    metropolis_ising.metropolis()
    print(metropolis_ising.lattice)
    print(metropolis_ising.energy)
    print(metropolis_ising.magnet_history)
    print(metropolis_ising.energy_history)
