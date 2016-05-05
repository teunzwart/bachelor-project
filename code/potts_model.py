"""A Monte Carlo simulation of the 3-state Potts model in 2D."""

import copy
import numpy as np


class PottsModel:
    """A Monte Carlo simulation of the Ising model."""

    def __init__(self, lattice_size, bond_energy, temperature,
                 initial_temperature, sweeps):
        """Initialize variables and the lattice."""
        self.rng_seed = int(lattice_size * temperature * 1000)
        np.random.seed(self.rng_seed)
        self.lattice_size = lattice_size
        self.no_of_sites = lattice_size**2
        self.states = [-1, 0, 1]
        self.bond_energy = bond_energy
        self.temperature = temperature
        self.beta = 1 / self.temperature
        self.initial_temperature = initial_temperature
        self.sweeps = sweeps
        self.lattice = self.init_lattice()
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
            lattice = np.random.choice(self.states, self.no_of_sites).reshape(self.lattice_size, self.lattice_size)

        elif self.initial_temperature == "lo":
            raise NotImplentedError("Low temperature starting lattices are not implemented.")
        else:
            raise Exception("{0} is not a valid bond energy.".format(self.initial_temperature))

        return lattice

    def calculate_lattice_energy(self, lattice):
        """Calculate the energy of the lattice using the Ising model Hamiltonian in zero-field."""
        energy = 0
        for y in range(self.lattice_size):
            offset_y = (y + 1) % self.lattice_size
            current_row = lattice[y]
            next_row = lattice[offset_y]
            for x in range(self.lattice_size):
                center = current_row[x]
                offset_x = (x + 1) % self.lattice_size
                if current_row[offset_x] == center:
                    energy -= self.bond_energy
                if next_row[x] == center:
                    energy -= self.bond_energy
        return energy

    def metropolis(self):
        """Implentation of the Metropolis alogrithm."""
        # Precalculate the exponenents because floating point operations are expensive.
        exponents = {2 * self.bond_energy * x: np.exp(-self.beta * 2 * self.bond_energy * x) for x in range(-4, 5, 2)}
        energy = self.calculate_lattice_energy(self.lattice)
        magnetization = np.sum(self.lattice)
        for t in range(self.sweeps):
            # Measurement every sweep.
            np.put(self.energy_history, t, energy)
            np.put(self.magnetization_history, t, magnetization)
            for k in range(self.lattice_size**2):
                # Pick a random location on the lattice.
                rand_y = np.random.randint(0, self.lattice_size)
                rand_x = np.random.randint(0, self.lattice_size)

                spin = self.lattice[rand_y, rand_x]  # Get spin at the random location.
                states = copy.deepcopy(self.states)
                temp_lattice = self.lattice
                random_new_spin = np.random.choice(self.states.remove(spin))
                temp_lattice[rand_y, rand_x] = random_new_spin
                new_energy = self.calculate_lattice_energy(temp_lattice)
                energy_delta = new_energy - energy

                # Energy may always be lowered.
                if energy_delta <= 0:
                    acceptance_probability = 1
                # Energy is increased with probability proportional to Boltzmann distribution.
                else:
                    acceptance_probability = exponents[energy_delta]
                if np.random.random() <= acceptance_probability:
                    # Flip the spin and change the energy.
                    self.lattice[rand_y, rand_x] = random_new_spin
                    energy += energy_delta
                    magnetization = np.sum(self.lattice)

    def wolff(self):
        """Simulate the lattice using the Wolff algorithm."""
        padd = 1 - np.exp(-2 * self.beta * self.bond_energy)
        cluster_sizes = []
        energy = self.calculate_lattice_energy()
        for t in range(self.sweeps):
            # Measurement every sweep.
            np.put(self.energy_history, t, energy)
            np.put(self.magnetization_history, t, np.sum(self.lattice))

            stack = []  # Locations for which the neighbours still have to be checked.

            # Pick a random location on the lattice as the seed.
            seed_y = np.random.randint(0, self.lattice_size)
            seed_x = np.random.randint(0, self.lattice_size)

            seed_spin = self.lattice[seed_y, seed_x]  # Get spin at the seed location.
            stack.append((seed_y, seed_x))
            self.lattice[seed_y, seed_x] *= -1  # Flip the spin.
            cluster_size = 1

            while stack:
                current_y, current_x = stack.pop()
                neighbours = [
                    (current_y, (current_x - 1) % self.lattice_size),
                    (current_y, (current_x + 1) % self.lattice_size),
                    ((current_y - 1) % self.lattice_size, current_x),
                    ((current_y + 1) % self.lattice_size, current_x)]

                for n in neighbours:
                    if self.lattice[n] == seed_spin:
                        if np.random.random() < padd:
                            stack.append(n)
                            self.lattice[n] *= -1
                            cluster_size += 1

            # Pure Python is very slow here.
            # energy = self.calculate_lattice_energy()
            energy = cy_ising_model.calculate_lattice_energy(self.lattice, self.lattice_size, self.bond_energy)
            cluster_sizes.append(cluster_size)

        return cluster_sizes