"""A Monte Carlo simulation of the Ising model."""

import numpy as np

import plotting


class IsingModel:
    """A Monte Carlo simulation of the Ising model."""

    def __init__(self, lattice_size, bond_energy, temperature,
                 initial_temperature, sweeps):
        """Initialize variables and the lattice."""
        self.rng_seed = int(lattice_size * temperature * 1000)
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

    def wolff(self, show_progress=False):
        """Simulate the lattice using the Wolff algorithm."""
        padd = 1 - np.exp(-2 * self.beta * self.bond_energy)
        cluster_sizes = []
        for t in range(self.sweeps):
            # Measurement every sweep.
            np.put(self.energy_history, t, self.energy)
            np.put(self.magnetization_history, t, np.sum(self.lattice))

            cluster = []
            to_consider = []  # Locations for which the neighbours still have to be checked.

            # Pick a random location on the lattice as the seed.
            seed_y = np.random.randint(0, self.lattice_size)
            seed_x = np.random.randint(0, self.lattice_size)
            cluster.append((seed_y, seed_x))

            seed_spin = self.lattice[seed_y, seed_x]  # Get spin at the seed location.
            neighbours = [
                (seed_y, (seed_x - 1) % self.lattice_size),
                (seed_y, (seed_x + 1) % self.lattice_size),
                ((seed_y - 1) % self.lattice_size, seed_x),
                ((seed_y + 1) % self.lattice_size, seed_x)]

            for n in neighbours:
                if self.lattice[n] == seed_spin:
                    if np.random.random() < padd:
                        cluster.append(n)
                        to_consider.append(n)

            while to_consider:
                neighbour_y, neighbour_x = to_consider.pop()
                neighbours = [
                    (neighbour_y, (neighbour_x - 1) % self.lattice_size),
                    (neighbour_y, (neighbour_x + 1) % self.lattice_size),
                    ((neighbour_y - 1) % self.lattice_size, neighbour_x),
                    ((neighbour_y + 1) % self.lattice_size, neighbour_x)]

                for n in neighbours:
                    if n in cluster:
                        continue
                    if self.lattice[n] == seed_spin:
                        if np.random.random() < padd:
                            cluster.append(n)
                            to_consider.append(n)

            # Flip all spins in the cluster.
            for spin in cluster:
                self.lattice[spin] *= -1
            self.energy = self.calculate_lattice_energy()

            cluster_sizes.append(len(cluster))

            if t % 100 == 0 and show_progress:
                print("Sweep {0}".format(t))
                plotting.show_cluster(cluster, self.lattice_size)

        return cluster_sizes



if __name__ == "__main__":
    ising = IsingModel(4, 1, 2, "lo", 1000)
    ising.metropolis()
    print(ising.lattice)
    print(ising.energy)
    print(ising.magnet_history)
    print(ising.energy_history)
