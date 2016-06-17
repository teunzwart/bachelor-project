"""Performance critical functions for Potts Model Monte Carlo simulations."""

import numpy as np
import cython
cimport numpy as np

from libc.math cimport exp as c_exp
from libc.stdlib cimport rand, RAND_MAX, malloc, free

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef calculate_lattice_energy(np.ndarray[np.int_t, ndim=2] lattice, int lattice_size, int bond_energy):
    """
    Calculate the energy of the lattice using the Potts model Hamiltonian in zero-field.

    The Cython implementations is 225 times faster than the pure Python implementation.
    (2.47 ms vs 562 ms for 1024**2 lattice)
    To increase performance only the bonds between x and x + 1 for the same y and
    y and y + 1 for the same x are calculated.
    Note: parallelism does not help here, at least it shows no speedup.
    """
    cdef int energy = 0
    cdef int y, x, center, offset_y, offset_x, xnn, ynn
    with nogil:
        for y in range(lattice_size):
            # The same for all x values, so can be precalculated.
            offset_y = y + 1
            # Wraparound the lattice. Note that truncated assignment operaters are used
            # to infer reduction variables, so you can't use those for offset_y.
            if y + 1 >= lattice_size:
                offset_y = offset_y - lattice_size
            for x in range(lattice_size):
                offset_x = x + 1
                if x + 1 >= lattice_size:
                    offset_x = offset_x - lattice_size
                center = lattice[y, x]
                # For some reason this is faster as a typed variable than directly
                # putting those lattice lookups in the if-statements.
                xnn = lattice[y, offset_x]
                ynn = lattice[offset_y, x]
                if xnn == center:
                    energy -=  bond_energy
                if ynn == center:
                    energy -=  bond_energy
    return energy


@cython.boundscheck(False)
@cython.wraparound(False)
def cy_potts_order_parameter(np.ndarray[np.int_t, ndim=2] lattice, int lattice_size):
    """
    Calculate the order parameter of the 2d 3-state Potts model.

    Done by considering each of the three possible orientations as an
    equally space vector on a plane with direction exp(2*pi*i*n/3) with
    n = 0, 1, 2. The direction of each state is multiplied by the number
    of spins in that state and the absolute value is taken.
    To prevent the use of cmath the real and imaginary parts are handled as
    ordinary numbers.
    """
    cdef double state1_im = 0.5 * 3**(0.5)
    cdef double state2_im = -0.5 * 3**(0.5)
    cdef int no_of_state0 = 0
    cdef int no_of_state1 = 0
    cdef int no_of_state2 = 0
    cdef int s, y, x
    for y in range(lattice_size):
        for x in range(lattice_size):
            s = lattice[y, x]
            if s == 0:
                no_of_state0 += 1
            elif s == 1:
                no_of_state1 += 1
            else:
                no_of_state2 += 1
    cdef double re = no_of_state0 - 0.5 * (no_of_state1 + no_of_state2)
    cdef double im = state1_im * no_of_state1 + state2_im * no_of_state2
    cdef double order_parameter = (re**2 + im**2)**0.5
    return order_parameter


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cy_metropolis(np.ndarray[np.int64_t, ndim=2] lattice, int lattice_size, int bond_energy, double beta, int sweeps):
    """Implentation of the Metropolis alogrithm."""
    cdef int t, rand_x, rand_y, spin, new_spin
    cdef double energy_delta, acceptance_probability, new_energy
    cdef double energy = calculate_lattice_energy(lattice, lattice_size, bond_energy)
    cdef double magnetization = cy_potts_order_parameter(lattice, lattice_size)
    cdef np.ndarray[np.float_t, ndim=1] energy_history = np.empty(sweeps, dtype=np.float64)
    cdef np.ndarray[np.float_t, ndim=1] magnetization_history = np.empty(sweeps, dtype=np.float64)
    for t in range(sweeps):
        # Measurement every sweep.
        energy_history[t] = energy
        magnetization_history[t] = magnetization
        for k in range(lattice_size**2):
            # Pick a random location on the lattice.
            rand_y = int(<double> rand() * lattice_size / RAND_MAX)
            rand_x = int(<double> rand() * lattice_size / RAND_MAX)

            spin = lattice[rand_y, rand_x]  # Get spin at the random location.
            new_spin = spin
            # Get a new spin different from the old spin
            while new_spin == spin:
                new_spin = int((<double> rand() * 3) / RAND_MAX)

            lattice[rand_y, rand_x] = new_spin
            new_energy = calculate_lattice_energy(lattice, lattice_size, bond_energy)
            energy_delta = new_energy - energy

            # Energy may always be lowered.
            if energy_delta <= 0:
                acceptance_probability = 1
            # Energy is increased with probability proportional to Boltzmann distribution.
            else:
                acceptance_probability = c_exp(-beta * energy_delta)
            if <double> rand() / RAND_MAX <= acceptance_probability:
                # Flip the spin and change the energy.
                energy += energy_delta
                magnetization = cy_potts_order_parameter(lattice, lattice_size)
            else:
                # Change back to the original configuration
                lattice[rand_y, rand_x] = spin

    return energy_history, magnetization_history


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cy_wolff(np.ndarray[np.int64_t, ndim=2] lattice, int lattice_size, int bond_energy, double beta, int sweeps):
    """Simulate the Potts lattice using the Wolff algorithm."""
    cdef double padd = 1 - c_exp(-beta * bond_energy)
    cdef double energy = calculate_lattice_energy(lattice, lattice_size, bond_energy)
    cdef np.ndarray[np.float_t, ndim=1] energy_history = np.empty(sweeps, dtype=np.float64)
    cdef np.ndarray[np.float_t, ndim=1] magnetization_history = np.empty(sweeps, dtype=np.float64)
    cdef np.ndarray[np.int_t, ndim=1] cluster_sizes = np.empty(sweeps, dtype=np.int)
    cdef int t, seed_x, seed_y, cluster_size, seed_spin, new_spin, prev_x, next_x, prev_y, next_y, current_x, current_y, stack_counter

    cdef int *stack_x = <int *>malloc(lattice_size**2 * sizeof(int))
    cdef int *stack_y = <int *>malloc(lattice_size**2 * sizeof(int))

    for t in range(sweeps):
        # print(t)
        # Measurement every sweep.
        energy_history[t] = energy
        magnetization_history[t] = cy_potts_order_parameter(lattice, lattice_size)

        stack = []  # Locations for which the neighbours still have to be checked.

        stack_counter = 1
        stack_x[0] = seed_x  # Locations for which the neighbours still have to be checked.
        stack_y[0] = seed_y

        # Pick a random location on the lattice as the seed.
        seed_y = int(<double> rand() * lattice_size / RAND_MAX)
        seed_x = int(<double> rand() * lattice_size / RAND_MAX)

        seed_spin = lattice[seed_y, seed_x]  # Get spin at the seed location.
        stack.append((seed_y, seed_x))

        new_spin = seed_spin
        # Get a new spin different from the old spin
        while new_spin == seed_spin:
            new_spin = int((<double> rand() * 3) / RAND_MAX)

        lattice[seed_y, seed_x] = new_spin  # Flip the spin.
        cluster_size = 1

        while stack:
            current_y, current_x = stack.pop()

            prev_x = current_x - 1
            if prev_x < 0:
                prev_x += lattice_size
            next_x = current_x + 1
            if next_x >= lattice_size:
                next_x -= lattice_size

            prev_y = current_y - 1
            if prev_y < 0:
                prev_y += lattice_size
            next_y = current_y + 1
            if next_y >= lattice_size:
                next_y -= lattice_size

            neighbours = [
                (current_y, prev_x),
                (current_y, next_x),
                (prev_y, current_x),
                (next_y, current_x)]

            for n in neighbours:
                if lattice[n] == seed_spin:
                    if np.random.random() < padd:
                        stack.append(n)
                        lattice[n] = new_spin
                        cluster_size += 1

        # Pure Python is very slow here.
        # energy = calculate_lattice_energy()
        energy = calculate_lattice_energy(lattice, lattice_size, bond_energy)
        cluster_sizes[t] = cluster_size

    free(stack_x)
    free(stack_y)
    return energy_history, magnetization_history, cluster_sizes
