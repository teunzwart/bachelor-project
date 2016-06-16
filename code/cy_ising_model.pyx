"""Performance critical functions for Ising Model Monte Carlo simulations."""

from libc.math cimport exp as c_exp
from libc.stdlib cimport rand, RAND_MAX

import numpy as np
import cython
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef calculate_lattice_energy(np.ndarray[np.int_t, ndim=2] lattice, int lattice_size, int bond_energy):
    """
    Calculate the energy of the lattice using the Ising model Hamiltonian in zero-field.

    The Cython implementations is 100 to 500 times faster than the pure Python implementation.
    (1.64 ms vs 660 ms for 1024**2 lattice)
    To increase performance only the bonds between x and x + 1 for the same y and
    y and y + 1 for the same x are calculated.
    Plain function: 1 loop, best of 3: 669 ms per loop
    Class method: 1 loop, best of 3: 760 ms per loop
    Simple Cython: 1 loop, best of 3: 500 ms per loop
    Some typing: 1 loop, best of 3: 420 ms per loop
    Full cython solution: 100 loops, best of 3: 3.57 ms per loop
    Including parallel execution: 100 loops, best of 3: 1.64 ms per loop Both plain and cython versions give same output, 395 times speedup
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
                    # Only one reduction operator can be used, so substraction has to be done this way.
                    energy += -1 * bond_energy
                else:
                    energy += bond_energy
                if ynn == center:
                    energy += -1 * bond_energy
                else:
                    energy += bond_energy
    return energy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cy_metropolis(np.ndarray[np.int_t, ndim=2] lattice, int lattice_size, int bond_energy, double beta, int sweeps):
    """
    Implentation of the Metropolis alogrithm.

    On a 4 by 4 lattice 10000 sweeps at T=8 take 1.29 seconds in a pure Python implementation,
    and 12 ms in a Cython implementation (107 times speedup).
    """
    cdef int t, k, spin, rand_y, rand_x, spin_sum, prev_x, next_x, prev_y, next_y
    cdef double energy_delta, acceptance_probability
    cdef double energy = calculate_lattice_energy(lattice, lattice_size, bond_energy)
    cdef double magnetization = np.sum(lattice)
    cdef np.ndarray[np.float_t, ndim=1] energy_history = np.empty(sweeps, dtype=np.float64)
    cdef np.ndarray[np.float_t, ndim=1] magnetization_history = np.empty(sweeps, dtype=np.float64)
    with nogil:
        for t in range(sweeps):
            # Measurement every sweep.
            energy_history[t] = energy
            magnetization_history[t] = magnetization
            for k in range(lattice_size**2):
                # Pick a random location on the lattice.
                rand_y = int(<double> rand() * lattice_size / RAND_MAX)
                rand_x = int(<double> rand() * lattice_size / RAND_MAX)

                spin = lattice[rand_y, rand_x]  # Get spin at the random location.

                spin_sum = 0

                prev_x = rand_x - 1
                if prev_x < 0:
                    prev_x += lattice_size
                spin_sum += lattice[rand_y, prev_x]

                next_x = rand_x + 1
                if next_x >= lattice_size:
                    next_x -= lattice_size
                spin_sum += lattice[rand_y, next_x]

                prev_y = rand_y - 1
                if prev_y < 0:
                    prev_y += lattice_size
                spin_sum += lattice[prev_y, rand_x]

                next_y = rand_y + 1
                if next_y >= lattice_size:
                    next_y -= lattice_size
                spin_sum += lattice[next_y, rand_x]

                energy_delta = 2 * bond_energy * spin * spin_sum

                # Energy may always be lowered.
                if energy_delta <= 0:
                    acceptance_probability = 1
                # Energy is increased with probability proportional to Boltzmann distribution.
                else:
                    acceptance_probability = c_exp(-beta * energy_delta)
                if <double> rand()/RAND_MAX <= acceptance_probability:
                    # Flip the spin and change the energy.
                    lattice[rand_y, rand_x] = -1 * spin
                    energy += energy_delta
                    magnetization += -2 * spin
    return energy_history, magnetization_history
