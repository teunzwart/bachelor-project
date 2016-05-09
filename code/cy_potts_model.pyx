"""Performance critical functions for Potts Model Monte Carlo simulations."""

import numpy as np
import cython
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_lattice_energy(np.ndarray[np.int_t, ndim=2] lattice, int lattice_size, int bond_energy):
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
