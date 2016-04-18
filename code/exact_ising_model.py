"""Exact solutions for the 2D Ising model magnetization, internal energy and heat capacity."""

import numpy as np
from scipy.special import ellipe, ellipk


def magnetization(bond_energy, lower_temperature, higher_temperature, step=0.02):
    """
    Calculate the exact magnetization. Boltzmann constant is set to 1.

    Formula from McCoy and Wu, 1973, The Two-Dimensional Ising Model.
    """
    exact_magnetization = []

    for t in np.arange(lower_temperature, higher_temperature, step):
        m = (1 - np.sinh((2 / t) * bond_energy)**(-4))**(1 / 8)
        if np.isnan(m):
            m = 0
        exact_magnetization.append((t, m))

    return exact_magnetization


def internal_energy(bond_energy, lower_temperature, higher_temperature, step=0.02):
    """
    Calculate the exact internal energy. Boltzmann constant is set to 1.

    Formula from McCoy and Wu, 1973, The Two-Dimensional Ising Model.
    """
    # Shorter variable name to ease formula writing.
    j = bond_energy
    exact_internal_energy = []
    for t in np.arange(lower_temperature, higher_temperature, step):
        b = 1 / t
        k = 2 * np.sinh(2 * b * j) / np.cosh(2 * b * j)**2
        u = -j * (1 / np.tanh(2 * b * j)) * (1 + (2 / np.pi) * (2 * np.tanh(2 * b * j)**2 - 1) * ellipk(k**2))
        exact_internal_energy.append((t, u))

    return exact_internal_energy


def heat_capacity(bond_energy, lower_temperature, higher_temperature, step=0.02):
    """
    Calculate the exact heat capacity. Boltzmann constant is set to 1.

    Formula from McCoy and Wu, 1973, The Two-Dimensional Ising Model.
    """
    # Shorter variable name to ease formula writing.
    j = bond_energy
    exact_heat_capacity = []
    for t in np.arange(lower_temperature, higher_temperature, step):
        b = 1 / t
        k = 2 * np.sinh(2 * b * j) / np.cosh(2 * b * j)**2
        kprime = np.sqrt(1 - k**2)
        c = (b * j / np.tanh(2 * b * j))**2 * (2 / np.pi) * (2 * ellipk(k**2) - 2 * ellipe(k**2) - (1 - kprime) * (np.pi / 2 + kprime * ellipk(k**2)))
        exact_heat_capacity.append((t, c))

    return exact_heat_capacity
