"""Plotting utilities for Monte Carlo simulations."""

import time

import matplotlib.pyplot as plt

SAVE_LOCATION = "../bachelor-thesis/images/"


def show_history(data, quantity):
    """Plot the quantity per spin for a run at a given temperature."""
    plt.title("{0} per Spin".format(quantity))
    plt.xlabel("Monte Carlo Sweeps")
    plt.ylabel("{0} per Spin".format(quantity))
    plt.plot(data)
    plt.show()


def show_lattice(lattice):
    """Show the lattice."""
    for tic in plt.gca().xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    for tic in plt.gca().yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    plt.gca().grid(False)
    plt.imshow(lattice, interpolation="nearest")
    plt.show()


def plot_correlation_time_range(data, lattice_size, quantity, show_plot=True, save=False):
    """Plot autocorrelation times for a range of temperatures."""
    plt.title("{0} Autocorrelation Time in Monte Carlo Sweeps".format(quantity))
    plt.xlabel("Temperature")
    plt.ylabel("Monte Carlo Sweeps")
    plt.plot([d[0] for d in data], [d[1] for d in data], marker='o', linestyle='None', label="{0} by {0} Lattice".format(lattice_size))
    plt.legend(loc='best')
    if save:
        plt.savefig("{0}{1}_Autocorrelation_Time_{2}.pdf".format(SAVE_LOCATION, quantity.replace(" ", "_"), time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))), bbox_inches='tight')
    if show_plot:
        plt.show()


def plot_quantity_range(data, errors, lattice_size, quantity, exact=None, show_plot=True, save=False):
    """Plot quantity over a temperature range."""
    plt.title(quantity)
    plt.xlabel("Temperature")
    plt.ylabel(quantity)
    plt.plot([d[0] for d in data], [d[1] for d in data], label="{0} by {0} Lattice".format(lattice_size), linestyle='None', marker='o')
    plt.errorbar([d[0] for d in data], [d[1] for d in data], [e[1] for e in errors], linestyle='None')
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
    plt.legend(loc="best")
    if save:
        plt.savefig("{0}{1}-{2}.pdf".format(SAVE_LOCATION, quantity.replace(" ", "_"), time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))), bbox_inches='tight')
    if show_plot:
        plt.show()
