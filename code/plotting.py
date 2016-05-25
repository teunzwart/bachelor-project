"""Plotting utilities for Monte Carlo simulations."""

import itertools
import time

import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
plt.rc('text', usetex=True)
sns.set_style("ticks")
sns.set_palette('colorblind')  # Options: deep, muted, pastel, bright, dark, colorblind

SAVE_LOCATION = "../bachelor-thesis/images"


def show_history(data, quantity):
    """Plot a quantity for a run at a given temperature."""
    plt.xlabel(r'$Monte Carlo Sweeps$')
    plt.ylabel(r'${0}$'.format('\mathrm{' + quantity.replace(' ', '\ ') + '}'))
    plt.plot(data)
    plt.show()


def show_lattice(lattice, lattice_size, save=False):
    """Show the lattice."""
    for tic in plt.gca().xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    for tic in plt.gca().yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    plt.gca().grid(False)
    plt.imshow(lattice, interpolation="nearest")
    if save:
        plt.savefig("{0}/{1}_{2}_by_{2}_Lattice.pdf".format(SAVE_LOCATION, time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time())), lattice_size), bbox_inches='tight')
    plt.show()


def show_cluster(cluster, lattice_size):
    """Show a Wolff algorithm cluster."""
    cluster_image = np.zeros((lattice_size, lattice_size))
    for k in cluster:
        cluster_image[k] = 1
    for tic in plt.gca().xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    for tic in plt.gca().yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    plt.gca().grid(False)
    plt.imshow(cluster_image, interpolation='nearest')
    plt.show()


def plot_correlation_time_range(data_range, quantity, show_plot=True, save=False):
    """Plot autocorrelation times for a range of temperatures."""
    plt.xlabel(r'$T$')
    plt.ylabel(r'${0}$'.format('\mathrm{' + quantity.replace(' ', '\ ') + "\ Autocorrelation\ Time\ in\ Monte\ Carlo\ Sweeps}"))
    for p in data_range:
        data = p[1]
        plt.plot([d[0] for d in data], [d[1] for d in data], marker='o', linestyle='None', label=r"${0}$".format(str(p[0]) + '\mathrm{\ by\ }' + str(p[0]) + "\mathrm{\ Lattice}"))
    plt.legend(loc='best')
    # We put all data together so it is easy to find the maximum and minimum values.
    zipped_data = list(itertools.chain(*[k[1] for k in data_range]))
    x_data_range = sorted(list(set(m[0] for m in zipped_data)))
    min_x = x_data_range[0]
    max_x = x_data_range[-1]
    x_step = x_data_range[1] - x_data_range[0]
    plt.xlim(min_x - x_step, max_x + x_step)
    data_max = max(zipped_data, key=lambda x: x[1])[1]
    plt.ylim(0, data_max * 1.15)
    sns.despine()
    if save:
        plt.savefig("{0}/{1}_Autocorrelation_Time_{2}.pdf".format(SAVE_LOCATION, quantity.replace(" ", "_"), time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))), bbox_inches='tight')
    if show_plot:
        plt.show()


def plot_quantity_range(data_range, quantity, exact=None, show_plot=True, save=False):
    """Plot quantity over a temperature range."""
    for p in data_range:
        data = p[1]
        plt.errorbar([d[0] for d in data], [d[1] for d in data], [d[2] for d in data], linestyle='None', label=r"${0}$".format(str(p[0]) + '\mathrm{\ by\ }' + str(p[0]) + "\mathrm{\ Lattice}"), marker='o')
    if exact is not None:
        plt.plot([e[0] for e in exact], [e[1] for e in exact], label=r'$\mathrm{Thermodynamic\ Limit}$')

    # We put all data together so it is easy to find the maximum and minimum values.
    zipped_data = list(itertools.chain(*[k[1] for k in data_range]))
    x_data_range = sorted(list(set(m[0] for m in zipped_data)))
    min_x = x_data_range[0]
    max_x = x_data_range[-1]
    x_step = x_data_range[1] - x_data_range[0]
    plt.xlim(min_x - x_step, max_x + x_step)
    data_min = min(zipped_data, key=lambda x: x[1])[1]
    data_max = max(zipped_data, key=lambda x: x[1])[1]
    if data_max <= 0:
        plt.ylim(ymin=1.15 * data_min, ymax=0.85 * data_max)
    else:
        plt.ylim(ymin=0, ymax=data_max * 1.15)
    plt.xlabel(r'$T$')
    plt.ylabel(r'${0}$'.format('\mathrm{' + quantity.replace(' ', '\ ') + '}'))
    plt.legend(loc="best")
    sns.despine()
    if save:
        plt.savefig("{0}/{1}_{2}.pdf".format(SAVE_LOCATION, quantity.replace(" ", "_"), time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))), bbox_inches='tight')
    if show_plot:
        plt.show()
