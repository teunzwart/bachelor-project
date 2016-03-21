"""Helper utilities for simulations of the Ising and 3-state Potts model."""


import argparse
import collections
import json
import logging
import math
import random
import signal
import sys
import time

import matplotlib.pyplot as plt


class Simulation():
    """
    Define a number of methods common to all q-state Potts model simulations.

    Classes implementing the actual simulations should inherit from this class.
    """

    def __init__(self, no_of_states, xsize, ysize, initial_temperature,
                 rng_seed=None, debug=False, silent=False, save_to_json=True):
        """
        Initialize q-state Potts model simulation.

        Args:
            no_of_states: 2 signifies Ising, 3 3-state Potts.
            xsize: Extent of simulation in x-direction.
            ysize: Extent of simulation in y-direction.
            initial_temperature: Initial temperature of the lattice.
            rng_seed: Optional seed for the random number generator.
            debug: Whether debugging info has to be shown.
            silent: Whether to run silent (useful for unittests).
            save_to_json: Whether information about the simulation has to be
                          saved in a json file.
        """
        self._argument_bound_check(no_of_states, xsize, ysize,
                                   initial_temperature, rng_seed, debug,
                                   silent)
        self._logging_config(debug, silent)
        logging.debug("Initializing simulation.")
        self.start_time = time.time()
        self.rng_seed = rng_seed
        self.no_of_states = no_of_states
        self.xsize = xsize
        self.ysize = ysize
        self.initial_temperature = initial_temperature
        self.save_to_json = save_to_json
        self._set_rng_seed()
        self.states = self._initialize_states()
        self.lattice = self._initialize_lattice()
        signal.signal(signal.SIGINT, self._interrupt_handler)
        logging.info("Simulation started.")

    def _set_rng_seed(self):
        """
        Set the seed of the random number generator.

        Setting the seed explicitly improves reproducibility as the same
        simulation can be run multiple times.
        """
        if self.rng_seed is None:
            self.rng_seed = self.start_time
            random.seed(self.rng_seed)
        else:
            random.seed(self.rng_seed)
        logging.debug("RNG seed has type {0}.".format(type(self.rng_seed)))
        logging.debug("RNG seed is {0}".format(self.rng_seed))

    def _argument_bound_check(self, no_of_states, xsize, ysize,
                              initial_temperature, rng_seed, debug, silent):
        """Check whether class arguments have valid values."""
        if no_of_states < 2 or type(no_of_states) is not int:
            raise ValueError("no_of_states has to be an integer larger than 2")
        if xsize < 1 or type(xsize) is not int:
            raise ValueError("xsize has to be an integer larger than 1")
        if ysize < 1 or type(ysize) is not int:
            raise ValueError("ysize has to be an integer larger than 1")
        if initial_temperature not in ["lo", "hi"]:
            raise ValueError("initial_temperature has to be either hi or lo")
        if rng_seed is not None:
            if type(rng_seed) not in [float, int]:
                raise TypeError("rng_seed has to be a float or integer")
        if debug and silent:
            raise Exception("debug and silent can not both be toggled")

    def _logging_config(self, debug, silent):
        """
        Configure the format of logging messages.

        Args:
            debug: Whether debugging info has to be shown.
        """
        logging_format = "%(asctime)s %(levelname)s:%(message)s"
        date_format = "%H:%M:%S"
        if debug:
            logging.basicConfig(format=logging_format, datefmt=date_format,
                                level=logging.DEBUG)
        elif silent:
            logging.basicConfig(format=logging_format, datefmt=date_format,
                                level=logging.CRITICAL)
        else:
            logging.basicConfig(format=logging_format, datefmt=date_format,
                                level=logging.INFO)

    def _interrupt_handler(self, *args):
        """
        Cleanly handle keyboard interrupts.

        Cleanup code is called and the program is terminated when cleanup
        finishes.
        """
        # Ignore any subsequent interrupt so cleanup can be performed.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        logging.info("Interrupted. Cleaning up...")
        self.save_simulation_run("INTERRUPTED")
        sys.exit(1)

    def _initialize_states(self):
        """Initialize the possible states a point on the lattice may take."""
        if self.no_of_states == 2:
            states = [-1, 1]
        elif self.no_of_states == 3:
            states = [-1, 0, 1]
        else:
            states = [-round(math.cos((n*math.pi) / (self.no_of_states-1)), 10)
                      for n in range(self.no_of_states)]
        logging.debug("States are {0}".format(states))
        return states

    def _initialize_lattice(self):
        """
        Initialize the lattice at either high or low temperature limits.

        High temperature means completly random distribution of spins, low
        temperature means completly ordered distribution.
        """
        if self.initial_temperature == "hi":
            initial_lattice = [[random.choice(self.states) for x in
                                range(self.xsize)] for y in range(self.ysize)]
        elif self.initial_temperature == "lo":
            initial_lattice = [[1 for x in range(self.xsize)] for y in
                               range(self.ysize)]
        return initial_lattice

    def save_simulation_run(self, status, **kwargs):
        """
        Save simulation information to a json file.

        Args:
            status: how the simulation finished (SUCCES, INTERRUPTED)
        """
        start_time_human = time.strftime("%Y-%m-%d %H:%M:%S",
                                         time.localtime(self.start_time))
        end_time = time.time()
        end_time_human = time.strftime("%Y-%m-%d %H:%M:%S",
                                       time.localtime(end_time))
        elapsed_time = round(end_time - self.start_time, 2)

        filename = "{0}state-{1}-{2}.json".format(
            self.no_of_states,
            status,
            time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(end_time)))

        plt.axis('off')
        plt.imshow(self.lattice, interpolation="nearest")
        # plt.show()

        if self.save_to_json:
            with open(filename, "w") as json_file:
                json.dump(collections.OrderedDict((
                    ("no_of_states", self.no_of_states),
                    ("status", status),
                    ("seed", self.rng_seed),
                    ("start_time", start_time_human),
                    ("end_time", end_time_human),
                    ("elapsed_time", elapsed_time),
                    ("initial_temperature", self.initial_temperature),
                    ("xsize", self.xsize),
                    ("ysize", self.ysize)
                )), json_file, indent=4)
        logging.info("Simulation ran for {0} seconds.".format(elapsed_time))
        logging.info("Simulation was {0}.".format(status))


def argument_parser():
    """
    Parse command line arguments.

    Returns:
        Namespace containing commandline arguments.
    """
    parser = argparse.ArgumentParser(
        description="Simulate the Ising model in 2D.")
    parser.add_argument(
        "no_of_states",
        help="number of states of Potts model (2 for Ising)",
        type=int)
    parser.add_argument(
        "xsize",
        help="specify x extent",
        type=int)
    parser.add_argument(
        "ysize",
        help="specify y extent",
        type=int)
    parser.add_argument(
        "temperature",
        choices=["hi", "lo"],
        help=('specify initial temperature of simulation ("hi" is infinite, '
              '"lo" is 0)'))
    parser.add_argument(
        "-s", "--seed",
        help="specify seed for random rumber generator (accepts only floats)",
        type=float)
    parser.add_argument(
        "-d", "--debug",
        help="print debugging information",
        action="store_true")
    parser.add_argument(
        "--silent",
        help="run silently",
        action="store_true")
    parser.add_argument(
        "-n", "--nojson",
        help="do not save information of simulation in a json file",
        action="store_false")
    arguments = parser.parse_args()
    return arguments
