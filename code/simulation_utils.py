"""Helper utilities for simulations of the Ising and 3-state Potts model."""


import argparse
import collections
import json
import logging
import random
import signal
import sys
import time


class Simulation():
    """
    Define a number of methods common to all q-state Potts model simulations.

    Classes implementing the actual simulations should inherit from this class.
    """

    def __init__(self, no_of_states, xsize, ysize, initial_temperature, debug,
                 rng_seed=None):
        """
        Initialize q-state Potts model simulation.

        Args:
            xsize: Extent of simulation in x-direction.
            ysize: Extent of simulation in y-direction.
            debug: Whether debugging info has to be shown.
            rng_seed: Optional seed for the random number generator.
        """
        self.start_time = time.time()
        self._logging_config(debug)
        self.rng_seed = rng_seed
        self._set_rng_seed()
        self.no_of_states = no_of_states
        self.states = self._initialize_states()
        self.xsize = xsize
        self.ysize = ysize
        self.lattice = self._initialize_lattice(initial_temperature)
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

    def _logging_config(self, debug):
        """
        Configure the format of logging messages.

        Args:
            debug: Whether debugging info has to be shown.
        """
        logging_format = "%(asctime)s %(levelname)s:%(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        if debug:
            logging.basicConfig(format=logging_format, datefmt=date_format,
                                level=logging.DEBUG)
        else:
            logging.basicConfig(format=logging_format, datefmt=date_format,
                                level=logging.INFO)

    def _interrupt_handler(self, interrupt_signal, frame):
        """
        Cleanly handle keyboard interrupts.

        Cleanup code is called and the program is terminated when cleanup
        finishes.

        Args:
            interrupt_signal: Signal to handle (2 for KeyBoardInterrupt).
            frame: Current stack frame.
        """
        # Ignore any subsequent interrupt so cleanup can be performed.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        logging.info("Interrupted. Cleaning up...")
        self.save_simulation_run("INTERRUPTED")
        sys.exit(1)

    def _initialize_states(self):
        if self.no_of_states == 2:
            return [-1, 1]
        elif self.no_of_states == 3:
            return [-1, 0, 1]

    def _initialize_lattice(self, initial_temperature):
        """
        Initiate the lattice at either high or low temperature limits.

        High temperature means completly random distribution of spins, low
        temperature means completly ordered distribution.

        Args:
            temperature: Initial temperature of simulation.

        Returns:
            initial_lattice: List of lists describing the lattice.
        """
        initial_lattice = []
        for y in range(self.ysize):
            lattice_row = []
            for x in range(self.xsize):
                if initial_temperature == "hi":
                    lattice_row.append(random.choice(self.states))
                elif initial_temperature == "lo":
                    lattice_row.append(1)
            initial_lattice.append(lattice_row)

        return initial_lattice

    def save_simulation_run(self, status):
        start_time_human = time.strftime("%Y-%m-%d %H:%M:%S",
                                         time.localtime(self.start_time))
        end_time = time.time()
        end_time_human = time.strftime("%Y-%m-%d %H:%M:%S",
                                       time.localtime(end_time))
        elapsed_time = round(end_time - self.start_time, 2)
        if self.no_of_states == 2:
            filename = "Ising-{0}".format(
                time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(end_time)))
        elif self.no_of_states == 3:
            filename = "Potts-{0}".format(
                time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(end_time)))
        with open("{0}.json".format(filename), "w") as file:
            json.dump(collections.OrderedDict((
                ("no_of_states", self.no_of_states),
                ("status", status),
                ("seed", self.rng_seed),
                ("start_time", start_time_human),
                ("end_time", end_time_human),
                ("elapsed_time", elapsed_time)
            )), file, indent=4)
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
        help="number of states of Potts model (2 for Ising, 3 for 3-state)",
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
        "-d", "--debug",
        help="print debugging information",
        action="store_true")
    parser.add_argument(
        "-s", "--seed",
        help="specify seed for random rumber generator (accepts only floats)",
        type=float)
    arguments = parser.parse_args()
    return arguments

# TODO: Add bound cheching for commandline arguments
