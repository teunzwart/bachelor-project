"""Helper utilities for simulations of the Ising and 3-state Potts model."""

import argparse
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

    def __init__(self, xsize, ysize, temperature, debug, rng_seed=None):
        """
        Initialize q-state Potts model simulation.

        Args:
            xsize: Extent of simulation in x-direction.
            ysize: Extent of simulation in y-direction.
            debug: Whether debugging info has to be shown.
            rng_seed: Optional seed for the random number generator.
        """
        self.start_time = time.time()
        self.date_format = "%Y-%m-%d %H:%M:%S"
        self._logging_config(debug)
        self._set_rng_seed(rng_seed)
        self.xsize = xsize
        self.ysize = ysize
        self.lattice = self._initiate_lattice(temperature)
        signal.signal(signal.SIGINT, self._interrupt_handler)
        logging.debug(self.lattice)
        logging.info("Simulation started.")

    def _set_rng_seed(self, rng_seed):
        """
        Set the seed of the random number generator.

        Setting the seed explicitly improves reproducibility as the same
        simulation can be run multiple times.

        Args:
            seed: Seed to use in random number generator.
        """
        if rng_seed is None:
            random.seed(self.start_time)
        else:
            random.seed(rng_seed)

    def _logging_config(self, debug):
        """
        Configure the format of logging messages.

        Args:
            debug: Whether debugging info has to be shown.
        """
        logging_format = "%(asctime)s %(levelname)s:%(message)s"
        if debug:
            logging.basicConfig(
                format=logging_format,
                datefmt=self.date_format,
                level=logging.DEBUG)
        else:
            logging.basicConfig(
                format=logging_format,
                datefmt=self.date_format,
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
        # TODO: Add way to pass cleanup function here.
        logging.debug("Done")
        sys.exit(1)

    def _initiate_lattice(self, temperature):
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
        if temperature == "hi":
            pass

        elif temperature == "lo":
            pass

        return initial_lattice


def argument_parser():
    """
    Parse command line arguments.

    Returns:
        Namespace containing commandline arguments.
    """
    parser = argparse.ArgumentParser(
        description="Simulate the Ising model in 2D.")
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
        help="specify seed for random rumber generator")
    arguments = parser.parse_args()
    return arguments

# TODO: Implement json save
