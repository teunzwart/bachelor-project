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

    def __init__(self, xsize, ysize, debug):
        """
        Initialize q-state Potts model simulation.

        Args:
            xsize: Extent of simulation in x-direction.
            ysize: Extent of simulation in y-direction.
            debug: Whether debugging info has to be shown.
        """
        self.date_format = "%Y-%m-%d %H:%M:%S"
        self._logging_config(debug)

        self.xsize = xsize
        self.ysize = ysize
        self.start_time = time.time()
        # Explicitly set state of random number generator. This increases
        # reproduciblity. TODO: Implemnt seed passing.
        random.seed(self.start_time)
        self.start_time_human = time.strftime(
            self.date_format,
            self.start_time)
        signal.signal(signal.SIGINT, self._interrupt_handler)
        logging.info("Simulation started.")

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
        finished

        Args:
            interrupt_signal: Signal to handle (2 for KeyBoardInterrupt)
            frame: Current stack frame.
        """
        # Ignore any subsequent interrupt so cleanup can be performed.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        logging.info("Interrupted. Cleaning up...")
        # TODO: Add way to pass cleanup function here.
        logging.debug("Done")
        sys.exit(1)


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
        "-d", "--debug",
        help="print debugging information",
        action="store_true")
    arguments = parser.parse_args()
    return arguments

# TODO: Implemnt json save
# TODO: Implent hi/lo temp initial condition.
