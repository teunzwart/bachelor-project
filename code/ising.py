import argparse
import json
import logging
import time

import matplotlib.pyplot as plt


class IsingModel:
    """

    """

    def __init__(self, xsize, ysize, debug):
        """
        Initialize Ising model simulation.

        Args:
            xsize: Extent of simulation in x-direction.
            ysize: Extent of simulation in y-direction.
            debug: Whether debugging info has to be shown.
        """
        if debug:
            logging.basicConfig(
                format="%(levelname)s:%(message)s",
                level=logging.DEBUG)
        self.xsize = xsize
        self.ysize = ysize
        self.start_time = time.time()
        self.start_time_human = 0  #TODO: Something todo
        logging.info("Simulation started on {0}".format(time.strftime("%a, %d %b %Y %H:%M:%S, time.localtime())))


def _argument_parser():
    """
    Parse command line arguments.

    Returns:
        Namespace containing argumentsself.
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

if __name__ == "__main__":
    arguments = _argument_parser()
    print(arguments)
    ising = IsingModel(arguments.xsize, arguments.ysize, arguments.debug)
