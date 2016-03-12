import logging
import time

import matplotlib.pyplot as plt

import simulation_utils as utils


class IsingModel(utils.Simulation):
    """

    """


    def main(self):
        while True:
            logging.debug("Sleeping...")
            time.sleep(1)


if __name__ == "__main__":

    arguments = utils.argument_parser()
    ising = IsingModel(
        arguments.xsize,
        arguments.ysize,
        arguments.debug,
        arguments.seed,
        arguments.temperature)
    ising.main()
