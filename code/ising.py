import logging
import random
import time

import matplotlib
import matplotlib.pyplot as plt
import seaborn

import simulation_utils as utils


class IsingModel(utils.Simulation):
    """
    Simulate the Ising Model
    """


    def main(self):
        total_energy = 0
        for t in range(10000):
            for x in range(self.xsize):
                for y in range(self.ysize):
                    energy = random.random()
                    if energy < 0.5:
                        total_energy += energy
        logging.debug(total_energy)
        self.save_simulation_run("SUCCES")



if __name__ == "__main__":
    arguments = utils.argument_parser()
    print(arguments)
    print(type(arguments.xsize))
    ising = IsingModel(
        arguments.no_of_states,
        arguments.xsize,
        arguments.ysize,
        arguments.temperature,
        arguments.debug,
        arguments.seed,)
    ising.main()
