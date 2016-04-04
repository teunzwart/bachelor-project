import logging
import random

import matplotlib
import matplotlib.pyplot as plt
import numba

import simulation_utils as utils

@numba.jit
def testing():
    total = 0
    for x in range(1000):
        for y in range(1000):
            for z in range(100):
                total += x*z+y

testing()


class IsingModel(utils.Simulation):
    """Simulate the Ising Model."""

    # @numba.autojit
    def main(self):
        # This is NOT the energy but the magnetization!
        energy = sum(sum(row) for row in self.lattice)
        # logging.debug("Energy is {0}".format(energy))
        for p in range(10000):
            pass
            # for q in range(self.xsize*self.ysize):
            #     rand_x = random.randint(0, self.xsize-1)
            #     rand_y = random.randint(0, self.ysize-1)
            #     temp_lattice = self.lattice
            #     temp_lattice[rand_y][rand_x] *= -1
            #     # logging.debug(temp_lattice)
            #     temp_energy = sum(sum(row) for row in temp_lattice)



        self.save_simulation_run("SUCCES")


if __name__ == "__main__":
    # arguments = utils.argument_parser()
    # ising = IsingModel(
    #     arguments.no_of_states,
    #     arguments.xsize,
    #     arguments.ysize,
    #     arguments.temperature,
    #     arguments.seed,
    #     arguments.debug,
    #     arguments.silent,
    #     arguments.nojson)
    ising = IsingModel(2, 10, 10, 'hi', debug=True, save_to_json=False)
    # ising.main()
