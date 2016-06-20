This repository contains the code and thesis for my bachelor project in physics.

The focus of the project is the thermodynamic behaviour and the phase transition
of the 3-state Potts model (a generalization of the Ising model). The partition
function of this system can not be exactly determined. Therefore numerical
approaches using Monte Carlo methods are employed.
Using finite size scaling methods the critical temperature and critical exponents are found.

## Workflow
You can play with the simulations yourself.
The required Python 3 packages are
```
- Matplotlib
- Scipy
- Numpy
- Cython
- Seaborn
```
You also need a C-compiler.

To run a simulation yourself:
```shell
make # Build the Cython extensions
# Run a simulation of the Ising model through the command line
python3 ./simulation.py ising wolff 10 20 1 hi 5000 16384 2 3
```

The iPython notebooks `ising.ipynb` and `potts.ipynb` can be run to determine the critical temperature and critical exponents of the respective models. You can just do `run all`.
The required datasets are included in the repository.

```shell
python3 ./simulation.py -h
usage: simulation.py [-h] [--step STEP] [--show_plots] [--nosave]
                     {ising,potts} {metropolis,wolff} lattice_sizes
                     [lattice_sizes ...] bond_energy {hi,lo}
                     thermalization_sweeps measurement_sweeps lower upper

Simulate the 2- and 3-state Potts model in 2D.

positional arguments:
  {ising,potts}         Either Ising (q=2) or Potts (q=3)
  {metropolis,wolff}    Algorithm to use for simulation
  lattice_sizes         Lattice sizes to simulate
  bond_energy           Specify the bond energy
  {hi,lo}               specify initial temperature of simulation ("hi" is
                        infinite, "lo" is 0)
  thermalization_sweeps
                        Number of sweeps to perform before measurements start
  measurement_sweeps    Number of sweeps to measure
  lower                 Lower temperature bound
  upper                 Upper temperature bound

optional arguments:
  -h, --help            show this help message and exit
  --step STEP           Temperature step, default is 0.2
  --show_plots          Show plots on error calculations, default is false
                        because this is a blocking operation
  --nosave              Do not save output in a binary pickle file, default
                        behaviour is that it is saved
```
