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
You also need an OpenMP-compatible C compiler.

To run a simulation yourself:
```shell
make # Build the Cython extensions
# Run a simulation of the Ising model through the command line
python3 ./simulation.py ising wolff 10 20 1 hi 5000 16384 2 3
```

The iPython notebooks `ising.ipynb` and `potts.ipynb` can be run to determine the critical temperature and critical exponents of the respective models. You can just do `run all`.
The required datasets are included in the repository.
