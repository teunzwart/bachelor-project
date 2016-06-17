from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

modules = [Extension("cy_ising_model",
                     ["cy_ising_model.pyx"]),
           Extension("cy_potts_model",
                     ["cy_potts_model.pyx"])]

setup(name='Monte Carlo',
      ext_modules=cythonize(modules, annotate=True),
      include_dirs=[numpy.get_include()])
