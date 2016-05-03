from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

modules = [Extension("cy_ising_model",
                     ["cy_ising_model.pyx"],
                     extra_compile_args=["-fopenmp"],
                     extra_link_args=["-fopenmp"])]

setup(name='cy_ising_model',
      cmdclass={"build_ext": build_ext},
      ext_modules=modules,
      include_dirs=[numpy.get_include()])
