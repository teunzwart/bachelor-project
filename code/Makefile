# To use OpenMP parallelism, an OpenMP capable compiler is needed.
# Clang on OSX does not natively support it.
all:
	CC=gcc-5 python3 setup.py build_ext --inplace

clean:
	rm *.c *.so
