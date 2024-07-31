from distutils.core import setup
from distutils.core import Extension
# from distutils.extension import Extension

from Cython.Build import cythonize
import numpy


setup(
    name="cySim_lib",
    ext_modules=cythonize(
        Extension(
            "cySim_lib", ["cySim_lib.pyx"],
            libraries=["csim_utils", "csubgrid", "cVoronoi_sample"], 
            extra_compile_args=['-fopenmp'], 
            extra_link_args=['-fopenmp'],
            include_dirs=[numpy.get_include(), 
                          './Clibs'],
            library_dirs=['./Clibs']
            ), 
        annotate = False
        ),
    )