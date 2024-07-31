# Brenda's Cauldron of Everything


## Description
Brenda's Cauldron of Everything is a python accessible library that contains useful tools to be shared among the researchers of the cosmo LSS group at MPE.


## Installation
As for the python packages, git cloning the repository is sufficient.
```
git clone git@gitlab.mpcdf.mpg.de:mesposito/brenda_lib.git
```

Some packages though are written in C with a cython interface. To use these, you need to compile and link them. A Makefile is provided for that. It is then sufficient to do:
```
cd brenda_lib
make
```
In case of errors in the compilation, make sure that you have cython installed (a pip installation is sufficient) and a gcc compiler.

## Usage
As for now, the package contains only two libraries for analysing Nbody simulations (pySim_lib and cySim_lib) and one with general utilities.

The two Sim_lib libraries do similar things, but pySim_lib is fully written in Python while cySim_lib is written in C with a cython interface and includes more features. What I suggest is to use cySim_lib as long as it works smoothly. If it gives you problems, then pySim_lib has more robust functions although they are much slower.

The cySim_lib package revolves around a cython class (G4_Simulation) which encompasses methods for reading particles and computing useful tools for dealing with Nbody simulations.

NEW: The package now contains a module called "cosmo_tools" which makes linear theory calculations. The module only has a class **Cosmology** which can be initialized with cosmological parameters and has methods for calculating growth factors, calling camb, ecc. and a function **get_cosmo** which can be called to get a class instance with one of the Aletheia cosmologies. See the tutorial in **brenda_lib_tutorial.ipynb** to check the usage.


