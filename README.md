
# What is MesoHOPS?

MesoHOPS is a Python library for running simulations with the Hierarchy of Pure States (HOPS), a formally exact trajectory-based approach for solving the time-evolution of open quantum systems coupled to non-Markovian thermal environments. The main feature of MesoHOPS is the implementation of adaptive HOPS (adHOPS), an extension of the HOPS formalism that leverages the dynamic localization of excitations to construct an adaptive basis. The moving adHOPS basis significantly reduces the computational cost of simulations and exhibits a size-invariant scaling in large systems.


Get started with [the MesoHOPS website](https://captainexasperated.github.io/Readthedocs-Tutorial/)! 

# Dependencies and Installation
MesoHOPS is supported by [Python](https://www.python.org/) 3.9-3.11 and relies on the [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), and [Numba](https://numba.readthedocs.io/en/stable/#) packages. Additionally, our tests use the [pytest](https://docs.pytest.org/en/7.4.x/) and [pytest-level](https://pypi.org/project/pytest-level/) packages. All of these packages are automatically installed by the included pyproject.toml file.

To download and install MesoHOPS, enter the following commands:
```
git clone https://github.com/MesoscienceLab/mesohops.git
cd mesohops
python3 -m pip install .
```
Using a version of pip earlier than 23.3.2 may cause installation to fail.

# Using MesoHOPS
## Getting Started
To run a simple MesoHOPS simulation, please refer to our [quickstart tutorial](https://captainexasperated.github.io/Readthedocs-Tutorial/Quickstart/). [Additional tutorials](https://captainexasperated.github.io/Readthedocs-Tutorial/tutorials/) explain the use of mesoHOPS in more detail. We provide the input scripts for the advanced simulations run in various papers as supplementary materials.

## Units
The units of the HOPS equation of motion are energy and time, both given by the definition of $\hbar$, which is located in mesohops.util.physical_constants and is set to units of  cm$^{-1}\cdot$fs.

## Large-Scale Calculations
Because MesoHOPS supports perfectly-parallel trajectories, we recommend running large-scale simulations with parallelization through a Slurm array or the [multiprocessing package](https://docs.python.org/3/library/multiprocessing.html).

## Memory and CPU Profiling
When necessary, we recommend using [Memray](https://bloomberg.github.io/memray/) to profile memory and [SnakeViz](https://jiffyclub.github.io/snakeviz/) to profile CPU time.

## Convergence Scans
Because of the large number of convergence parameters present in MesoHOPS, testing convergence is non-trivial. We provide an introduction in our [convergence tutorial](https://captainexasperated.github.io/Readthedocs-Tutorial/Convergence/) and an in-depth example of convergence testing on production-scale calculations in the SI of [this paper on simulating exciton dynamics in large LH2 complexes](https://pubs.acs.org/doi/10.1021/acs.jpclett.3c00086).

# Developer Guide
We believe that good scientific code reflects two of the dearest scientific virtues: collaboration and clarity. As such, we encourage users to experiment with the MesoHOPS code and develop new features.

Most of the development in this library takes place in a private repository. Because we regularly implement new features and alter the structure of our code to improve performance and clarity, updates made to the current version of the MesoHOPS code directly may have to be rewritten for the next version of MesoHOPS.  If you have developed a feature or found an improvement that you believe is broadly useful, or are interested in development, please [reach out to us](https://cm.utexas.edu/component/cobalt/item/12-chemistry/5200-raccah-doran?Itemid=1251) about a potential collaboration.

## Style
All code developed by the MesoScience Lab follows stylistic guidelines laid out on our style guide (hosted on the [lab website](https://www.mesosciencelab.com/tools)). Documentation is the key to well-maintained code: as such, all functions and methods should be detailed with a docstring that clearly defines parameters and returns.

## Exceptions
We support a number of built-in exceptions at mesohops.util.exceptions for user and developer convenience.

## Testing
To test the code, navigate to the testing directory and enter the command
```
pytest
```
to run all tests or
```
pytest --level 1
```
to skip the time-consuming tests of Gaussian complex white noise generation, which is managed by code in mesohops/dynamics/hops_noise.py that should not be altered. Tests should be run whenever a change is made to the code to ensure that no unexpected errors have been introduced.

All newly-implemented features should be matched with unit tests (that is, a test of exactly one function or method) when possible and an integrated test (a test of how multiple pieces of the code interact) when necessary. Tests are managed with pytest and pytest-level and should be placed in an appropriate file in the testing directory. If a test is particularly time-consuming, it should have the decorator
```
@pytest.mark.level(2)
```
to allow users to run only the tests that take a short time. A properly-written test includes cases that intentionally result in errors to ensure that the expected exceptions are raised: examples may be found in our own code or in [the pytest documentation](https://docs.pytest.org/en/7.1.x/how-to/assert.html).

# Citing MesoHOPS
When using MesoHOPS version 1.4 or later please cite:
- B. Citty, J. K. Lynd, T. Gera, L. Varvelo, and D. I. G. B. Raccah, "MesoHops: Size-invariant scaling calculations of multi-excitation open quantum systems," [J. Chem. Phys. (2024)](https://doi.org/10.1063/5.0197825). 
*This paper extends the adaptive algorithm to account for arbitrary couplings between thermal environments and vertical excitation energies. Furthermore, it introduces a low-temperature correction and effective integration of the noise that simplify simulations with ultrafast vibrational relaxations.*

When using the adaptive basis (by setting  $\delta_A,\delta_S>0$) , please also cite:
- L. Varvelo, J. K. Lynd, and D. I. G. B(ennett) Raccah, "Formally exact simulations of mesoscale exciton dynamics in molecular materials," [Chem. Sci. (2021)](https://doi.org/10.1039/D1SC01448J). 
*This paper introduces and derives an adaptive basis construction algorithm motivated by dynamic localization in the HOPS equation of motion. Proof-of-concept calculations show that the resulting adaptive Hierarchy of Pure States (adHOPS) exhibits size-invariant scaling in large molecular aggregates.*

When using the linear absorption HOPS equation of motion, please also cite:
- L. Chen, D. I. G. B(ennett) Raccah, and A. Eisfeld, "Simulation of absorption spectra of molecular aggregates: A hierarchy of stochastic pure state approach," [J. Chem. Phys. (2022)](https://doi.org/10.1063/5.0078435). 
*This paper introduces a new HOPS equation of motion for simulating linear absorption spectra using a pure-state decomposition of the dipole correlation function.*

When using the Dyadic adaptive HOPS (DadHOPS) equations, please also cite:
- T. Gera, A. Hartzell, L. Chen, A. Eisfeld, and D. I. G. B. Raccah, "Formally exact fluorescence spectroscopy simulations for mesoscale molecular aggregates with $N^0$ scaling," [preprint (2025)](https://arxiv.org/abs/2503.00584).
*This paper extends the dyadic adaptive Hierarchy of Pure States (DadHOPS) implementation to simulate fluorescence spectra in large aggregates and introduces excitation operator decomposition, a generalization of the previously introduced initial state decomposition. Proof-of-concept calculations show that DadHOPS exhibits size-invariant scaling in large molecular aggregates.*  
- T. Gera, L. Chen, A. Eisfeld, J. R. Reimers, E. J. Taffet, and D. I. G. B. Raccah, "Simulating optical linear absorption for mesoscale molecular aggregates: An adaptive hierarchy of pure states approach," [J. Chem. Phys. (2023)](https://doi.org/10.1063/5.0141882).
*This paper introduces a dyadic adaptive Hierarchy of Pure States (DadHOPS) implementation for simulating linear absorption spectra in large aggregates, as well as an initial state decomposition that allows for convenient scaling. Proof-of-concept calculations show that DadHOPS exhibits size-invariant scaling in large molecular aggregates.*


To better understand the NMQSD and HOPS formalism, please review:
- L. Diósi and W. T. Strunz, "The non-Markovian stochastic Schrödinger equation for open systems," [Phys. Lett. A (1997)](https://doi.org/10.1016/S0375-9601(97)00717-2).
*This paper presents a formally exact non-Markovian stochastic Schrödinger (NMQSD) equation of motion for trajectories representing realizations of the environmental state of open quantum systems.*

- D. Suess, A. Eisfeld, and W. T. Strunz, "Hierarchy of Stochastic Pure States for Open Quantum System Dynamics," [Phys. Rev. Lett. (2014)](https://doi.org/10.1103/PhysRevLett.113.150403). 
*This paper derives the Hierarchy of Pure States (HOPS) equation, a solution to the formally exact trajectory-based Non-Markovian Quantum State Diffusion (NMQSD) method for solving the time-evolution of open quantum systems. Both a linear and nonlinear HOPS equation are presented.*

- L. Chen, D. I. G. B(ennett) Raccah, and A. Eisfeld, "Calculating nonlinear response functions for multidimensional electronic spectroscopy using dyadic non-Markovian quantum state diffusion," [J. Chem. Phys. (2022)](https://doi.org/10.1063/5.0107925). 
*This paper introduces an NMQSD formalism propagated in a dyadic Hilbert space to construct multi-point time correlation functions. This is then mapped into HOPS calculations of nonlinear spectra.*