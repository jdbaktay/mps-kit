# MPS Kit

This repository provides an implementation of Matrix Product States (MPS) methods from the ground up for the study of quantum many-body systems.

The code demonstrates how to write MPS methods from scratch (without pre-existing libraries beyond numpy/scipy), 
making it a useful resource for those looking to understand the underlying mechanics of MPS algorithms.

This code was not intended/designed as a public-use library. 
The owner encourages the reader toward the many excellent libraries dedicated to tensor network methods.

## Main Contents
- VUMPS algorithm for calculating ground states, separated for two-site and three-site hamiltonians. See [here](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.045145) for details.
- Tangent-space methods for calculating excitations. See [here](https://scipost.org/SciPostPhysLectNotes.7/pdf) for details.
- Static and dynamic correlation functions implemented for spinless fermions.
- Various supporting tools for the main algorithms and additional analysis.

## Usage Notes
- Parameter input from command line follows this convention:  
  `python filename.py model_name local_dim bond_dim hamiltonian_param_1 hamiltonian_param_2 hamiltonian_param_3 chemical_potential`
  which corresponds to the input conventions for the *fermion* hamiltonians in hamiltonians.py (not the spin hamiltonians)
  - To maintain the input convention, fermions models with more than 4 parameters have had nearest-neighbor hopping set to t=1 
- For excitation files, add to the command line at the end `number_of_eigenvalues momentum_value`
- Because the excitation calculations are the most computationally expensive the parameter input and file output specifies a *single* momentum value to be used in HPC contexts. By comparison, the inputs/outputs of the dynamic correlations do *not* follow this convention.

## Associated Publications
This code was used to produce the results in the following papers:
- [Phys. Rev. B 108, 245134](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.108.245134)
- [arXiv:2406.10063](https://arxiv.org/abs/2406.10063)
