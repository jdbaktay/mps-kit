# MPS Kit

This repository provides an implementation of Matrix Product States (MPS) methods from the ground up for the study of quantum many-body systems with novel extensions for fermions.

This branch, soon to be merged, is fully refactored for a more uniform, streamlined use, with additional features on the way.

The code demonstrates how to write MPS methods from scratch (without pre-existing libraries beyond numpy/scipy), chiefly useful for those looking to understand the underlying, fine-grained mechanics of MPS algorithms.

For those seeking a simpler, more pre-packaged experience, the owner encourages the reader toward the many excellent libraries dedicated to tensor network methods.

## Main Contents
- VUMPS algorithm for calculating ground states, separated for two-site and three-site hamiltonians. See [here](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.045145) for details.
- TDVP for uMPS in the mixed gauge for real and imaginary time evolution.
- Tangent-space methods for calculating excitations. See [here](https://scipost.org/SciPostPhysLectNotes.7/pdf) for details.
- Static and dynamic correlation functions implemented for spinless fermions.
- Various supporting tools for the main algorithms and additional analysis.

## Usage Notes
- All usage has been combined in the `run.py` file with a sample layout of the main modules.
  - Within `run.py`, we create a dictionary of the elements of `hamiltonians.py` so it can be more easily queried.
- Parameter input from command line follows this convention:
  `python filename.py model_name local_dim bond_dim hamiltonian_param_1 hamiltonian_param_2 hamiltonian_param_3 chemical_potential`
  which corresponds to the input conventions for the *fermion* hamiltonians in hamiltonians.py (not the spin hamiltonians).
  - To maintain the input convention, fermion models with more than 4 parameters have had the nearest-neighbor hopping set to t=1.
- For excitation files, add to the command line at the end `percentage_of_eigenvalues`.

## Associated Publications
This code was used to produce the results in the following papers:
- [Phys. Rev. B 108, 245134](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.108.245134)
- [arXiv:2406.10063](https://arxiv.org/abs/2406.10063)
