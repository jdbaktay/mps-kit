# MPS Kit

This repository provides an implementation of Matrix Product States (MPS) methods from the ground up for the study of quantum many-body systems.

The code demonstrates how to write MPS methods from scratch (without pre-existing libraries beyond numpy/scipy), 
making it a useful resource for those looking to understand the underlying mechanics of MPS algorithms.

This code was not designed as a public-use library. 
The owner encourages the reader toward the many excellent libraries dedicated to tensor network methods.

## Main Contents
- VUMPS algorithm for calculating ground states, separated for two-site and three-site hamiltonians. See [here](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.045145) for details.
- Tangent-space methods for calculating excitations. See [here](https://scipost.org/SciPostPhysLectNotes.7/pdf) for details.
- Static and dynamic correlation functions implemented for spinless fermions.
- Various supporting tools for the main algorithms and additional analysis.

## Associated Publications
This code was used to produce the results in the following papers:
- [Phys. Rev. B 108, 245134](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.108.245134)
- [arXiv:2406.10063](https://arxiv.org/abs/2406.10063)
