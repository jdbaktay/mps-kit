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
- `run.py` accepts the following arguments for the CLI to configure the simulations:

### Required Arguments

| Argument   | Type   | Description                                      |
|------------|--------|--------------------------------------------------|
| `--model`  | `str`  | Type of Hamiltonian model (`Spin`, `Fermion`, etc.) |
| `--d`      | `int`  | Physical index dimension                         |
| `--D`      | `int`  | Virtual bond dimension                           |
| `--x`      | `float`| First Hamiltonian parameter                      |
| `--y`      | `float`| Second Hamiltonian parameter                     |
| `--z`      | `float`| Third Hamiltonian parameter                      |
| `--mu`     | `float`| Chemical potential                               |

### Optional Arguments

| Argument            | Type   | Description                                                    |
|---------------------|--------|----------------------------------------------------------------|
| `--percent_evals`    | `int`  | Percentage of total eigenvalues to retain (e.g., for truncation) |
| `--out_dir`         | `str`  | Output directory to save results. Defaults to current working directory. |

### Example

To run a simulation with the tVV2 fermion model with nearest-neighbor hopping t=x=1, nearest-neighbor interactions V=y=1, next-nearest-neighbor interactions V2=z=0 and chemical potential, mu=0:

```bash
python run.py --model tVV2 --d 2 --D 5 --x 1.0 --y 1.0 --z 0.0 --mu 0.0
```

- Within `run.py`, we create a dictionary of the elements of `hamiltonians.py` so it can be more easily queried.
- The convention for the hamiltonian parameters for command line input abides by the *fermion* hamiltonians in hamiltonians.py (not the spin hamiltonians).
  - To maintain this convention, fermion models with more than 4 parameters have had the nearest-neighbor hopping set to t=1.

## Associated Publications
This code was used to produce the results in the following papers:
- [Phys. Rev. B 108, 245134](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.108.245134)
- [arXiv:2406.10063](https://arxiv.org/abs/2406.10063)
