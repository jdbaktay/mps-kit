from mps_kit import *

import os
import numpy as np
import hamiltonians
import inspect
import argparse
import matplotlib.pyplot as plt

def get_cfg():
    """
    Parses command-line arguments for configuring MPS simulations.
    Returns:
        argparse.Namespace with configuration attributes.
    """
    ap = argparse.ArgumentParser(description="Configure run.")

    ap.add_argument("--model", type=str, required=True,
                    help="Model type: e.g., Spin or Fermion Hamiltonian")

    ap.add_argument("--d", type=int, required=True,
                    help="Physical index dimension")

    ap.add_argument("--D", type=int, required=True,
                    help="Virtual bond dimension")

    hgroup = ap.add_argument_group("Hamiltonian parameters")
    hgroup.add_argument("--x", type=float, required=True,
                        help="First Hamiltonian parameter")
    hgroup.add_argument("--y", type=float, required=True,
                        help="Second Hamiltonian parameter")
    hgroup.add_argument("--z", type=float, required=True,
                        help="Third Hamiltonian parameter")
    hgroup.add_argument("--mu", type=float, required=True,
                        help="Chemical potential")

    ap.add_argument("--percent_evals", type=int, required=False, default=None,
                    help="Percentage of total eigenvalues to retain (optional)")

    ap.add_argument("--out_dir", type=str, required=False, default=None,
                    help="Directory to save output data (defaults to current working directory)")

    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = os.getcwd()

    return args

cfg = get_cfg()

model = cfg.model
d = cfg.d
D = cfg.D
x = cfg.x
y = cfg.y
z = cfg.z
mu = cfg.mu

if cfg.percent_evals==None:
    N = 0
else:
    N = int(np.floor(cfg.percent_evals / 100 * D**2))

params = (model, x, y, z, mu, D)
print('input params', params)

hamiltonian_dict = {name: obj for name, obj
                    in inspect.getmembers(hamiltonians, inspect.isfunction)}

h = hamiltonian_dict[model](x, y, z, mu)

if d == 2:
    sx = np.array([[0, 1],[1, 0]])
    sy = np.array([[0, -1j],[1j, 0]])
    sz = np.array([[1, 0],[0, -1]])

if d == 3:
    sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    sy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]])
    sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]]) # no 1/2

sp = 0.5 * (sx + 1.0j * sy)
sm = 0.5 * (sx - 1.0j * sy)
n = 0.5 * (sz + np.eye(d))

tol = stol = 1e-12

mps = mixed_canon_mps(d, D, tol, stol)

gauge_checks(*mps)

site_num = 'two'

gs_mps, gse = vumps(*mps, h, tol, stol, site_num, 0, 0)
# gse, error, gs_mps = tdvp(*mps, h, 0.2, tol, stol, site_num)

qm, nk = momentum_dist(*gs_mps, sp, sm, -sz)
qs, sk = stat_struc_fact(*gs_mps, n, n, None)

mom_vec = np.linspace(0, 1, 5) * np.pi

excits = excitations(*gs_mps, h, site_num, mom_vec, tol, N)

dsf = dynamic_struc_fact(*gs_mps, *excits, mom_vec, n)
specfxn = spectral_fxn(*gs_mps, *excits, mom_vec, sp, sm, sz)


filename = f'{model}_gs_{x}_{y}_{z}_{mu}_{D:03}_'
np.savez(os.path.join(cfg.out_dir, filename), AL=gs_mps[0],
                                              AR=gs_mps[1],
                                              C=gs_mps[-1],
)

filename = f'{model}_excits_{x}_{y}_{z}_{mu}_{D:03}_{N:05}_'
np.savez(os.path.join(cfg.out_dir, filename), mom=mom_vec,
                                              excit_energy=excits[0],
                                              excit_states=excits[-1]
                                              )
