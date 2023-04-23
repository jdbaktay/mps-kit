# Hamiltonians

import ncon as nc
import numpy as np
import scipy.linalg as spla
import scipy.sparse.linalg as spspla

si_half = np.array([[1, 0],[0, 1]])
sx_half = np.array([[0, 1],[1, 0]]) # gets 1/2
sy_half = np.array([[0, -1j],[1j, 0]]) # gets 1/2
sz_half = np.array([[1, 0],[0, -1]]) # gets 1/2

si_one = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
sx_one = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) # gets 1/sqrt2
sy_one = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) # gets 1/sqrt2
sz_one = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])

def TFI(J, g, size):
   si = si_half
   sx = sx_half
   sy = sy_half
   sz = sz_half

   if size == 'two':
      TFI = ((- J / 1) * np.kron(sx, sx) 
           + (- g / 2) * (np.kron(sz, si) + np.kron(si, sz))
           )
   return TFI

def XYZ_half(x, y, z, g, size):
   si = si_half
   sx = sx_half
   sy = sy_half
   sz = sz_half

   if size == 'two':
      XYZ = ((x / 1) * np.kron(sx, sx) 
           + (y / 1) * np.kron(sy, sy)
           + (z / 1) * np.kron(sz, sz)
           + (- g / 2) * (np.kron(sz, si) + np.kron(si, sz))
           )
   return XYZ

def XYZ_one(x, y, z, size):
   si = si_one
   sx = sx_one
   sy = sy_one
   sz = sz_one

   if size == 'two':
      XYZ = ((x / 2) * np.kron(sx, sx) 
           + (y / 2) * np.kron(sy, sy)
           + (z / 1) * np.kron(sz, sz)
           )
   return XYZ

def tV(t, V, mu):
   si, sx, sy, sz =  si_half, sx_half, sy_half, sz_half
   
   tV = ((-2 * t) * (1 / 4) * (np.kron(sx, sx) 
                             + np.kron(sy, sy)
                             )

              + V * (1 / 4) * np.kron(sz, sz)
             + mu * (1 / 4) * (np.kron(sz, si) 
                             + np.kron(si, sz)
                             )
             )
   return tV

def tVV2(t, V, V2, mu):
   si, sx, sy, sz =  si_half, sx_half, sy_half, sz_half

   tVV2 = ((-t / 4) * (np.kron(np.kron(sx, sx), si) 
                      + np.kron(np.kron(sy, sy), si)
                      )

          + (-t / 4) * (np.kron(si, np.kron(sx, sx)) 
                      + np.kron(si, np.kron(sy, sy))
                      )

          + (V / 8) * np.kron(np.kron(sz, sz), si)
          + (V / 8) * np.kron(si, np.kron(sz, sz))
          + (V2 / 4) * np.kron(np.kron(sz, si), sz)
          + (mu / 6) * (np.kron(sz, np.kron(si, si)) 
                     + np.kron(np.kron(si, sz), si)
                     + np.kron(np.kron(si, si), sz)
                     )
          )
   return tVV2

# def tt2tc(t, t2, tc):
#    tt2tc = ((-t / 4) * (np.kron(np.kron(sx, sx), si) + np.kron(np.kron(sy, sy), si))
#            + (-t / 4) * (np.kron(si, np.kron(sx, sx)) + np.kron(si, np.kron(sy, sy)))
#            + (t2 / 2) * (np.kron(sx, np.kron(sz, sx)) + np.kron(sy, np.kron(sz, sy)))
#            + (tc / 4) * (np.kron(sz, np.kron(sx, sx)) + np.kron(sz, np.kron(sy, sy)))
#            + (g / 6) * (np.kron(sz, np.kron(si, si)) + np.kron(np.kron(si, sz), si)
#                       + np.kron(np.kron(si, si), sz))
#            )
#    return tt2tc










