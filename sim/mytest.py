import numpy as np
from scipy.linalg import expm



def make_meshgrid(arrays, indexing="ij"):
    """
    arrays: list/tuple of 1D arrays, length = s 
    returns: list of s arrays, all with the same shape
    """
    return np.meshgrid(*arrays, indexing=indexing)
    
s = 4
sizes = [2, 3, 4, 5][:s]  # take from 0 to sth (exclude) element from [2,3,4,5]

axes = [np.arange(n) for n in sizes]   # list of s 1D arrays

print(axes)

grids = make_meshgrid(axes, indexing="ij")
#g is one of the meshgrid arrays, g.ravel(): flatten g, np.stack() makes it into a 2d array 
coords = np.stack([g.ravel() for g in grids], axis=-1)   

print(coords)