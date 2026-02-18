import numpy as np
from MajProp import *
import itertools


sn = 10 # sample numbers per (alpha_h, alpha_v)
dt = 0.01
n = 10 #num of timestep

#number of fermionic mode 
nf = 4
nf2 = 2 * nf

trott = 2 # trotterization order 

len_trunc = 6
coeff_trunc = 1e-6

init_bin = np.array([1, 1, 0, 0, 0, 0, 1, 1])
#init_bin = np.array([1, 1, 0, 0, 1, 1])
M1= MajoranaOp(nf2, init_bin)
N1 = Node(init_bin, 1j**M1.rb())

bin2 = np.array([1, 1, 1, 1, 1, 1, 1, 1])
#bin2 = np.array([0, 0, 1, 1, 0, 0])
M2= MajoranaOp(nf2, bin2)
N2 = Node(bin2, 1j**M2.rb())

bin3 = np.array([1, 0, 1, 0, 1, 0, 1, 0])
#bin3 = np.array([0, 0, 1, 1, 1, 1])
M3= MajoranaOp(nf2, bin3)
N3 = Node(bin3, 1j**M3.rb())

#bin4 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
#M4= MajoranaOp(nf2, bin4)
#N4 = Node(bin4, 1j**M4.rb())

Init_Node = [N1, N2, N3]
init_len = len(Init_Node)
init_maj = [M1, M2, M3]

