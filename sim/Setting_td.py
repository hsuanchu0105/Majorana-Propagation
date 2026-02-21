import numpy as np
from MajProp import *
import itertools

dt = 0.01
n = 10 #num of timestep

#number of fermionic mode 
nf = 4
nf2 = 2 * nf

trott = 2 # trotterization order 
len_trunc = 8
coeff_trunc = 1e-8

h = np.zeros((nf2, nf2))

h[0][1] = 1 
h[0][2] = 1
#h[0][3] = 1
#h[0][4] = 1
#h[1][5] = 1
#h[1][5] = 0.2
#h[2][3] = 0.5


V = np.zeros((nf2, nf2, nf2, nf2))
V[0][1][2][3] = 0.5
V[1][2][3][4] = 0.5
#V[2][3][4][5] = 0.5
#V[1][3][4][5] = 0.1
#V[0][1][3][4] = -3
#V[0][1][4][6] = 1
#V[0][1][6][7] = 8


h_inds = np.nonzero(h) #seed 
v_inds = np.nonzero(V)



len_hs = len(h_inds[0])
len_vs = len(v_inds[0])

len_h = 2 * len_hs
len_v = len_vs

for i in range(len_hs):
    h[h_inds[1][i]][h_inds[0][i]] = -h[h_inds[0][i]][h_inds[1][i]]
	
#print(h)

#for i in range(len_v):
#	v_entry = [v_inds[0][i], v_inds[1][i], v_inds[2][i], v_inds[3][i]]
#	perms = [list(p) for p in itertools.permutations(v_entry)]
#	for j in range(len(perms)):
#		V[perms[j][0], perms[j][1], perms[j][2], perms[j][3]] = perm_parity(perms[j][0], perms[j][1], perms[j][2], perms[j][3]) * V[v_inds[0][i], v_inds[1][i], v_inds[2][i], v_inds[3][i]]
	
# different form of V might affect the efficiency (h we need it to be antisymmetric)

h_ind = np.nonzero(h)
v_ind = np.nonzero(V)

init_bin = np.array([1, 1, 0, 0, 0, 0, 0, 0])
#init_bin = np.array([1, 1, 0, 0, 1, 1])
M1= MajoranaOp(nf2, init_bin)
N1 = Node(init_bin, 1j**M1.rb())

bin2 = np.array([1, 1, 1, 1, 1, 1, 1, 1])
#bin2 = np.array([0, 0, 1, 1, 0, 0])
M2= MajoranaOp(nf2, bin2)
N2 = Node(bin2, 1j**M2.rb())

bin3 = np.array([1, 1, 1, 1, 0, 0, 0, 0])
#bin3 = np.array([0, 0, 1, 1, 1, 1])
M3= MajoranaOp(nf2, bin3)
N3 = Node(bin3, 1j**M3.rb())

bin4 = np.array([1, 1, 1, 1, 1, 1, 0, 0])
M4= MajoranaOp(nf2, bin4)
N4 = Node(bin4, 1j**M4.rb())

Init_Node = [N1, N2, N3, N4]
init_len = len(Init_Node)
init_maj = [M1, M2, M3, M4]

