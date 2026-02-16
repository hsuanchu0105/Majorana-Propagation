import numpy as np
from MajProp import *

dt = 0.01
n = 8 #num of timestep

#number of fermionic mode 
nf = 4
nf2 = 2 * nf


h = np.zeros((nf2, nf2))

h[2][0] = -1 
h[4][0] = -1
#h[5][1] = -0.2
#h[3][2] = -0.2
#h[5][1]= -0.1


for i in range(nf2):
    for j in range(i+1, nf2):
        h[i][j] = -h[j][i]

#print(h)



V = np.zeros((nf2, nf2, nf2, nf2))
#V[0][1][2][3] = 0.5
#V[1][2][3][4] = 0.5
#V[2][3][4][5] = 0.5
#V[1][3][4][5] = 0.1
#V[0][1][3][4] = -3
V[0][1][4][6] = 1
V[0][1][6][7] = 8

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

h_ind = np.nonzero(h)
v_ind = np.nonzero(V)