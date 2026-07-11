import numpy as np
from src.MajProp import *
from src.Op import *
from src.Gate import *
import itertools

dt = 0.01
n = 11 #num of timestep

#number of fermionic mode 
nf = 6
nf2 = 2 * nf

trott = 2 # trotterization order 
len_trunc = 8
coeff_trunc = 1e-8

rmp_hist = True # save the length distribution histogram in RMP
random_init = True

h = np.zeros((nf2, nf2))
V = np.zeros((nf2, nf2, nf2, nf2))



if(random_init):
    seed_init = 23939151

    rng = np.random.default_rng(seed_init)
    init_len = 10 # terms in intial observable
    m = rng.integers(1, len_trunc//2 + 1, size = init_len)            # m in [1, len_trunc // 2]
    k = 2 * m



    init_bin = np.zeros((init_len, nf2), dtype=int)

    for i in range(init_len):
        idx = rng.choice(nf2, size=k[i], replace=False)
        init_bin[i][idx] = 1


    Init_Node = []
    init_maj = []

    for i in range(init_len):
        M = MajoranaOp(nf2, init_bin[i])
        N = Node(init_bin[i], 1j**M.rb())
        #print("Node", N.b, N.c)
        Init_Node.append(N)
        init_maj.append(M)

    cf_seed = 382315
    alpha_h = 10
    alpha_v = 10
    pairs_h, h = random_sparse_h(alpha_h, nf2, complex_coeff=False, seed=int(cf_seed))
    pairs_v, V = random_sparse_v(alpha_v, nf2, complex_coeff=False, seed=int(cf_seed))

else:
    h[0][1] = 1 
    h[0][2] = 1
    #h[0][3] = 1
    #h[0][4] = 1
    #h[2][4] = 1
    #h[1][5] = 1
    #h[1][5] = 0.2
    #h[2][3] = 0.5

    V[0][1][2][3] = 0.5
    V[1][2][3][4] = 0.5
    #V[2][3][4][5] = 0.5
    #V[1][3][4][5] = 0.1
    #V[0][1][3][4] = -3
    #V[0][1][4][6] = 1
    #V[0][1][6][7] = 1

    init_bin = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    #init_bin = np.array([1, 1, 0, 0, 1, 1])
    M1= MajoranaOp(nf2, init_bin)
    N1 = Node(init_bin, 1j**M1.rb())

    bin2 = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    #bin2 = np.array([0, 0, 1, 1, 0, 0, 0, 0])
    M2= MajoranaOp(nf2, bin2)
    N2 = Node(bin2, 1j**M2.rb())

    bin3 = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    #bin3 = np.array([0, 0, 0, 0, 1, 0, 1, 0])
    M3= MajoranaOp(nf2, bin3)
    N3 = Node(bin3, 1j**M3.rb())

    bin4 = np.array([1, 1, 1, 1, 1, 1, 0, 0])
    #bin4 = np.array([0, 0, 0, 0, 0, 0 , 1, 1])
    M4= MajoranaOp(nf2, bin4)
    N4 = Node(bin4, 1j**M4.rb())

    Init_Node = [N1, N2, N3, N4]
    init_len = len(Init_Node)
    init_maj = [M1, M2, M3, M4]


h_inds = np.nonzero(h) 
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



