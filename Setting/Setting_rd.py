import numpy as np
from src.MajProp import *
import itertools

cn = 1 # number of cases 
sn = 2 # sample numbers per (alpha_h, alpha_v)
dt = 0.01
n = 10 #num of timestep

#number of fermionic mode 
nf = 6
nf2 = 2 * nf

trott = 1 # trotterization order 

len_trunc = 8
coeff_trunc = 1e-6

hist_save = False
seed_init = 393164

rng = np.random.default_rng(seed_init)
init_len = 6 # terms in intial observable
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



