from MajProp import * 
from Setting_rd import *
import time 
from datetime import date
import sys
import os

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


dir_ = f"ta{date.today():%m%d}/"
note = "_rd_ana"

nz_seed = 21
mst_seed = 123

nmin_h=2
nmax_h=2
nmin_v=2
nmax_v=2

#sps = sparsity(h_ind, v_ind, nf2)
#print("Delta = ", sps)

#print(np.allclose(len(U), (n+1) * np.count_nonzero(h) + 2 * n * np.count_nonzero(V)))

err = np.zeros(sn)
err_rot = np.zeros(sn)

for cs in range(cn):
    # for each case we have one (alpha_h, alpha_v) pair
    rng_nzc = np.random.default_rng(nz_seed)  # non-zero count 
    # here alpha_h only consider upper triangle terms 
    alpha_h = rng_nzc.integers(nmin_h, nmax_h + 1)
    alpha_v = rng_nzc.integers(nmin_v, nmax_v + 1)

    log_path =  dir_  + str(len_trunc) + "_" + str(coeff_trunc) + "_" + str(nf) + "_" + str(dt) + "_" + str(n) + "_" + str(init_len)+ "_" + str(alpha_h) + "_" + str(alpha_v) + note  + ".txt"

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print("alpha_h, alpha_v = ", alpha_h, alpha_v)
    print('\t')

    master_rng = np.random.default_rng(mst_seed)  # pick any fixed master seed
    seeds = master_rng.integers(0, 2**32 - 1, size=sn, dtype=np.uint32)

    print("seed for nonzero terms = ", nz_seed)
    print("master seed =  ", mst_seed)
    print("seed for coefficient generation:", seeds)
    print('\t')


    for s in range(sn):
        
        pairs_h, h = random_sparse_h(alpha_h, nf2, complex_coeff=False, seed=int(seeds[s]))
        pairs_v, V = random_sparse_v(alpha_v, nf2, complex_coeff=False, seed=int(seeds[s]))


        #print(h)
        print(pairs_h)
        print(pairs_v)

        h_ind = np.nonzero(h)
        v_ind = np.nonzero(V)



        U = []

        AppendH(U, h_ind, dt, trott, nf2)

        AppendV(U, v_ind, dt, trott, nf2)

        for i in range(n-1):
            AppendH(U, h_ind, 2 * dt, trott, nf2)
            AppendV(U, v_ind, dt, trott, nf2)

        AppendH(U, h_ind, dt, trott, nf2)
            
        print("gate count", len(U))
        
        trunc_param = np.array([len_trunc, coeff_trunc])
        
        #print("Fermionic gate:", U)
        #print('\t')
        rho_st = np.zeros(nf, dtype = int)
        c = np.zeros(nf, dtype = int)

        tic = time.perf_counter()
        Output_Node = MajoranaPropagation(trunc_param, Init_Node, len(U), U)
        toc = time.perf_counter()
        print(f"Majorana Propagation : {toc - tic:0.4f} seconds")

        # Rotated Majorana Propogation
        #tic = time.perf_counter()
        Node_out = twofourMajStrEvo(nf, h, V, n, dt, Init_Node, trunc_param)
        #toc = time.perf_counter()
        #print(f"Rotated Evolution : {toc - tic:0.4f} seconds")

        for node in Output_Node:
            #print(sum(node.b))
            pass


        obexp = np.zeros(2**nf, dtype = complex)
        rexp = np.zeros(2**nf, dtype = complex)
        dexp = np.zeros(2**nf, dtype = complex)

        # transform into binary representation |n> = |n_1 n_2 n_3 \cdots n_{nf} >
        for i in range(2**nf):
            rem = i
            for j in range(nf):
                c[j] = rem / 2**(nf - j - 1)
                rem = rem - c[j] * 2**(nf - j - 1)

            for j in range(nf):
                rho_st[j] = c[j]
            #print("input Fock state = ", rho_st)
            
            
            obexp[i] = ExpectVal(Output_Node, len(Output_Node) , rho_st)
            #print("Expectation value by Majorana Propagation = ", obexp[i])

            rexp[i] = Rotated_ExpectVal(Node_out , h, dt, n, rho_st, nf)
            #print("Expectation value by Rotated Majorana Propagation = ", rexp[i])

            dexp[i] = DirectExp(init_len, init_maj, V, h, n * dt, i, nf, nf2)
        


    
        #Trotterization(dexp, init_len, init_maj, U, nf)

        # 2-norm 
        eps = 1e-15
        err[s] = 0.0 if np.linalg.norm(dexp) < eps else np.linalg.norm(obexp - dexp)/np.linalg.norm(dexp)
        err_rot[s] = 0.0 if np.linalg.norm(dexp) < eps else np.linalg.norm(rexp - dexp)/np.linalg.norm(dexp)
        ErrorPrint(dexp, obexp, rexp, 2)


err_avg = err.mean()
#sample std
err_std = err.std(ddof=1)
err_avg_r = err_rot.mean()
err_std_r = err_rot.std(ddof=1)


print("avg error = ", err_avg)
print("std = ", err_std)
print("avg error (rot)= ", err_avg_r)
print("std (rot) = ", err_std_r)

# truncation method + dt + n + init_maj_len + nonzero_term_h + nonzero+term_v + note(option) 
trunc_met = ""
dir_ = f"plot{date.today():%m%d}/"
note = "_rd_test"
filename =  dir_ + trunc_met + "_" + str(nf) + "_" + str(dt) + "_" + str(n) + "_" + str(init_len)+ "_" + str(np.count_nonzero(h)) + "_" + str(np.count_nonzero(V)) + note  + ".png"



