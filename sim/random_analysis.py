from MajProp import * 
from Setting_rd import *
import time 
from datetime import date
from datetime import datetime 
import sys
import os
from matplotlib.ticker import MaxNLocator

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
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
note = "rd_ana"

#nz_seed = 21
mst_seed = 123
cs_seed = 15553

nmin_h=1
nmax_h=3
nmin_v=1
nmax_v=3

#sps = sparsity(h_ind, v_ind, nf2)
#print("Delta = ", sps)

#print(np.allclose(len(U), (n+1) * np.count_nonzero(h) + 2 * n * np.count_nonzero(V)))

err = np.zeros((cn, sn))
err_rot = np.zeros((cn, sn))
err_avg = np.zeros(cn)
err_std = np.zeros(cn)
err_avg_r = np.zeros(cn)
err_std_r = np.zeros(cn)

alpha_h = np.zeros(cn, dtype = int)
alpha_v = np.zeros(cn, dtype = int)

mcs_rng = np.random.default_rng(cs_seed)  # pick any fixed master seed
cs_seeds = mcs_rng.integers(0, 2**32 - 1, size=cn, dtype=np.uint32)

for cs in range(cn):
    # for each case we have one (alpha_h, alpha_v) pair
    
    rng_nzc = np.random.default_rng(int(cs_seeds[cs]))  # non-zero count 
    # here alpha_h only consider upper triangle terms 
    alpha_h[cs] = rng_nzc.integers(nmin_h, nmax_h + 1)
    alpha_v[cs] = rng_nzc.integers(nmin_v, nmax_v + 1)

    log_path =  dir_  + str(len_trunc) + "_" + str(coeff_trunc) + "_" + str(nf) + "_" + str(dt) + "_" + str(n) + "_" + str(init_len)+ "_" + str(alpha_h[cs]) + "_" + str(alpha_v[cs]) + "_" + ts + "_" + note  + ".txt"

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print("alpha_h, alpha_v = ", alpha_h[cs], alpha_v[cs])
    print('\t')

    master_rng = np.random.default_rng(mst_seed)  # pick any fixed master seed
    seeds = master_rng.integers(0, 2**32 - 1, size=sn, dtype=np.uint32)

    
    print("seed for nonzero terms = ", cs_seeds)
    print("master seed =  ", mst_seed)
    print("seed for coefficient generation:", seeds)
    print('\t')


    for s in range(sn):
        
        pairs_h, h = random_sparse_h(alpha_h[cs], nf2, complex_coeff=False, seed=int(seeds[s]))
        pairs_v, V = random_sparse_v(alpha_v[cs], nf2, complex_coeff=False, seed=int(seeds[s]))


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
        tic = time.perf_counter()
        Node_out = twofourMajStrEvo(nf, h, V, n, dt, Init_Node, trunc_param)
        toc = time.perf_counter()
        print(f"Rotated Evolution : {toc - tic:0.4f} seconds")

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
        err[cs, s] = 0.0 if np.linalg.norm(dexp) < eps else np.linalg.norm(obexp - dexp)/np.linalg.norm(dexp)
        err_rot[cs, s] = 0.0 if np.linalg.norm(dexp) < eps else np.linalg.norm(rexp - dexp)/np.linalg.norm(dexp)
        ErrorPrint(dexp, obexp, rexp, 2)

    

    err_avg[cs] = err[cs].mean()
    #sample std
    err_std[cs] = err[cs].std(ddof=1)
    err_avg_r[cs] = err_rot[cs].mean()
    err_std_r[cs] = err_rot[cs].std(ddof=1)

print('\t')
print("relative error = ", err)
print("relative erorr (rot) = ", err_rot)
print('\t')

print("avg error = ", err_avg)
print("std = ", err_std)
print("avg error (rot)= ", err_avg_r)
print("std (rot) = ", err_std_r)

# truncation method + dt + n + init_maj_len + nonzero_term_h + nonzero+term_v + note(option) 

dir_ = f"plot{date.today():%m%d}/"
note = "_rd"
filename =  dir_  + str(len_trunc) + "_" + str(coeff_trunc) + "_" + str(nf) + "_" + str(dt) + "_" + str(n) + "_" + str(init_len)+ "_" + str(alpha_h) + "_" + str(alpha_v) + "_" + ts + "_" + note  + ".png"




x = np.arange(1, cn+1)          
w = 0.075


plt.figure()
plt.errorbar(x-w, err_avg, yerr=err_std, fmt='o', capsize=3, label='MP')
plt.errorbar(x+w, err_avg_r, yerr=err_std_r, fmt='o', capsize=3, label='RMP')
plt.xticks(x)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))


plt.xlabel('index')
plt.ylabel('relative error')
plt.yscale('log')  # optional
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()


os.makedirs(os.path.dirname(filename), exist_ok=True)
plt.savefig(filename, dpi=200, bbox_inches="tight")
plt.show()
