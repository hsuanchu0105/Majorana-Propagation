from src.MajProp import * 
from Setting.Setting_rd import *
from src.Gate import *
from src.Err_anlys import *
from src.RMP import *
from src.TE import *
import time 
from datetime import date
from datetime import datetime 
import sys
import os
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt



dir_ = f"analysis/ta{date.today():%m%d}/"
note = "rd_ana"

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path =  dir_  + str(len_trunc) + "_" + str(coeff_trunc) + "_" + str(nf) + "_" + str(dt) + "_" + str(n) + "_" + str(init_len) + "_" + ts + "_" + note  + ".txt"

os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

log_file = open(log_path, "w", encoding="utf-8")

def log(*args, **kwargs):
    # show on console
    print(*args, **kwargs)
    # also write to file
    print(*args, **kwargs, file=log_file)
    log_file.flush()


nz_seed = 716732 # seed for non-zero terms 
cf_seed = 126338 # seed for coefficients


nmin_h=1
nmax_h=8
nmin_v=1
nmax_v=8

#sps = sparsity(h_ind, v_ind, nf2)
#print("Delta = ", sps)

#print(np.allclose(len(U), (n+1) * np.count_nonzero(h) + 2 * n * np.count_nonzero(V)))

err = np.zeros((cn, sn))
err_rot = np.zeros((cn, sn))


alpha_h = np.zeros(cn, dtype = int)
alpha_v = np.zeros(cn, dtype = int)

nz_rng = np.random.default_rng(nz_seed)  
nz_seeds = nz_rng.integers(0, 2**32 - 1, size=cn, dtype=np.uint32) # choose cn random integers

log("seed for initial terms = ", seed_init, '\t')
log("seed for non-zero terms = ", nz_seed, '\t')
log("seed for coefficients =  ", cf_seed, '\t')


for cs in range(cn):
    # for each case we have one (alpha_h, alpha_v) pair
    
    rng_nzc = np.random.default_rng(int(nz_seeds[cs]))  # non-zero count 
    # here alpha_h only consider upper triangle terms 
    alpha_h[cs] = rng_nzc.integers(nmin_h, nmax_h + 1)
    alpha_v[cs] = rng_nzc.integers(nmin_v, nmax_v + 1)

    

    #log_file = open(log_path, "w", encoding="utf-8")
    #sys.stdout = Tee(sys.__stdout__, log_file)
    #sys.stderr = Tee(sys.__stderr__, log_file)

    print("alpha_h, alpha_v = ", alpha_h[cs], alpha_v[cs])
    print('\t')

    cf_rng = np.random.default_rng(cf_seed)  # pick any fixed master seed
    cf_seeds = cf_rng.integers(0, 2**32 - 1, size=sn, dtype=np.uint32)

    


    for s in range(sn):
        
        pairs_h, h = random_sparse_h(alpha_h[cs], nf2, complex_coeff=False, seed=int(cf_seeds[s]))
        pairs_v, V = random_sparse_v(alpha_v[cs], nf2, complex_coeff=False, seed=int(cf_seeds[s]))


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
        Node_out = twofourMajStrEvo(nf, h, V, n, dt, Init_Node, trunc_param, trott, hist_save)
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

    


#log('\t')
#log("relative error = ", err)
#log("relative error (rot) = ", err_rot)
#log('\t')
print('\t')
print("relative error = ", err)
print("relative error (rot) = ", err_rot)
print('\t')

dir_ = f"analysis/ta{date.today():%m%d}/" 
prefix = str(seed_init) + "_" +  str(nz_seed) + "_" + str(cf_seed)
np.savez(dir_ + prefix + "_errors.npz", err=err, err_rot=err_rot)


# truncation method + dt + n + init_maj_len + nonzero_term_h + nonzero+term_v + note(option) 

dir_ = f"analysis/plot{date.today():%m%d}/"
note = "_rd"
filename =  dir_  + str(len_trunc) + "_" + str(coeff_trunc) + "_" + str(nf) + "_" + str(dt) + "_" + str(n) + "_" + str(init_len)+ "_" + str(alpha_h) + "_" + str(alpha_v) + "_" + ts + "_" + note  + ".png"



log_err = np.log(err)  
mu = log_err.mean(axis=1)  # ln(\mu_g)
sigma = log_err.std(axis=1)  # ln(\sigma_g)
# geometric mean 
mean_geo_mp = np.exp(mu) 
low = np.exp(mu-sigma) 
high = np.exp(mu+sigma) 
yerr_mp = np.vstack([mean_geo_mp - low, high - mean_geo_mp])


log_err_r = np.log(err_rot)
mu2 = log_err_r.mean(axis=1) 
sigma2 = log_err_r.std(axis=1) 
mean_geo_r = np.exp(mu2)
low2 = np.exp(mu2-sigma2) 
high2 = np.exp(mu2+sigma2) 
yerr_r = np.vstack([mean_geo_r - low2, high2 - mean_geo_r])


w = 0.075
x = np.arange(1, cn + 1)

plt.figure()
plt.errorbar(x - w, mean_geo_mp, yerr=yerr_mp, fmt='o', capsize=3, label='MP')
plt.errorbar(x + w, mean_geo_r,  yerr=yerr_r,  fmt='o', capsize=3, label='RMP')

plt.yscale('log')
plt.xticks(x)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Case')
plt.ylabel('Relative error')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.title("Geometric mean and standard deviation for random coefficients")

os.makedirs(os.path.dirname(filename), exist_ok=True)
plt.savefig(filename, dpi=200, bbox_inches="tight")
plt.show()

