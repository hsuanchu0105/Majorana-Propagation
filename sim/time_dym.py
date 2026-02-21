from MajProp import * 
from Setting_td import *
import time 
from datetime import date
from datetime import datetime 
from pathlib import Path



#sps = sparsity(h_ind, v_ind, nf2)
#print("Delta = ", sps)

#print(np.allclose(len(U), (n+1) * np.count_nonzero(h) + 2 * n * np.count_nonzero(V)))



ts_st = 1
mp = np.zeros(n - ts_st)
rmp = np.zeros(n - ts_st)
ana = np.zeros(n - ts_st)
rel_mp = np.zeros(n - ts_st)
rel_rmp = np.zeros(n - ts_st)
mp_trott = np.zeros(n - ts_st)


for ts in range(n,n+1):
    

    trunc_param = np.array([len_trunc, coeff_trunc])
    U = []

    AppendH(U, h_ind, dt, trott, nf2)

    AppendV(U, v_ind, dt, trott, nf2)

    for i in range(ts-1):
        AppendH(U, h_ind, 2 * dt, trott, nf2)
        AppendV(U, v_ind, dt, trott, nf2)

    AppendH(U, h_ind, dt, trott, nf2)
        
    print("gate count", len(U))
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
    Node_out = twofourMajStrEvo(nf, h, V, n, dt, Init_Node, trunc_param, trott)
    toc = time.perf_counter()
    print(f"Rotated Evolution : {toc - tic:0.4f} seconds")

    for node in Output_Node:
        #print(sum(node.b))
        pass


    obexp = np.zeros(2**nf, dtype = complex)
    rexp = np.zeros(2**nf, dtype = complex)
    dexp = np.zeros(2**nf, dtype = complex)
    trottexp = np.zeros(2**nf, dtype = complex)

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

        rexp[i] = Rotated_ExpectVal(Node_out , h, dt, ts, rho_st, nf)
        #print("Expectation value by Rotated Majorana Propagation = ", rexp[i])

        dexp[i] = DirectExp(init_len, init_maj, V, h, ts * dt, i, nf, nf2)
        


    
    Trotterization(trottexp, init_len, init_maj, U, nf)

    # 2-norm 
    #mp[ts - ts_st] = np.linalg.norm(obexp)
    #rmp[ts - ts_st] = np.linalg.norm(rexp)
    #ana[ts - ts_st] = np.linalg.norm(dexp)
    #rel_mp[ts - ts_st] = np.linalg.norm(obexp - dexp)/np.linalg.norm(dexp)
    #rel_rmp[ts - ts_st] = np.linalg.norm(rexp - dexp)/np.linalg.norm(dexp)
    mp_trott[ts - ts_st] = np.linalg.norm(obexp - trottexp)
    
    #ErrorPrint(dexp, obexp, rexp, 2)

#'''
dir_ = f"ta{date.today():%m%d}/"
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# fermionic mode + dt + n + init_maj_len + nonzero_term_h + nonzero+term_v + len_trunc + coeff_trunc + trotter order + note(optional)
prefix = dir_ + str(nf) + "_" + str(dt) + "_" + str(n) + "_" + str(init_len)+ "_" + str(np.count_nonzero(h)) + "_" + str(np.count_nonzero(V)) + "_" + str(len_trunc) + "_" + str(coeff_trunc) + "_" + str(trott) 

fname =  prefix + ".csv"
path = Path(fname)   # any nested path is fine
path.parent.mkdir(parents=True, exist_ok=True)
#np.savetxt(path, mp_trott, delimiter=",")
#'''
dir_ = f"plot{date.today():%m%d}/"
prefix = dir_ + str(nf) + "_" + str(dt) + "_" + str(n) + "_" + str(init_len)+ "_" + str(np.count_nonzero(h)) + "_" + str(np.count_nonzero(V)) + "_" + str(len_trunc) + "_" + str(coeff_trunc) + "_" + str(trott) 
pltname =  prefix + ".png"


#'''
ts_len = np.arange(ts_st, n)  
plt.figure()
#plt.plot(ts_len, rel_mp , marker='o', linestyle='-', label='MP')
#plt.plot(ts_len, rel_rmp , marker='o', linestyle='-', label='RMP')
plt.plot(ts_len, mp_trott , marker='o', linestyle='-')



#plt.xlabel('truncation length')
plt.xlabel('timestep')
#plt.ylabel('relative error (global)')
#plt.ylabel(r'$\left\|O(t)\right\|_{2}$')
plt.ylabel(r'$\left\|Tr(\rho O^{\ell}_{\mathrm{MP}}) - Tr(\rho O_{\mathrm{trott}})\right\|_{2}$')
plt.yscale('log')  
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.title("Comparison between Majorana Propagation and Trotterization")
plt.tight_layout()


os.makedirs(os.path.dirname(pltname), exist_ok=True)
#plt.savefig(pltname, dpi=200, bbox_inches="tight")
plt.show()
'''

trunc_met = "lt"
dir = "plot0216/"
note = "_td4"
filename =  dir + trunc_met + "_" + str(nf) + "_" + str(dt) + "_" + str(n) + "_" + str(init_len)+ "_" + str(np.count_nonzero(h)) + "_" + str(np.count_nonzero(V)) + note  + ".png"

ts_len = np.arange(ts_st, n)
plt.figure()
plt.plot(ts_len, mp , marker='o', linestyle='-', label='MP')
plt.plot(ts_len, rmp , marker='o', linestyle='-', label='RMP')
plt.plot(ts_len, ana , marker='o', linestyle='-', label='ANA')
#plt.xlabel('truncation length')
plt.xlabel('timestep')
#plt.ylabel('relative error (global)')
plt.ylabel(r'$\left\|O(t)\right\|_{2}$')
plt.yscale('log')  
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()


os.makedirs(os.path.dirname(filename), exist_ok=True)
plt.savefig(filename, dpi=200, bbox_inches="tight")
plt.show()
'''