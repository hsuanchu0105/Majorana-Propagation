from MajProp import * 
from Setting import *
import time 




U = []

#sps = sparsity(h_ind, v_ind, nf2)
#print("Delta = ", sps)

#print(np.allclose(len(U), (n+1) * np.count_nonzero(h) + 2 * n * np.count_nonzero(V)))



ts_st = 1
mp = np.zeros(n - ts_st)
rmp = np.zeros(n - ts_st)
ana = np.zeros(n - ts_st)
rel_mp = np.zeros(n - ts_st)
rel_rmp = np.zeros(n - ts_st)


for ts in range(1, n):
    

    trunc_param = np.array([4, 1e-6])

    AppendH(U, h_ind, dt, 2, nf2)

    AppendV(U, v_ind, dt, 2, nf2)

    for i in range(ts-1):
        AppendH(U, h_ind, 2 * dt, 2, nf2)
        AppendV(U, v_ind, dt, 2, nf2)

    AppendH(U, h_ind, dt, 2, nf2)
        
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

        rexp[i] = Rotated_ExpectVal(Node_out , h, dt, ts, rho_st, nf)
        #print("Expectation value by Rotated Majorana Propagation = ", rexp[i])

        dexp[i] = DirectExp(init_len, init_maj, V, h, ts * dt, i, nf, nf2)
        


    
    #Trotterization(dexp, init_len, init_maj, U, nf)

    # 2-norm 
    mp[ts - ts_st] = np.linalg.norm(obexp)
    rmp[ts - ts_st] = np.linalg.norm(rexp)
    ana[ts - ts_st] = np.linalg.norm(dexp)
    rel_mp[ts - ts_st] = np.linalg.norm(obexp - dexp)/np.linalg.norm(dexp)
    rel_rmp[ts - ts_st] = np.linalg.norm(rexp - dexp)/np.linalg.norm(dexp)
    
    ErrorPrint(dexp, obexp, rexp, 2)


# truncation method + dt + n + init_maj_len + nonzero_term_h + nonzero+term_v + note(option) 
trunc_met = "ct"
dir = "plot0212/"
note = ""
filename =  dir + trunc_met + "_" + str(nf) + "_" + str(dt) + "_" + str(n) + "_" + str(init_len)+ "_" + str(np.count_nonzero(h)) + "_" + str(np.count_nonzero(V)) + note  + ".png"


#'''
ts_len = np.arange(ts_st, n)  
plt.figure()
plt.plot(ts_len, rel_mp , marker='o', linestyle='-', label='MP')
plt.plot(ts_len, rel_rmp , marker='o', linestyle='-', label='RMP')



#plt.xlabel('truncation length')
plt.xlabel('timestep')
plt.ylabel('relative error (global)')
#plt.ylabel(r'$\left\|O(t)\right\|_{2}$')
plt.yscale('log')  
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()


os.makedirs(os.path.dirname(filename), exist_ok=True)
plt.savefig(filename, dpi=200, bbox_inches="tight")
plt.show()


trunc_met = "lt"
dir = "plot0212/"
note = "_td4"
filename =  dir + trunc_met + "_" + str(nf) + "_" + str(dt) + "_" + str(n) + "_" + str(init_len)+ "_" + str(np.count_nonzero(h)) + "_" + str(np.count_nonzero(V)) + note  + ".png"

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
#'''