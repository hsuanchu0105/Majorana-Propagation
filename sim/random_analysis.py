from MajProp import * 
from Setting_rd import *
import time 
from datetime import date





#sps = sparsity(h_ind, v_ind, nf2)
#print("Delta = ", sps)

#print(np.allclose(len(U), (n+1) * np.count_nonzero(h) + 2 * n * np.count_nonzero(V)))






U = []

AppendH(U, h_ind, dt, trott, nf2)

AppendV(U, v_ind, dt, trott, nf2)

for i in range(n-1):
    AppendH(U, h_ind, 2 * dt, trott, nf2)
    AppendV(U, v_ind, dt, trott, nf2)

AppendH(U, h_ind, dt, trott, nf2)
    
print("gate count", len(U))

err_avg = 0
err_std = 0
err = np.zeros(sn)

for sam in range(sn):
    

    trunc_param = np.array([6, 1e-8])
    
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
    err[sam] = np.linalg.norm(obexp - dexp)
    
    ErrorPrint(dexp, obexp, rexp, 2)


err_avg = err.mean()
#sample std
err_std = err.std(ddof=1)

print("avg error = ", err_avg)
print("std = ", err_std)

# truncation method + dt + n + init_maj_len + nonzero_term_h + nonzero+term_v + note(option) 
trunc_met = ""
dir_ = f"plot{date.today():%m%d}/"
note = "_rd_test"
filename =  dir_ + trunc_met + "_" + str(nf) + "_" + str(dt) + "_" + str(n) + "_" + str(init_len)+ "_" + str(np.count_nonzero(h)) + "_" + str(np.count_nonzero(V)) + note  + ".png"



