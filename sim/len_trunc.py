from MajProp import * 
from Setting import * 
import time 
from datetime import date

"""
Comparison of Majorana Propagation and Rotated Majorana Propagation at different truncation length
"""

U = []

#sps = sparsity(h_ind, v_ind, nf2)
#print("Delta = ", sps)

AppendH(U, h_ind, dt, trott, nf2)

AppendV(U, v_ind, dt, trott, nf2)

for i in range(n-1):
    AppendH(U, h_ind, 2 * dt, trott, nf2)
    AppendV(U, v_ind, dt, trott, nf2)

AppendH(U, h_ind, dt, trott, nf2)
    
print("gate count", len(U))

#print(np.allclose(len(U), (n+1) * np.count_nonzero(h) + 2 * n * np.count_nonzero(V)))

tc_st = 2
tc_end = 9
tc_len = tc_end - tc_st 
rel_mp_global = np.zeros(tc_len)
rel_rot_global = np.zeros(tc_len)


for tl in range(tc_st, tc_end):
    trunc_param = np.array([tl , 1e-6])


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
    
    rel_mp_global[tl - tc_st] = np.linalg.norm(obexp - dexp) / np.linalg.norm(dexp)
    rel_rot_global[tl - tc_st] = np.linalg.norm(rexp - dexp) / np.linalg.norm(dexp)
    
    
    ErrorPrint(dexp, obexp, rexp, 2)


# truncation method + dt + n + init_maj_len + nonzero_term_h + nonzero+term_v + note(option) 
trunc_met = "lt"
dir_ = f"plot{date.today():%m%d}/"
note = "_2"
filename =  dir_ + trunc_met + "_" + str(nf) + "_" + str(dt) + "_" + str(n) + "_" + str(init_len)+ "_" + str(np.count_nonzero(h)) + "_" + str(np.count_nonzero(V)) + note  + ".png"

tc_arr = np.arange(tc_st, tc_end)
lentrunc_plot(tc_arr, rel_mp_global, rel_rot_global, filename)

