from MajProp import * 
from Setting_ta import * 
from datetime import date
import time 
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


coeff_trunc = 1e-8


trunc_met = "lt"
dir_ = f"ta{date.today():%m%d}/"
note = ""
log_path =  dir_ + trunc_met + "_" + str(coeff_trunc) + "_" + str(nf) + "_" + str(dt) + "_" + str(n) + "_" + str(init_len)+ "_" + str(np.count_nonzero(h)) + "_" + str(np.count_nonzero(V)) + note  + ".txt"

os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

log_file = open(log_path, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)




U = []

#sps = sparsity(h_ind, v_ind, nf2)
#print("Delta = ", sps)

# the comparison can be written into more efficient way
AppendH(U, h_ind, dt, trott, nf2)

AppendV(U, v_ind, dt, trott, nf2)

for i in range(n-1):
    AppendH(U, h_ind, 2 * dt, trott, nf2)
    AppendV(U, v_ind, dt, trott, nf2)

AppendH(U, h_ind, dt, trott, nf2)
    

print("Fermionic mode = ", nf)
print("dt = ", dt)
print("n = ", n)
print("alpha_h = ", len_h)
print("alpha_v = ", len_v) 

print("gate count", len(U))
print("Coeff trunc = ", coeff_trunc)

#print(np.allclose(len(U), (n+1) * np.count_nonzero(h) + 2 * n * np.count_nonzero(V)))

tc_st = 2
tc_end = 9


tic0 = time.perf_counter()
for tc_len in range(tc_st, tc_end, 2):

    print('\t')
    print("Length Truncation at ", tc_len)
    trunc_param = np.array([tc_len, coeff_trunc])


    #print("Fermionic gate:", U)
    #print('\t')
    rho_st = np.zeros(nf, dtype = int)
    c = np.zeros(nf, dtype = int)

    tic1 = time.perf_counter()
    Output_Node = MajoranaPropagation(trunc_param, Init_Node, len(U), U)
    toc1 = time.perf_counter()
    print(f"Majorana Propagation : {toc1 - tic1:0.4f} seconds")

    # Rotated Majorana Propogation
    tic2 = time.perf_counter()
    Node_out = twofourMajStrEvo(nf, h, V, n, dt, Init_Node, trunc_param, trott)
    toc2 = time.perf_counter()
    print(f"Rotated Evolution : {toc2 - tic2:0.4f} seconds")

    


    obexp = 0
    rexp = 0
    dexp = 0

    # transform into binary representation |n> = |n_1 n_2 n_3 \cdots n_{nf} >
    
    rem = nf
    for j in range(nf):
        c[j] = rem / 2**(nf - j - 1)
        rem = rem - c[j] * 2**(nf - j - 1)

    for j in range(nf):
        rho_st[j] = c[j]
    #print("input Fock state = ", rho_st)
    
    tic3 = time.perf_counter()
    obexp = ExpectVal(Output_Node, len(Output_Node) , rho_st)
    toc3 = time.perf_counter()
    print(f"Expectation calculation : {toc3 - tic3:0.4f} seconds")
    #print("Expectation value by Majorana Propagation = ", obexp[i])

    tic4 = time.perf_counter()
    rexp = Rotated_ExpectVal(Node_out , h, dt, n, rho_st, nf)
    toc4 = time.perf_counter()
    print(f"Rotated expectation calculation : {toc4 - tic4:0.4f} seconds")
    #print("Expectation value by Rotated Majorana Propagation = ", rexp[i])

    tic5 = time.perf_counter()
    dexp = DirectExp(init_len, init_maj, V, h, n * dt, i, nf, nf2)
    toc5 = time.perf_counter()
    print(f"Direct Exponential : {toc5 - tic5:0.4f} seconds")
        
        

    
    #Trotterization(dexp, init_len, init_maj, U, nf)

toc0 = time.perf_counter()
print(f"Total time : {toc0 - tic0:0.4f} seconds")








log_file.close()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
