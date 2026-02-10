from MajProp import * 
import time 





dt = 0.01
n = 10 #num of timestep

#number of fermionic mode 
nf = 3
nf2 = 2 * nf


h = np.zeros((nf2, nf2))

h[2][0] = -0.1 # introduce some randomness here (different values or so)
#h[1][0] = -1
#h[3][1]= -1


for i in range(nf2):
    for j in range(i+1, nf2):
        h[i][j] = -h[j][i]

#print(h)



V = np.zeros((nf2, nf2, nf2, nf2))
V[0][2][4][5] = 0.8
V[1][3][4][5] = 1

init_bin = np.array([1, 1, 0, 0, 0, 0])
M1= MajoranaOp(6, init_bin)
N1 = Node(init_bin, 1j**M1.rb())

Init_Node = [N1]
init_len = 1
init_maj = [M1]

h_ind = np.nonzero(h)
v_ind = np.nonzero(V)


U = []

sps = sparsity(h_ind, v_ind, nf2)
print("Delta = ", sps)

AppendH(U, h_ind, dt, 2, nf2)

AppendV(U, v_ind, dt, 2, nf2)
 
for i in range(n-1):
    AppendH(U, h_ind, 2 * dt, 2, nf2)
    AppendV(U, v_ind, dt, 2, nf2)

AppendH(U, h_ind, dt, 2, nf2)
    
print("gate count", len(U))

#tc_st = 3
#tc_end = 7
#rel_mp_global = np.zeros(tc_end - tc_st)
#rel_rot_global = np.zeros(tc_end - tc_st)

#tc_len = 5
#coef_st = -2
#coef_end = -10
#rel_mp_global = np.zeros(coef_st - coef_end)
#rel_rot_global = np.zeros(coef_st - coef_end)

#for tc_len in range(tc_st, tc_end):
#for coef_tr in range(coef_st, coef_end, -1):
#trunc_param = np.array([tc_len, 1e-6])
#trunc_param = np.array([tc_len, 10**(coef_tr)])

trunc_param = np.array([4, 1e-6])

#print("Fermionic gate:", U)
#print('\t')
rho_st = np.array([0, 0, 0])
c = np.array([0, 0, 0])

tic = time.perf_counter()
Output_Node = MajoranaPropagation(trunc_param, Init_Node, len(U), U)
toc = time.perf_counter()
print(f"Majorana Propagation : {toc - tic:0.4f} seconds")

# Rotated Majorana Propogation
#tic = time.perf_counter()
Node_out = twofourMajStrEvo(nf, h, V, n, dt, Init_Node, trunc_param)
#toc = time.perf_counter()
#print(f"Rotated Evolution : {toc - tic:0.4f} seconds")

#for node in Output_Node:
    #print(node)




#'''
# transform into binary representation |n> = |n_1 n_2 n_3>
obexp = np.zeros(2**nf, dtype = complex)
rexp = np.zeros(2**nf, dtype = complex)
dexp = np.zeros(2**nf, dtype = complex)

for i in range(2**nf):
    c[0] = i/4
    r1 = i - c[0] * 4
    c[1] = r1/2
    r2 = r1 - c[1] * 2
    c[2] = r2
    
    for j in range(nf):
        rho_st[j] = c[j]
    #print("input Fock state = ", rho_st)
    
    
    obexp[i] = ExpectVal(Output_Node, len(Output_Node) , rho_st)
    #print("Expectation value by Majorana Propagation = ", obexp[i])

    rexp[i] = Rotated_ExpectVal(Node_out , h, dt, n, rho_st, nf)
    #print("Expectation value by Rotated Majorana Propagation = ", rexp[i])


DirectExp(dexp, init_len, init_maj, V, h, n * dt, nf, nf2)

#2-norm 
#eps = 1e-15
#rel_mp_global[tc_len - tc_st] = np.linalg.norm(obexp - dexp) / max(np.linalg.norm(dexp), eps)
#rel_rot_global[tc_len - tc_st] = np.linalg.norm(rexp - dexp) / max(np.linalg.norm(dexp), eps)
#rel_mp_global[coef_tr - coef_st] = np.linalg.norm(obexp - dexp) / max(np.linalg.norm(dexp), eps)
#rel_rot_global[coef_tr - coef_st] = np.linalg.norm(rexp - dexp) / max(np.linalg.norm(dexp), eps)

    



print('\t')
print("-----------------Direct exponential---------------")
print(dexp)
print("-----------------Majorana Propagation-------------")
print(obexp)
print("------------Rotated Majorana Propagation----------")
print(rexp)



eps = 1e-15
rel_maj = np.abs(obexp - dexp) / np.abs(dexp)
rel_rotm = np.abs(rexp - dexp) / np.abs(dexp)

print('\t')
print("Relative error Majorana Propagation")
print(rel_maj)

print("Relative error rotated Majorana")
print(rel_rotm)

#2-norm 
rel_ob_global = np.linalg.norm(obexp - dexp) / np.linalg.norm(dexp)
rel_re_global = np.linalg.norm(rexp - dexp) / np.linalg.norm(dexp)

print('\t')
print("Global relative error")
print(rel_ob_global, rel_re_global)
#'''