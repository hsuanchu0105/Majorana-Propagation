from MajProp import * 
import time 





dt = 0.01
n = 10 #num of timestep

#number of fermionic mode 
nf = 3
nf2 = 2 * nf


h = np.zeros((nf2, nf2))

h[2][0] = -0.1 # introduce some randomness here (different values or so)
h[1][0] = -0.1
h[3][1] = -0.2
#h[3][2] = -0.2
#h[5][1]= -0.1


for i in range(nf2):
    for j in range(i+1, nf2):
        h[i][j] = -h[j][i]

#print(h)



V = np.zeros((nf2, nf2, nf2, nf2))
#V[0][1][2][3] = 0.5
#V[1][2][3][4] = 0.5
#V[2][3][4][5] = 0.5
#V[1][3][4][5] = 0.1
#V[0][1][3][4] = -0.3
#V[0][1][4][6] = 0.1
#V[0][1][6][7] = 0.1

#init_bin = np.array([1, 1, 0, 0, 0, 0, 1, 1])
init_bin = np.array([1, 1, 0, 0, 1, 1])
M1= MajoranaOp(nf2, init_bin)
N1 = Node(init_bin, 1j**M1.rb())

#bin2 = np.array([1, 1, 1, 1, 1, 1, 1, 1])
bin2 = np.array([0, 0, 1, 1, 0, 0])
M2= MajoranaOp(nf2, bin2)
N2 = Node(bin2, 1j**M2.rb())

#bin3 = np.array([1, 0, 1, 0, 1, 0, 1, 0])
bin3 = np.array([0, 0, 1, 1, 1, 1])
M3= MajoranaOp(nf2, bin3)
N3 = Node(bin3, 1j**M3.rb())

#bin4 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
#M4= MajoranaOp(nf2, bin4)
#N4 = Node(bin4, 1j**M4.rb())

Init_Node = [N1, N2, N3]
init_len = len(Init_Node)
init_maj = [M1, M2, M3]

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
print(np.allclose(len(U), (n+1) * np.count_nonzero(h) + 2 * n * np.count_nonzero(V)))

tc_st = 2
tc_end = 7
rel_mp_global = np.zeros(tc_end - tc_st)
rel_rot_global = np.zeros(tc_end - tc_st)

#tc_len = 5
#coef_st = -2
#coef_end = -10
#rel_mp_global = np.zeros(coef_st - coef_end)
#rel_rot_global = np.zeros(coef_st - coef_end)

for tc_len in range(tc_st, tc_end):
#for coef_tr in range(coef_st, coef_end, -1):
    trunc_param = np.array([tc_len, 1e-6])
    #trunc_param = np.array([tc_len, 10**(coef_tr)])

    #trunc_param = np.array([4, 1e-6])

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


    #DirectExp(dexp, init_len, init_maj, V, h, n * dt, nf, nf2)
    Trotterization(dexp, init_len, init_maj, U, nf)

    # 2-norm 
    #eps = 1e-15
    rel_mp_global[tc_len - tc_st] = np.linalg.norm(obexp - dexp) / np.linalg.norm(dexp)
    rel_rot_global[tc_len - tc_st] = np.linalg.norm(rexp - dexp) / np.linalg.norm(dexp)
    #rel_mp_global[coef_tr - coef_st] = np.linalg.norm(obexp - dexp) / max(np.linalg.norm(dexp), eps)
    #rel_rot_global[coef_tr - coef_st] = np.linalg.norm(rexp - dexp) / max(np.linalg.norm(dexp), eps)
    ErrorPrint(dexp, obexp, rexp, 2)


# truncation method + dt + n + init_maj_len + nonzero_term_h + nonzero+term_v + note(option) 
trunc_met = "lt"
dir = "plot0211/"
note = "_trott"
filename =  dir + trunc_met + "_" + str(nf) + "_" + str(dt) + "_" + str(n) + "_" + str(init_len)+ "_" + str(np.count_nonzero(h)) + "_" + str(np.count_nonzero(V)) + note  + ".png"

#filename = "N=6test.png"
lentrunc_plot(tc_st, tc_end, rel_mp_global, rel_rot_global, filename)

