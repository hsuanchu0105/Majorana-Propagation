from MajProp import * 
import time 

N = 3
dt = 0.1
n = 3 #num of timestep


h = np.zeros((2*N, 2*N))

h[2][0] = -1

for i in range(nf2):
    for j in range(i+1, 2*N):
        h[i][j] = -h[j][i]

print(h)



V = np.zeros((2*N, 2*N, 2*N, 2*N))
V[0][2][4][5] = 1
V[1][3][4][5] = 1

init_bin = np.array([1, 1, 0, 0, 0, 0])
M1= MajoranaOp(6, init_bin)
N1 = Node(init_bin, 1j**M1.rb())

Init_Node = [N1]
init_len = 1
init_maj = [M1]

h_ind = np.nonzero(h)
v_ind = np.nonzero(V)
nm = len(h_ind[0])
nmv = len(v_ind[0])


U = []

hcnt = 0
theta1 = dt 
for i in range(nm):
        b1 = np.zeros(2 * N)
        if(h_ind[0][i] < h_ind[1][i]):
            b1[h_ind[0][i]] = 1
            b1[h_ind[1][i]] = 1
            U.append([theta1, b1])
            hcnt += 1

cur_pos = len(U)
for i in range(cur_pos-1, -1, -1):
    U.append(U[i])

 
print("free fermion gate count", 2 * hcnt)

theta2 = dt * 2

for i in range(nmv):
    b2 = np.zeros(2 * N)
    b2[v_ind[0][i]] = (b2[v_ind[0][i]]+ 1) % 2
    b2[v_ind[1][i]] = (b2[v_ind[1][i]]+ 1) % 2
    b2[v_ind[2][i]] = (b2[v_ind[2][i]]+ 1) % 2
    b2[v_ind[3][i]] = (b2[v_ind[3][i]]+ 1) % 2

    # parity check? 

    U.append([theta2, b2])

cur_pos = len(U)
for i in range(cur_pos-1, cur_pos- nmv - 1, -1):
    U.append(U[i])

print("after first V", len(U))

theta1 = dt * 2  #second order trotterization

for i in range(nm):
    b1 = np.zeros(2 * N)
    if(h_ind[0][i] < h_ind[1][i]):
        b1[h_ind[0][i]] = 1
        b1[h_ind[1][i]] = 1   
        U.append([theta1, b1])
        
cur_pos = len(U)
for i in range(cur_pos-1, cur_pos- hcnt - 1, -1):
    U.append(U[i])

print("after second U", len(U))

ly_end = len(U)
for ts in range(n-2):
    for i in range(2 * hcnt, ly_end):
        U.append(U[i])

theta2 = dt * 2

for i in range(nmv):
    b2 = np.zeros(2 * N)
    b2[v_ind[0][i]] = (b2[v_ind[0][i]]+ 1) % 2
    b2[v_ind[1][i]] = (b2[v_ind[1][i]]+ 1) % 2
    b2[v_ind[2][i]] = (b2[v_ind[2][i]]+ 1) % 2
    b2[v_ind[3][i]] = (b2[v_ind[3][i]]+ 1) % 2

    # parity check? 

    U.append([theta2, b2])

cur_pos = len(U)
for i in range(cur_pos-1, cur_pos- nmv - 1, -1):
    U.append(U[i])

for i in range(2 * hcnt):
    U.append(U[i])
    
print("gate count", len(U))
trunc_param = np.array([20, 1e-10])


#print("Fermionic gate:", U)
#print('\t')
rho_st = np.array([0, 0, 0])
c = np.array([0, 0, 0])

tic = time.perf_counter()
Output_Node = MajoranaPropagation(trunc_param, Init_Node, len(U), U)
toc = time.perf_counter()
print(f"Majorana Propagation : {toc - tic:0.4f} seconds")

for node in Output_Node:
    pass
    #print(node)
# transform into binary representation |n> = |n_1 n_2 n_3>
for i in range(8):
    c[0] = i/4
    r1 = i - c[0] * 4
    c[1] = r1/2
    r2 = r1 - c[1] * 2
    c[2] = r2
    
    for j in range(nf):
        rho_st[j] = c[j]
    #print("input Fock state = ", rho_st)
    
    
    Exp_val = ExpectVal(Output_Node, len(Output_Node) , rho_st)
    print("Expectation value by Majorana Propagation = ", Exp_val)


DirectCal(init_len, init_maj, U)



tic = time.perf_counter()
Node_out = twofourMajStrEvo(N, h, V, n, dt, Init_Node, trunc_param)
toc = time.perf_counter()
print(f"Rotated Evolution : {toc - tic:0.4f} seconds")

for node in Node_out:
    pass
    #print(node)

rho_st = np.array([0, 0, 0])
c = np.array([0, 0, 0])
for i in range(8):
    c[0] = i/4
    r1 = i - c[0] * 4
    c[1] = r1/2
    r2 = r1 - c[1] * 2
    c[2] = r2
    
    for j in range(nf):
        rho_st[j] = c[j]
    #print("input Fock state = ", rho_st)
    
    tic = time.perf_counter()
    rexp = Rotated_ExpectVal(Node_out , h, dt, n, rho_st)
    toc = time.perf_counter()
    print(f"Calculate Rotated Expectation value in {toc - tic:0.4f} seconds")
    print("Expectation value by Rotated Majorana Propagation = ", rexp)
