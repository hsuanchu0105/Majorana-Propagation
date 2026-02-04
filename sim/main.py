from MajProp import * 
import time 


dt = 0.1
n = 3 #num of timestep


h = np.zeros((nf2, nf2))

h[2][0] = -1
h[1][0] = -1
h[3][1]= -1


for i in range(nf2):
    for j in range(i+1, nf2):
        h[i][j] = -h[j][i]

print(h)



V = np.zeros((nf2, nf2, nf2, nf2))
#V[0][2][4][5] = 1
#V[1][3][4][5] = 1

init_bin = np.array([1, 1, 0, 0, 0, 0])
M1= MajoranaOp(6, init_bin)
N1 = Node(init_bin, 1j**M1.rb())

Init_Node = [N1]
init_len = 1
init_maj = [M1]

h_ind = np.nonzero(h)
v_ind = np.nonzero(V)



U = []



AppendH(U, h_ind, dt, 2)

AppendV(U, v_ind, dt, 2)
 
for i in range(n-1):
    AppendH(U, h_ind, 2 * dt, 2)
    AppendV(U, v_ind, dt, 2)

AppendH(U, h_ind, dt, 2)
    
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
DirectExp(init_len, init_maj,V,h,n * dt)


tic = time.perf_counter()
Node_out = twofourMajStrEvo(nf, h, V, n, dt, Init_Node, trunc_param)
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
