from MajProp import * 
import time 

dt = 0.1
n = 1 #num of timestep


h = np.zeros((nf2, nf2), dtype = float)

pairs = [(i, (i + 1) % sn) for i in range(sn)]

for i, j in pairs:
    i += 1 # convertion that site number starts from 1 
    j += 1
    # spin up
    h[4 * i - 4][4 * j - 2] = -1/2
    h[4 * i - 2][4 * j - 4] = 1/2
    # spin down 
    h[4 * i - 3][4 * j - 1] = -1/2
    h[4 * i - 1][4 * j - 3] = 1/2


print(np.allclose(h, -h.T, atol=1e-12, rtol=0))
# check antisymmetry


V = np.zeros((nf2, nf2, nf2, nf2))


init_bin = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
M1= MajoranaOp(len(init_bin), init_bin)
N1 = Node(init_bin, 1j**M1.rb())

Init_Node = [N1]
init_len = 1
init_maj = [M1]

h_ind = np.nonzero(h)
v_ind = np.nonzero(V)

print(h_ind)

U = []



AppendH(U, h_ind, dt, 2)


#AppendV(U, v_ind, dt, 2)
 
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