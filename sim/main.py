from MajProp import * 



a2 = np.array([1, 1, 0, 0, 0, 0])
M2 = MajoranaOp(6, a2)
N2 = Node(a2, 1j**M2.rb())

Init_Node = [N2]
init_len = 1
init_maj = [M2]



b1 = np.array([1, 0, 1, 0, 0, 0])
b2 = np.array([1, 0, 1, 0, 1, 1])
theta1 = 0.4
theta2= 0.2
U = [[theta1, b1], [theta2, b2]]

trunc_param = np.array([20, 1e-10])


#print("Fermionic gate:", U)
#print('\t')
rho_st = np.array([0, 0, 0])
c = np.array([0, 0, 0])

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
    Output_Node = MajoranaPropagation(trunc_param, Init_Node, len(U), U)
    Exp_val = ExpectVal(Output_Node, len(Output_Node) , rho_st)
    print("Expectation value by Majorana Propagation = ", Exp_val)




N = 3
h = np.zeros((2*N, 2*N))

h[2][0] = -1

for i in range(nf2):
    for j in range(i+1, 2*N):
        h[i][j] = -h[j][i]


print(h)
dt = 0.1
n = 1
V = np.zeros((2*N, 2*N, 2*N, 2*N))

V[0][2][4][5] = 1


rho_st = np.zeros(3)
Node_out = twofourMajStrEvo(N, h, V, n, dt, Init_Node, trunc_param)
for node in Node_out:
    print(node)

Exp_val = Rotated_ExpectVal(Node_out , h, rho_st)
#print("Expectation value by Majorana Propagation = ", Exp_val)