from MajProp import * 
from itertools import combinations


# initial Majorana 
init_bin = np.array([1, 1, 1, 1, 0, 0])
M = MajoranaOp(6, init_bin)
N = Node(init_bin, 1j**M.rb())

Init_Node = [N]
init_len = 1
init_maj = [M]

# Fermionic gate

b1 = np.array([1, 0, 1, 0, 0, 0])
theta1 = 0.4

U = [[theta1, b1]]
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

A = np.nonzero(init_bin)[0] 

print(h)
t = 0.1
R = expm(4 * h * t)
n = np.array([0, 0, 0])
sum = 0

m = int(np.sum(init_bin)/2)
#print(m)
for combo in combinations(list(range(N)), m):
    J = list(combo)
    Js = [x for j in J for x in (2*j, 2*j + 1)]
    Q = R[np.ix_(Js, A)]
    sum+= np.linalg.det(Q) * np.prod(1j * (2 * n[J] - 1))

print("Expectation value after rotation = ", sum)