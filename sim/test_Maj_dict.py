from MP_dict import * 

'''
trunc_param = np.array([6, 1e-10])

b1 = np.array([1, 1, 0, 0, 0, 0])
M1 = MajoranaOp(6, b1)
N1 = Node(b1, 1j**M1.rb())
b2 = np.array([1, 0, 1, 0, 0, 0 ])
M2 = MajoranaOp(6, b2)
N2 = Node(b2, 1j**M2.rb())
Init_Node = [N1, N2]


theta1 = cmath.pi/7
#theta2 = 0.53
#theta3 = 1.05
#b1 = np.array([1, 1, 0, 0, 1, 0])
#b2 = np.array([0, 0, 0, 1, 1, 1])
#b3 = np.array([0, 1, 0, 1, 0, 1])
#U_wid = 3
#U = [[theta1, b1], [theta2, b2], [theta3, b3]]

#theta1 = 0.45

b1 = np.array([1, 0, 0, 1, 1, 1])

U_wid = 1
U = [[theta1, b1]]
#"""

print("Fermionic gate:", U)
print('\t')
rho_st = np.array([0, 0, 0])
c = np.array([0, 0, 0])

# transform into binary representation |n> = |n_1 n_2 n_3>

Nout, sampled_levels = MajoranaPropagation(trunc_param, Init_Node, 1, U)

mp = np.zeros(8)

for i in range(8):
    c[0] = i/4
    r1 = i - c[0] * 4
    c[1] = r1/2
    r2 = r1 - c[1] * 2
    c[2] = r2
    
    for j in range(3):
        rho_st[j] = c[j]
    print("input Fock state = ", rho_st)

    mp[i] = ExpectVal(Nout, len(Nout), rho_st)


print(mp)


# direct calculation 
theta = cmath.pi/7
# Pauli gates 
X = np.array([[0, 1], [1, 0]])
Y = 1j * np.array([[0, -1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)

m1 = np.kron(np.kron(X, I), I)
m2 = np.kron(np.kron(Y, I), I)
m3 = np.kron(np.kron(Z, X), I)
m4 = np.kron(np.kron(Z, Y), I)
m5 = np.kron(np.kron(Z, Z), X)
m6 = np.kron(np.kron(Z, Z), Y)

Mb = 1j * m1 @ m2
Mc = 1j * m1 @ m3
Mbj = m1 @ m4 @ m5 @ m6

#print("Mb = ", Mb)
#print("Mc = ", Mc)
#print("Mbj = ", Mbj)

for i in range(8):
    test = np.eye(8)[i]
    rho = np.reshape(test, (8, 1))
    rhoT = np.transpose(rho)

    Expect_dir = np.cos(theta) * np.trace(rho @ rhoT @ Mb) + np.sin(theta) * 1j * np.trace(rho @ rhoT @ Mbj @ Mb) + np.cos(theta) * np.trace(rho @ rhoT @ Mc) + np.sin(theta) * 1j * np.trace(rho @ rhoT @ Mbj @ Mc)

    print("Expectation value by direct calculation = ", Expect_dir)


'''
nf = 3
trunc_param = np.array([20, 1e-10])

#"""
init_len = np.random.randint(1, 5)

maj_bin = []
for i in range(init_len):
    #b = np.array([1,1, 1])
    #while(sum(b)%2 != 0 or sum(b)== 1):
    b = np.random.randint(0, 2, size=2 * nf)
    #print("length= ", len(b))
    maj_bin.append(b)

print(maj_bin)

#Initial Majorana operator
init_maj =[]
for i in range(init_len):
    M = MajoranaOp(len(maj_bin[i]), maj_bin[i])
    init_maj.append(M)

Init_Node = []
for i in range(init_len):
    N = Node(maj_bin[i], 1j**init_maj[i].rb())
    Init_Node.append(N)


U = []
U_wid = np.random.randint(1, 2 ** nf)
for i in range(U_wid):
    theta = np.random.rand() * 2 * cmath.pi
    #b = np.array([1,1, 1])
    #while(sum(b)%2 != 0 or sum(b)== 1):
    b = np.random.randint(0, 2, size= 2 * nf)
    U.append([theta, b])

#"""




print("Fermionic gate:", U)
print('\t')
rho_st = np.zeros(nf, dtype = int)
c = np.zeros(nf, dtype = int)

Nout, sampled_levels = MajoranaPropagation(trunc_param, Init_Node, len(U), U)

mp = np.zeros(2 ** nf)

for i in range(2**nf):
    rem = i
    for j in range(nf):
        c[j] = rem / 2**(nf - j - 1)
        rem = rem - c[j] * 2**(nf - j - 1)

    for j in range(nf):
        rho_st[j] = c[j]
    
    print("input Fock state = ", rho_st)

    mp[i] = ExpectVal(Nout, len(Nout), rho_st)


print(mp)





for i in range(2**nf):
    test = np.eye(2**nf)[i]
    rho = np.reshape(test, (2**nf, 1))
    rhoT = np.transpose(rho)

    #print(rho)
    
    H = Maj_to_mtx(init_len, init_maj, nf)
    #print(H)
    #Expect_dir = np.cos(theta) * np.trace(rho @ rhoT @ Mb) + np.sin(theta) * 1j * np.trace(rho @ rhoT @ Mbj @ Mb) + np.cos(theta) * np.trace(rho @ rhoT @ Mc) + np.sin(theta) * 1j * np.trace(rho @ rhoT @ Mbj @ Mc)
    
    for k in range(U_wid):
        M = MajoranaOp(len(U[k][1]), U[k][1]) 
        Mbj = Maj_to_mtx(1, [M], nf)
        theta = U[k][0]
        #print(theta)
        H = expm(1j * theta  *  Mbj/2) @ H @ expm(-1j * theta  *  Mbj/2)
        #print(H)
    
    #print(H)
    #print("diff ", H - H.conj().T)
    Expect_dir = np.trace(rho @ rhoT @ H)
    #Expect_dir = np.trace(rho @ rhoT @  expm(1j * theta  *  Mbj/2) @ Mb @ expm(-1j * theta  *  Mbj/2) ) + np.trace(rho @ rhoT @  expm(1j * theta  *  Mbj/2)  @ Mc @ expm(-1j * theta * Mbj/2))

    print("Expectation value by direct calculation = ", Expect_dir)




