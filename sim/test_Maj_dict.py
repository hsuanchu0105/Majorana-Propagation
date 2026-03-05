from MP_dict import * 


nf = 8
trunc_param = np.array([10, 1e-8])

#"""
init_len = np.random.randint(1, 5)

maj_bin = []
for i in range(init_len):
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




