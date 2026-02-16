import cmath 
import numpy as np
from scipy.linalg import expm
import functools
from itertools import combinations
import matplotlib.pyplot as plt
import os


# Pauli matrices
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

def kron_all(ops):
    """Kronecker product of a list of 2x2 operators, left-to-right."""
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out

def majorana_matrices(nf):
    
    Maj = []
    for k in range(nf):
        prefix = [Z] * k
        suffix = [I] * (nf - k - 1)

        Maj.append(kron_all(prefix + [X] + suffix))  # gamma_{2k}
        Maj.append(kron_all(prefix + [Y] + suffix))  # gamma_{2k+1}
    return Maj

def Maj_to_mtx(len, MajList, nf):
    Maj_mtx = majorana_matrices(nf)
    mtx = np.zeros( (2 ** nf, 2 ** nf ))

    for i in range(len):
        MajOp = MajList[i]
        x = np.eye(2**nf) * (1j ** MajOp.rb())
        for j in range(MajOp.N):
            if(MajOp.b[j]==1):
                x = x @ Maj_mtx[j]
        mtx = mtx + x
    return mtx

# tranform observable in Majorana form into matrix form 
def ObsToMtx(Input_Node, lenN, N, nf):
    Maj_mtx = majorana_matrices(nf)
    mtx = np.zeros((2**N, 2**N), dtype = complex)
    for i in range(lenN):   
        bin = Input_Node[i].b
        coeff = Input_Node[i].c
        factors = [Maj_mtx[j] for j in range(2 * N) if bin[j] == 1]
        Maj1 = functools.reduce(np.dot, factors, np.eye(2**N, dtype = complex))
        mtx += coeff * Maj1
    return mtx

def DirectExp(init_len, init_maj, v, h, t, i, nf, nf2):

    Maj_mtx = majorana_matrices(nf)
    
    #for i in range(2**nf):
    test = np.eye(2**nf)[i]
    rho = np.reshape(test, (2**nf, 1))
    rhoT = np.transpose(rho)

    F = np.zeros((2**nf, 2**nf), dtype = complex)
    V = np.zeros((2**nf, 2**nf), dtype = complex)
    
    for m in range(nf2):
        for k in range(m, nf2):
            F += 2 * 1j * h[m][k] * (Maj_mtx[m] @ Maj_mtx[k])
    for j in range(nf2):
        for k in range(nf2):
            for l in range(nf2):
                for m in range(nf2):
                    V+= v[j][k][l][m] * (Maj_mtx[j] @ Maj_mtx[k] @ Maj_mtx[l]@ Maj_mtx[m])
    H = Maj_to_mtx(init_len, init_maj, nf)
    H = expm(1j * (V+F) * t) @ H @ expm(-1j * (V+F) * t)

    Expect_dir = np.trace(rho @ rhoT @ H, dtype = complex)
    return Expect_dir
        #print("Expectation value by direct exponential = ", Expect_dir)

def Trotterization(exp, init_len, init_maj, U, nf):


    for i in range(2**nf):
        test = np.eye(2**nf)[i]
        rho = np.reshape(test, (2**nf, 1))
        rhoT = np.transpose(rho)

        #print(rho)
        
        H = Maj_to_mtx(init_len, init_maj, nf)
        #print(H)
        #Expect_dir = np.cos(theta) * np.trace(rho @ rhoT @ Mb) + np.sin(theta) * 1j * np.trace(rho @ rhoT @ Mbj @ Mb) + np.cos(theta) * np.trace(rho @ rhoT @ Mc) + np.sin(theta) * 1j * np.trace(rho @ rhoT @ Mbj @ Mc)
        
        for k in range(len(U)):
            M = MajoranaOp(len(U[k][1]), U[k][1]) 
            Mbj = Maj_to_mtx(1, [M], nf)
            theta = U[k][0]
            #print(theta)
            H = expm(1j * theta  *  Mbj/2) @ H @ expm(-1j * theta  *  Mbj/2)
            #print(H)
        
        #print(H)
        #print("diff ", H - H.conj().T)
        Expect_dir = np.trace(rho @ rhoT @ H)
        exp[i] = Expect_dir

"""
return the parity of a certain permutation 
"""
def perm_parity(k, l, m, n):
    par = 1
    input = np.array([k, l, m, n])
    for i in range(1, 4):
        for j in range(i):
            if(input[i]< input[j]):
                input[i], input[j] = input[j], input[i]
                par *= -1

    #print(input)
    return par

"""
Node used for Majorana Propagation. Each node contains one binary representation and coefficient (default as 1)
"""
class Node:
    def __init__(self, b, c = 1):
        self.b = b
        self.c = c
        self.rb = 0
        self.N = len(b)
    """
    returned a numpy array of paired indices (input:2N, output:N), Ex. [0 1 0 0 1 1] -> [-1]
    if one unpaired is found, one get -1 at the end of the array, and return immediately 
    """
    def BinPair(self):
        Pair = []
        for i in range(0, self.N, 2):
            s = self.b[i] + self.b[i+1]
            if s == 1:
                Pair.append(-1)
                return np.array(Pair)
            elif s == 2:
                Pair.append(1)
            else:
                Pair.append(0)
        #print("Pair in the function", Pair)
        return np.array(Pair)
    def __str__(self):
        return f"Node(b={self.b}, c={self.c})"
"""
Majorana Operator 
"""
class MajoranaOp:   
    def __init__(self, N, b):
        self.b = b 
        self.N = N                          # 2N in paper
    def rb(self):
        w = sum(self.b)
        if(w % 4 == 0 or w % 4 == 1):
            return 0
        else:
            return 1
        
def sparsity(hind, Vind, nf2):
# note that we didn't consider V have terms V[i][i][j][k] != 0
    delta = 0
    for i in range(nf2):
        tmp = 0
        for j in range(2):
            tmp += np.count_nonzero(hind[j] == i)/2 #symmetric
        for j in range(4):
            tmp += np.count_nonzero(Vind[j] == i)
        if(tmp > delta):
            delta = tmp
    
    return delta 
    
def AppendH(U, ind, theta, trott_order, nf2):

    if(trott_order == 2):
        hcnt = 0
        for i in range(len(ind[0])):
            b1 = np.zeros(nf2)
            if(ind[0][i] < ind[1][i]):
                b1[ind[0][i]] = 1
                b1[ind[1][i]] = 1
                U.append([theta, b1])
                hcnt += 1

        for i in range(len(U)-1, len(U) - hcnt -1, -1):
            U.append(U[i])     

def AppendV(U, ind, theta, trott_order, nf2):

    if(trott_order ==2):
        for i in range(len(ind[0])):
            b2 = np.zeros(nf2)
            b2[ind[0][i]] = (b2[ind[0][i]]+ 1) % 2
            b2[ind[1][i]] = (b2[ind[1][i]]+ 1) % 2
            b2[ind[2][i]] = (b2[ind[2][i]]+ 1) % 2
            b2[ind[3][i]] = (b2[ind[3][i]]+ 1) % 2

            # parity check? 

            U.append([theta, b2])

        cur_pos = len(U)
        for i in range(cur_pos-1, cur_pos- len(ind[0]) - 1, -1):
            U.append(U[i])

"""
Majorana Propagation for 1 Fermionic gate
"""
def M1Prg(Nin, theta_ex, b_ex):
    neg_cnt = 0                                 #negative sign from anti-commutivity 
    cons_len = min(len(b_ex), len(Nin.b))       #considered length
    
    #sign added by multiplication of two Majorana operators (nodes)
    for i in range(cons_len):
        if(Nin.b[i]==1):
            shade = [0] * (i + 1) + [1] * (len(b_ex) - i - 1)
            neg_cnt += np.inner(b_ex, shade)
    
    sign = 1
    if(neg_cnt % 2 == 1):
        sign = -1

    # put two binaries into same length 
    if(len(b_ex) < len(Nin.b)):
        long_arr = Nin.b
        short_arr = b_ex
    else:
        long_arr = b_ex
        short_arr = Nin.b

    short_padded = np.zeros_like(long_arr)
    short_padded[:short_arr.shape[0]] = short_arr

    
    bsum = short_padded + long_arr
    bout = np.array([x % 2 for x in bsum])

    imag = MajoranaOp(len(b_ex), b_ex).rb() + 1

    c1 = Nin.c * cmath.cos(theta_ex)
    c2 = Nin.c * cmath.sin(theta_ex) *  (1j ** imag) * sign 
    #print(imag, sign, c2)

    return c1,  c2 , bout 

"""
Main function of Majorana Propagation 
"""
def MajoranaPropagation(trunc, Nin, lenU, U):
    # trunc: List of truncation parameters [length truncation, coefficient truncation]
    # Nin: List of input nodes 
    # lenU: width of Fermionic gate 
    # U: Fermionic gate 
    # output: 

    # parameters for truncation
    length_trunc = trunc[0]
    coeff_thres = trunc[1]

    
    Nin = list(Nin) #shallow copy
    # parameters of Fermionic circuit U
    L = lenU                   
    
    # index bookkeeping of current level (lv_end exclusive)
    lv_st = 0               
    lv_end = len(Nin) 
    current_pos = len(Nin) - 1

    '''
    print("length threshold = ", length_trunc, ", coefficient threshold = ", coeff_thres)
    print("Level 0 :")
    #print("input length = ", len(Nin))
    for k in range(lv_st, lv_end):
            print("coeff = ", Nin[k].c, "binary = ", Nin[k].b)
    '''

    for i in range(L):
        for j in range(lv_st, lv_end):
            if(len(Nin[j].b) < len(U[i][1])):
                long_arr = U[i][1]
                short_arr = Nin[j].b
            else:
                long_arr = Nin[j].b
                short_arr = U[i][1]

            short_padded = np.zeros_like(long_arr)
            short_padded[:short_arr.shape[0]] = short_arr

            #if(np.inner(short_padded, long_arr) % 2 == 0):
            if((sum(short_padded) * sum(long_arr) - np.inner(short_padded, long_arr)) % 2 == 0): #if M_b and M_{b_j} commute
                #pass
                N = Node(Nin[j].b, Nin[j].c)
                Nin.append(N)
                current_pos += 1
            else:
                
                coeff1, coeff2, bnew = M1Prg(Nin[j], U[i][0], U[i][1])
                #print(coeff2)
                #print(PpgList[j].b)
                Nl = Node(Nin[j].b, coeff1) 
                Nr = Node(bnew, coeff2)
                Nin.append(Nl)
                if(sum(bnew) > length_trunc):
                    pass
                    #print("length truncation")
                elif(np.abs(coeff2) < coeff_thres):
                    pass
                    #print("coefficient truncation")
                else:
                    Nin.append(Nr)
                    current_pos += 1
                current_pos += 1
        #PpgList.traverseAndPrint()
        #print("length = ", PpgList.len)
        lv_st = lv_end 
        lv_end = current_pos + 1
        """
        print("Level", i+1, ":")
        for k in range(lv_st, lv_end):
            print("coeff = ", Nin[k].c, "binary = ", Nin[k].b)
        """

    Nin = Nin[lv_st: lv_end]

    return Nin

def ExpectVal(Input_Node, lenN, rho):
    Expect = 0
    for i in range(lenN):
        Pair = Input_Node[i].BinPair()

        if(Pair[-1] != -1):
            while(len(Pair) < len(rho)):
                Pair = np.append(Pair, 0)
            #print("Pairs = ", Pair)
            PairedOne = np.inner(Pair, rho)           # { i | |n_i> = 1 and (b_{2i}, b_{2i+1}) is paired }
            Expect += ((-1)**PairedOne) * (1j**sum(Pair))* Input_Node[i].c  
            
    i
    return Expect


        
# First change H' into new basis, then write in the form of fermionic gates
def BasisChange(N, h, V, dt, bdry):
	# N: number of Fermionic mode
	# h: free-fermion Hamiltonian coefficient (2N * 2N matrix)
    # V: 4-leg tensor 
    # dt: time per timestep
	# output: tensor after contraction with R^T, resulting fermionic gate


    # find a way to check the correctness of contraction
    if(bdry):
        R = expm(2 * h * dt)
    else:
        R = expm(4 * h * dt)
    #print("R = ", R)
    Rt = np.transpose(R)
    #print(np.allclose(R, np.transpose(R))) # Why do changing R into Rt not change the result?
    V1 = np.einsum("jklm, jn -> nklm", V, Rt)
    V2 = np.einsum("nklm, ko -> nolm", V1, Rt)
    V3 = np.einsum("nolm, lp -> nopm", V2, Rt)
    V4 = np.einsum("nopm, mq -> nopq", V3, Rt)
	
    coef_sh = V4.shape
    U = []

    for k in range(coef_sh[0]):
        for l in range(coef_sh[1]):
            for m in range(coef_sh[2]):
                for n in range(coef_sh[3]):
                    b = np.zeros(2 * N)
                    if(V4[k][l][m][n] !=0): # maybe apply a threshold for coefficient
                        
                        theta = V4[k][l][m][n] * dt  #second order trotterization
                        b[k] += 1
                        b[l] += 1
                        b[m] += 1
                        b[n] += 1
                        for q in range(2 * N):
                            b[q] = b[q] % 2
                        
                        theta = theta * perm_parity(k, l , m, n)
                        U.append([theta, b])

    for i in range(len(U)-1, -1, -1):
        U.append(U[i])
    

    return V4, U

def twofourMajStrEvo(N, h, V, n, dt, Init_Node, trunc_param):
	# N: number of Fermionic mode
	# h: free-fermion Hamiltonian coefficient (2N * 2N matrix)
    # V: 4-leg tensor 
    # n: number of timesteps
    # dt: evolution time each timestep 
	# Initial Node: 
	# output: coefficient of majorana operator after evolution
	
    Node_next = Init_Node

    bdry = False
    for i in range(n):
        if(i==0):
            bdry = True
        V, U = BasisChange(N, h, V, dt, bdry) #coefficient in new basis
        #print("V_update= ", V)
        #for j in range(nf2):
            #for k in range(nf2):
                #for m in range(nf2):
                    #for l in range(nf2):
                        #if(V[j][k][m][l]!=0):
                            #print(j, k, m, l, V[j][k][m][l])
        #print("Fermionic gate (V)", U)
        #print("gate count", len(U))
        Node_next = MajoranaPropagation(trunc_param, Node_next, len(U), U)

    return Node_next

def Rotated_ExpectVal(NodeList, h, dt, tstep_num, rho, nf):
    
    R0 = expm(2 * h * dt)
    R = expm(4 * h * dt)
    Rm = R0
    for i in range(tstep_num - 1):
        Rm = R @ Rm
    Rm = R0 @ Rm

    #Rm = expm(4 * h * dt)

    Exp = 0
    #'''
    for node in NodeList:
        if(np.sum(node.b) % 2 == 0):
            #print(node)
            bin = node.b
            coeff = node.c
            m = int(np.sum(bin)/2)
            #print("m=", m)
            for combo in combinations(list(range(nf)), m):
                J = list(combo)
                Js = [x for j in J for x in (2*j, 2*j + 1)]
                A = np.nonzero(bin)[0] 
                Q = Rm[np.ix_(A, Js)]
                Exp+= coeff * np.linalg.det(Q) * np.prod(1j * (1 - 2 * rho[J] ))
                #contr = coeff * np.linalg.det(Q) * np.prod(1j * (1 - 2 * rho[J] ))
                #print(Js, A, contr)
        #print('\t')
    return Exp
    #'''


def coefftrunc_plot(coef_st, coef_end,rel_mp_global, rel_rot_global, filename):
     
    tc_coeff = 10**np.arange(coef_st, coef_end, -1, dtype = float)
    plt.figure()
    plt.plot(tc_coeff, rel_mp_global, marker='o', linestyle='-', label='rel_mp_global')
    plt.plot(tc_coeff, rel_rot_global, marker='o', linestyle='-', label='rel_rot_global')

    #plt.gca().invert_xaxis()
    plt.xlabel('truncation coefficient order')
    plt.ylabel('relative error (global)')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()


    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.show()

def lentrunc_plot(tc_st, tc_end, rel_mp_global, rel_rot_global, filename):
    tc_len = np.arange(tc_st, tc_end)  
    plt.figure()
    #plt.plot(tc_len, rel_mp_global, marker='o', linestyle='-')
    plt.plot(tc_len, rel_mp_global, marker='o', linestyle='-', label='rel_mp_global')
    plt.plot(tc_len, rel_rot_global, marker='o', linestyle='-', label='rel_rot_global')

    
    #plt.xlabel('truncation length')
    plt.xlabel(r'maximal degree $\ell$')
    plt.ylabel('relative error (global)')
    #plt.ylabel(r'$\left\|O^{ell}_{\mathrm{MP}} - O_{\mathrm{trott}}\right\|_{2}$')
    plt.yscale('log')  
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()


    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.show()

def ErrorPrint(_dexp, _obexp, _rexp, _ord):
    print('\t')
    print("-----------------Direct exponential---------------")
    print(_dexp)
    print("-----------------Majorana Propagation-------------")
    print(_obexp)
    print("------------Rotated Majorana Propagation----------")
    print(_rexp)



    #eps = 1e-15
    rel_maj = np.abs(_obexp - _dexp) / np.abs(_dexp)
    rel_rotm = np.abs(_rexp - _dexp) / np.abs(_dexp)

    print('\t')
    print("Relative error Majorana Propagation")
    print(rel_maj)

    print("Relative error rotated Majorana")
    print(rel_rotm)

    #2-norm 
    rel_ob_global = np.linalg.norm(_obexp - _dexp, ord = _ord) / np.linalg.norm(_dexp, ord = _ord)
    rel_re_global = np.linalg.norm(_rexp - _dexp, ord = _ord) / np.linalg.norm(_dexp, ord = _ord)

    print('\t')
    print("Global relative error")
    print(rel_ob_global, rel_re_global)