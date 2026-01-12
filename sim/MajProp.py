import cmath 
import numpy as np
from scipy.linalg import expm
import functools

#number of fermionic mode 
nf = 3
nf2 = 2 * nf

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

Maj_mtx = [m1, m2, m3, m4, m5, m6]

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



def Maj_to_mtx(len, MajList):
    mtx = np.zeros( (2 ** nf, 2 ** nf ))

    for i in range(len):
        MajOp = MajList[i]
        x = np.eye(2**nf) * (1j ** MajOp.rb())
        for j in range(MajOp.N):
            if(MajOp.b[j]==1):
                x = x @ Maj_mtx[j]
        mtx = mtx + x
    return mtx

def ExpectVal(Input_Node, lenN, rho):
    Expect = 0
    for i in range(lenN):
        Pair = Input_Node[i].BinPair()

        if(Pair[-1] != -1):
            while(len(Pair) < len(rho)):
                Pair = np.append(Pair, 0)
            #print("Pairs = ", Pair)
            PairedOne = np.inner(Pair, rho)           # { # i | |n_i> = 1 and (b_{2i}, b_{2i+1}) is paired }
            Expect += ((-1)**PairedOne) * (1j**sum(Pair))* Input_Node[i].c  

    return Expect

                
    
# tranform observable in Majorana form into matrix form 
def ObsToMtx(Input_Node, lenN, N):
    mtx = np.zeros((2**N, 2**N), dtype = complex)
    for i in range(lenN):   
        bin = Input_Node[i].b
        coeff = Input_Node[i].c
        factors = [Maj_mtx[j] for j in range(2 * N) if bin[j] == 1]
        Maj1 = functools.reduce(np.dot, factors, np.eye(2**N, dtype = complex))
        mtx += coeff * Maj1
    return mtx

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

    # initial Majorana operator
    #PpgList = LinkedList()
    #for i in range(len(Nin)):
    #    PpgList.insertNodeAtPosition(Nin[i], i)
    Nin = list(Nin) #shallow copy
    # parameters of Fermionic circuit U
    L = lenU                   
    
    # index bookkeeping of current level (lv_end exclusive)
    lv_st = 0               
    lv_end = len(Nin) 
    current_pos = len(Nin) - 1

    #'''
    print("length threshold = ", length_trunc, ", coefficient threshold = ", coeff_thres)
    print("Level 0 :")
    #print("input length = ", len(Nin))
    for k in range(lv_st, lv_end):
            print("coeff = ", Nin[k].c, "binary = ", Nin[k].b)
    #'''

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
                    print("length truncation")
                elif(np.abs(coeff2) < coeff_thres):
                    print("coefficient truncation")
                else:
                    Nin.append(Nr)
                    current_pos += 1
                current_pos += 1
        #PpgList.traverseAndPrint()
        #print("length = ", PpgList.len)
        lv_st = lv_end 
        lv_end = current_pos + 1
        #"""
        print("Level", i+1, ":")
        for k in range(lv_st, lv_end):
            print("coeff = ", Nin[k].c, "binary = ", Nin[k].b)
        #"""

    Nin = Nin[lv_st: lv_end]

    return Nin


# First change H' into new basis, then write in the form of fermionic gates
def BasisChange(N, h, V, dt):
	# N: number of Fermionic mode
	# h: free-fermion Hamiltonian coefficient (2N * 2N matrix)
    # V: 4-leg tensor 
    # dt: time per timestep
	# output: tensor after contraction with R^T, resulting fermionic gate

    
    R = expm(4 * h * dt)
    print("R = ", R)
    Rt = np.transpose(R)
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
                        theta = V4[k][l][m][n] * dt
                        b[k] = 1
                        b[l] = 1
                        b[m] = 1
                        b[n] = 1
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

    for i in range(n):
        V, U = BasisChange(N, h, V, dt) #updating V
        #print("V_update= ", V)
        for j in range(6):
            for k in range(6):
                for m in range(6):
                    for l in range(6):
                        if(V[j][k][m][l]!=0):
                            print(j, k, m, l, V[j][k][m][l])
        print("Fermionic gate (V)", U)
        print("width of gate =", len(U))
        Node_next = MajoranaPropagation(trunc_param, Node_next, len(U), U)

    return Node_next

def Rotated_ExpectVal(NodeList, h, rho):
    
    Exp = 0
    for node in NodeList:
        if(sum(node.b) % 2 == 0):
            pass

    return Exp





