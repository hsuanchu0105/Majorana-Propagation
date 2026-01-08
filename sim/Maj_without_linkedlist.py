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

"""
LinkedList for recording Majorana Propagation 
"""
class LinkedList:
    def __init__(self):
        self.head = None
        self.len = 0
    def insertNodeAtPosition(self, newNode, position):
        if position == 0:
            newNode.next = self.head
            self.head =  newNode
        elif position > 0:    
            currentNode = self.head
            for _ in range(position - 1):
                if currentNode is None:
                    break
                currentNode = currentNode.next
            newNode.next = currentNode.next
            currentNode.next = newNode
        self.len +=1 

    def deleteSpecificNode(self, nodeToDelete):
        if self.head == nodeToDelete:
            self.head = self.head.next

        else:
            currentNode = self.head
            while currentNode.next and currentNode.next != nodeToDelete:
                currentNode = currentNode.next

            if currentNode.next is None:
                currentNode = None
            else:
                currentNode.next = currentNode.next.next
        self.len -= 1

    #endIndex exclusive 
    def getSlice(self, startIndex, endIndex):
        assert endIndex > startIndex 
        currentNode = self.head
        for i in range(startIndex):
            currentNode = currentNode.next
        self.head = currentNode
        for i in range(endIndex - startIndex):
            currentNode = currentNode.next
        currentNode = None

        self.len = endIndex - startIndex

    def traverseAndPrint(self):
        currentNode = self.head
        while currentNode:
            print( currentNode.b, ",", f"{currentNode.c.real:.3f}{currentNode.c.imag:+.3f}j", end=" -> ")
            currentNode = currentNode.next
        print("null")

    def PrintFrom(self, start, end):
        currentNode = self.head
        for i in range(start):
            currentNode = currentNode.next
        for i in range(start, end):
            print( currentNode.b, ",", f"{currentNode.c.real:.3f}{currentNode.c.imag:+.3f}j", end=" -> ")
            currentNode = currentNode.next
        print("null")

    def __getitem__(self, position):
        if(position ==0):
            return self.head
        else:
            currentNode = self.head
            for i in range(position, 0, -1):
                currentNode = currentNode.next
            return currentNode

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



"""
init_len = np.random.randint(1, 5)

maj_bin = []
for i in range(init_len):
    b = np.random.randint(0, 2, size=nf2)
    maj_bin.append(b)

#print(maj_bin)

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
U_wid = np.random.randint(1, 8)
for i in range(U_wid):
    theta = np.random.rand() * 2 * cmath.pi
    #b = np.array([1,1, 1])
    #while(sum(b)%2 != 0 or sum(b)== 1):
    b = np.random.randint(0, 2, size=6)
    U.append([theta, b])

"""
'''
a2 = np.array([1, 0, 1, 0, 0, 0])
M2 = MajoranaOp(6, a2)
N2 = Node(a2, 1j**M2.rb())

Init_Node = [N2]
init_len = 1
init_maj = [M2]

theta1 = cmath.pi/7
theta2 = cmath.pi/6
theta3 = cmath.pi/3
b1 = np.array([1, 0, 0, 1, 1, 1])
b2 = np.array([0, 1, 1, 1, 0, 0])
b3 = np.array([0, 0, 1, 0, 0, 0])

U_wid = 3
U = [[theta1, b1], [theta2, b2], [theta3, b3]]
'''
#'''

a2 = np.array([1, 1, 0, 0, 0, 0])
M2 = MajoranaOp(6, a2)
N2 = Node(a2, 1j**M2.rb())

Init_Node = [N2]
init_len = 1
init_maj = [M2]

#U_wid = 1
#theta = 0.2
#b = np.array([1, 1, 1, 1, 0, 0])
#U = [[theta, b]]


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
    #MajoranaPropagation(trunc_param, Init_Node, U_wid, U, rho_st)
    Output_Node = MajoranaPropagation(trunc_param, Init_Node, len(U), U)
    Exp_val = ExpectVal(Output_Node, len(Output_Node) , rho_st)
    print("Expectation value by Majorana Propagation = ", Exp_val)


#print("\t")

# direct calculation 
for i in range(2**nf):
    test = np.eye(2**nf)[i]
    rho = np.reshape(test, (2**nf, 1))
    rhoT = np.transpose(rho)

    #print(rho)
    
    H = Maj_to_mtx(init_len, init_maj)
    #print(H)
    #Expect_dir = np.cos(theta) * np.trace(rho @ rhoT @ Mb) + np.sin(theta) * 1j * np.trace(rho @ rhoT @ Mbj @ Mb) + np.cos(theta) * np.trace(rho @ rhoT @ Mc) + np.sin(theta) * 1j * np.trace(rho @ rhoT @ Mbj @ Mc)
    
    for k in range(len(U)):
        M = MajoranaOp(len(U[k][1]), U[k][1]) 
        Mbj = Maj_to_mtx(1, [M])
        theta = U[k][0]
        #print(theta)
        H = expm(1j * theta  *  Mbj/2) @ H @ expm(-1j * theta  *  Mbj/2)
        #print(H)
    
    #print(H)
    #print("diff ", H - H.conj().T)
    Expect_dir = np.trace(rho @ rhoT @ H)
    #Expect_dir = np.trace(rho @ rhoT @  expm(1j * theta  *  Mbj/2) @ Mb @ expm(-1j * theta  *  Mbj/2) ) + np.trace(rho @ rhoT @  expm(1j * theta  *  Mbj/2)  @ Mc @ expm(-1j * theta * Mbj/2))

    print("Expectation value by direct calculation = ", Expect_dir)
#'''



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
#print(Node_out)
#Exp_val = Rotated_ExpectVal(Node_out , rho_st)
#print("Expectation value by Majorana Propagation = ", Exp_val)


