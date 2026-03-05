import numpy as np
from collections import defaultdict
import cmath
from scipy.linalg import expm
from tqdm import tqdm

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
    
def b_to_key(b) -> tuple:
    """Binary numpy array/list -> hashable key."""
    return tuple(np.asarray(b, dtype=np.uint8).tolist())

def key_to_b(key) -> np.ndarray:
    """Key tuple -> numpy array."""
    return np.fromiter(key, dtype=np.uint8)

def commute_check(b_term, b_gate) -> bool:
    b_term = np.asarray(b_term, dtype=np.uint8)
    b_gate = np.asarray(b_gate, dtype=np.uint8)

    if b_term.shape[0] < b_gate.shape[0]:
        long_arr, short_arr = b_gate, b_term
    else:
        long_arr, short_arr = b_term, b_gate

    short_padded = np.zeros_like(long_arr)
    short_padded[:short_arr.shape[0]] = short_arr

    return ((int(short_padded.sum()) * int(long_arr.sum()) - int(np.inner(short_padded, long_arr))) % 2) == 0

def histogram_from_state(state, nf2):
    hist = np.zeros(nf2, dtype=int)
    for key in state.keys():
        j = sum(key)
        if 1 <= j <= nf2:
            hist[j - 1] += 1
    return hist


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

def MajoranaPropagation(trunc, Nin, lenU, U, save_hist=False, filesuffix="", stride=10):
    length_trunc = trunc[0]
    coeff_thres  = trunc[1]

    nf2 = len(Nin[0].b)
    L = lenU

    # state: key(tuple(b)) -> coefficient
    state = defaultdict(complex)
    for node in Nin:
        state[b_to_key(node.b)] += node.c

    sampled_levels = [0]
   

    #for i in range(L):
    for i in tqdm(range(L), total=L, desc="Running"):
        gate_coeff = U[i][0]
        gate_b = np.asarray(U[i][1], dtype=np.uint8)

        new_state = defaultdict(complex)

        for key, coeff in state.items(): # returns a view object containing tuples of key and value pairs from state
            if coeff_thres and abs(coeff) < coeff_thres:
                continue # skip current iteration of for loop

            b = key_to_b(key)

            if commute_check(b, gate_b):
                # no branching, keep same binary
                new_state[key] += coeff
            else:
                # branching
                node = Node(b, coeff)
                coeff1, coeff2, bnew = M1Prg(node, gate_coeff, gate_b)

                # left branch always stays on same binary
                new_state[key] += coeff1

                # right branch: apply truncations
                bnew = np.asarray(bnew, dtype=np.uint8)
                if int(bnew.sum()) <= length_trunc and (not coeff_thres or abs(coeff2) >= coeff_thres):
                    new_state[b_to_key(bnew)] += coeff2

        state = new_state



    

    # convert dict back to list of Nodes
    Nout = [Node(np.array(k, dtype=np.uint8), c) for k, c in state.items()]
    return Nout, sampled_levels

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
            
    
    return Expect