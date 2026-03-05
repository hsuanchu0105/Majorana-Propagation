import numpy as np
import functools


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
def Obs_to_mtx(Input_Node, lenN, N, nf):
    Maj_mtx = majorana_matrices(nf)
    mtx = np.zeros((2**N, 2**N), dtype = complex)
    for i in range(lenN):   
        bin = Input_Node[i].b
        coeff = Input_Node[i].c
        factors = [Maj_mtx[j] for j in range(2 * N) if bin[j] == 1]
        Maj1 = functools.reduce(np.dot, factors, np.eye(2**N, dtype = complex))
        mtx += coeff * Maj1
    return mtx

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