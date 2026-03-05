import cmath 
import numpy as np
from collections import defaultdict
from itertools import combinations
from datetime import date
from pathlib import Path
from tqdm import tqdm
from Op import *


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
    return Nout




        














