import cmath 
import numpy as np
from collections import defaultdict
from itertools import combinations
from datetime import date
from pathlib import Path
from tqdm import tqdm
from .Op import *


def b_to_int(b) -> int:
    """Binary numpy array/list -> bitmask int (bit i = b[i]). Hashable, O(1) popcount/AND."""
    x = 0
    for i, v in enumerate(b):
        if v:
            x |= (1 << i)
    return x

def int_to_b(x, n) -> np.ndarray:
    """Bitmask int -> binary numpy array of length n."""
    return np.array([(x >> i) & 1 for i in range(n)], dtype=np.uint8)

def gate_rb(weight) -> int:
    """Same as MajoranaOp(...).rb() but taking a precomputed Hamming weight."""
    return 0 if (weight % 4) in (0, 1) else 1

def histogram_from_state(state, nf2):
    hist = np.zeros(nf2, dtype=int)
    for key in state.keys():
        j = key.bit_count()
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
Main function of Majorana Propagation

State terms and gates are represented as Python-int bitmasks (bit i <-> b[i])
rather than numpy arrays/tuples. This lets commutation, sign, and XOR-update
be computed with O(1) int ops (a.bit_count(), a & gate_b, a ^ gate_b) instead
of allocating small numpy arrays per (state term, gate) pair, and quantities
that only depend on the gate (cos/sin of theta, the (1j**imag) phase, the
list of set-bit positions) are hoisted out of the per-term loop.
"""
def MajoranaPropagation(trunc, Nin, lenU, U, save_hist=False, filesuffix="", stride=10):
    length_trunc = trunc[0]
    coeff_thres  = trunc[1]

    nf2 = len(Nin[0].b)
    L = lenU

    # state: bitmask int -> coefficient
    state = defaultdict(complex)
    for node in Nin:
        state[b_to_int(node.b)] += node.c

    for i in tqdm(range(L), total=L, desc="Running"):
        theta = U[i][0]
        gate_b = b_to_int(U[i][1])
        gate_weight = gate_b.bit_count()

        # gate-dependent constants, computed once per gate rather than per term
        cos_theta = cmath.cos(theta)
        sin_i = cmath.sin(theta) * (1j ** (gate_rb(gate_weight) + 1))
        bit_positions = [j for j in range(nf2) if (gate_b >> j) & 1]

        new_state = defaultdict(complex)

        for a, coeff in state.items():
            if coeff_thres and abs(coeff) < coeff_thres:
                continue # skip current iteration of for loop

            a_weight = a.bit_count()
            overlap = (a & gate_b).bit_count()

            if ((a_weight * gate_weight - overlap) % 2) == 0:
                # no branching, keep same binary
                new_state[a] += coeff
            else:
                # branching: left branch always stays on same binary
                new_state[a] += coeff * cos_theta

                # sign from anti-commuting the gate's Majorana factors past
                # the ones already set in `a` (parity of inversions between
                # bits of a and bits of gate_b)
                parity = 0
                for j in bit_positions:
                    parity ^= (a & ((1 << j) - 1)).bit_count() & 1
                sign = -1 if parity else 1

                # right branch: apply truncations
                bnew = a ^ gate_b
                coeff2 = coeff * sin_i * sign
                if bnew.bit_count() <= length_trunc and (not coeff_thres or abs(coeff2) >= coeff_thres):
                    new_state[bnew] += coeff2

        state = new_state

    # convert dict back to list of Nodes
    Nout = [Node(int_to_b(k, nf2), c) for k, c in state.items()]
    return Nout




        














