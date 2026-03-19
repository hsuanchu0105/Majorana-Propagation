import numpy as np

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

def AppendH(U, h, ind, dt, trott_order, nf2):
               
    if(trott_order == 2):
        hcnt = 0
        for i in range(len(ind[0])):
            b1 = np.zeros(nf2)
            if(ind[0][i] < ind[1][i]):
                b1[ind[0][i]] = 1
                b1[ind[1][i]] = 1
                theta = h[ind[0][i]][ind[1][i]] * dt
                U.append([theta, b1])
                hcnt += 1
        for i in range(len(U)-1, len(U) - hcnt -1, -1):
            U.append(U[i])   

    elif(trott_order == 1):
        for i in range(len(ind[0])):
            b1 = np.zeros(nf2)
            if(ind[0][i] < ind[1][i]):
                b1[ind[0][i]] = 1
                b1[ind[1][i]] = 1
                theta = h[ind[0][i]][ind[1][i]] * dt
                U.append([2 * theta, b1])


def AppendV(U, V, ind, dt, trott_order, nf2):
    
    if(trott_order == 2):
        for i in range(len(ind[0])):
            b2 = np.zeros(nf2)
            # assume that we only have distinct and non-decreasing indices 
            b2[ind[0][i]] = 1
            b2[ind[1][i]] = 1
            b2[ind[2][i]] = 1
            b2[ind[3][i]] = 1
            theta = V[ind[0][i]][ind[1][i]][ind[2][i]][ind[3][i]] * dt
            U.append([theta, b2])
        cur_pos = len(U)
        for i in range(cur_pos-1, cur_pos- len(ind[0]) - 1, -1):
            U.append(U[i])

    elif(trott_order == 1):
        for i in range(len(ind[0])):
            b2 = np.zeros(nf2)
            b2[ind[0][i]] = 1
            b2[ind[1][i]] = 1
            b2[ind[2][i]] = 1
            b2[ind[3][i]] = 1

            theta = V[ind[0][i]][ind[1][i]][ind[2][i]][ind[3][i]] * dt
            U.append([2 * theta, b2])


def random_sparse_h(alpha, nf2, complex_coeff=False, seed=None):
    
    rng = np.random.default_rng(seed)

    # all (i,j) with i<j
    iu, ju = np.triu_indices(nf2, k=1)   # k=1 excludes diagonal
    all_pairs = np.column_stack([iu, ju])  # shape (M, 2)

    # sample alpha distinct pairs
    idx = rng.choice(all_pairs.shape[0], size=alpha, replace=False)
    pairs = all_pairs[idx]  # (alpha,2)

    # allocate h
    h = np.zeros((nf2, nf2), dtype=complex if complex_coeff else float)

    # assign coefficients ~ Unif(-1, 1)
    if complex_coeff:
        vals = rng.uniform(-1, 1, size=alpha) + 1j * rng.uniform(-1, 1, size=alpha)
    else:
        vals = rng.uniform(-1, 1, size=alpha)

    for (i, j), v in zip(pairs, vals):
        h[i, j] = v
        # mirror if you want Hermitian / symmetric:
        h[j, i] = -np.conj(v) if complex_coeff else -v

    return pairs, h


def random_sparse_v(alpha, nf2, complex_coeff=False, seed=None):

    rng = np.random.default_rng(seed)

    # number of possible unique 4-tuples = C(nf2, 4)
    total = (nf2 * (nf2-1) * (nf2-2) * (nf2-3)) // 24
    if alpha > total:
        raise ValueError(f"n={alpha} too large; max is C(nf2,4)={total}")

    # 2) sample n unique 4-tuples by sampling 4 distinct indices then sorting
    tuples = set()
    while len(tuples) < alpha:
        idx = rng.choice(nf2, size=4, replace=False)
        idx.sort()
        tuples.add(tuple(idx))
    tuples = np.array(list(tuples), dtype=int)  # shape (n,4)

    # 3) coefficients
    if complex_coeff:
        vals = rng.uniform(-1, 1, size=alpha) + 1j * rng.uniform(-1, 1, size=alpha)
        v = np.zeros((nf2, nf2, nf2, nf2), dtype=complex)
    else:
        vals = rng.uniform(-1, 1, size=alpha)
        v = np.zeros((nf2, nf2, nf2, nf2), dtype=float)

    # 4) assign (only canonical entry)
    for (j,k,l,m), val in zip(tuples, vals):
        v[j, k, l, m] = val

    return tuples, v