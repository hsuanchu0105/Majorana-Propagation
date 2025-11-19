import numpy as np
from scipy.linalg import expm



def make_meshgrid(arrays, indexing="ij"):
    """
    arrays: list/tuple of 1D arrays, length = s 
    returns: list of s arrays, all with the same shape
    """
    return np.meshgrid(*arrays, indexing=indexing)

'''
s = 3

axes = [np.random.randint(0, 2, size=6) for _ in range(s)]    # list of s 1D arrays

print(axes)

grids = make_meshgrid(axes, indexing="ij")
#g is one of the meshgrid arrays, g.ravel(): flatten g, np.stack() makes it into a 2d array 
coords = np.stack([g.ravel() for g in grids], axis=-1)   

print(coords)


s = 3
axes = [np.random.randint(0, 2, size=6) for _ in range(s)]

print(axes)

grids = np.meshgrid(*axes, indexing="ij")  # same as make_meshgrid

shape = grids[0].shape  # (6, 6, 6)

for idx in np.ndindex(shape):   # idx is a tuple like (i, j, k)
    # values from each axis at this position:
    point = [g[idx] for g in grids]  # or [axes[d][idx[d]] for d in range(s)]
    
    print("indices:", idx, "values:", point)

'''

def canonicalize_index(idx):
    """
    idx: a tuple of integers, e.g. (2, 1, 2)
    returns: (sign, canonical_tuple)
        - sign is +1 or -1
        - canonical_tuple is sorted, strictly increasing, with no repeats
    """
    arr = list(idx)

    # 1) compute sign from number of inversions (pairs out of order)
    inv_count = 0
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                inv_count += 1

    sign = -1 if (inv_count % 2 == 1) else 1

    # 2) sort the indices (this is the reordered product)
    arr.sort()

    # 3) remove duplicates in pairs (e_i^2 = 1)
    canonical = []
    i = 0
    while i < len(arr):
        j = i + 1
        # count how many times arr[i] appears
        while j < len(arr) and arr[j] == arr[i]:
            j += 1
        count = j - i

        # if odd multiplicity, keep one copy
        if count % 2 == 1:
            canonical.append(arr[i])

        i = j

    return sign, tuple(canonical)



s = 5
axes = [np.random.randint(0, 2, size=6) for _ in range(s)]
shape = tuple(len(a) for a in axes)   

for idx in np.ndindex(shape):
    # idx is like (i, j, k), 0-based
    
    sign, new_idx = canonicalize_index(idx)

    # if you prefer 1-based labels for math:
    idx_1based     = tuple(i + 1 for i in idx)
    new_idx_1based = tuple(i + 1 for i in new_idx)

    print(
        f"original indices (0-based): {idx}, "
        f"canonical: sign={sign}, indices={new_idx}"
    )
    # or with 1-based:
    # print(f"{idx_1based}  ->  {sign} * {new_idx_1based}")
