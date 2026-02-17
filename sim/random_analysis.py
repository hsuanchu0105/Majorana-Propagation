def random_sparse_h(nf2, nmin=1, nmax=6, complex_coeff=False, seed=None):
    rng = np.random.default_rng(seed)

    # 1) choose number of terms
    n = rng.integers(nmin, nmax + 1)   # inclusive

    # 2) all (i,j) with i<j
    iu, ju = np.triu_indices(nf2, k=1)   # k=1 excludes diagonal
    all_pairs = np.column_stack([iu, ju])  # shape (M, 2)

    # 3) sample n distinct pairs
    idx = rng.choice(all_pairs.shape[0], size=n, replace=False)
    pairs = all_pairs[idx]  # (n,2)

    # allocate h
    h = np.zeros((nf2, nf2), dtype=complex if complex_coeff else float)

    # 4) assign coefficients ~ N(0,1)
    if complex_coeff:
        vals = rng.uniform(-1, 1, size=n) + 1j * rng.uniform(-1, 1, size=n)
    else:
        vals = rng.uniform(-1, 1, size=n)

    for (i, j), v in zip(pairs, vals):
        h[i, j] = v
        # mirror if you want Hermitian / symmetric:
        h[j, i] = -np.conj(v) if complex_coeff else -v

    return n, pairs, h