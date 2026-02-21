import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def parse_name(path: Path):
    stem = path.stem  # filename without extension
    parts = stem.split("_")

    # If you truly have exactly 9 fields (no optional note):
    # nf, dt, n, init_len, nnz_h, nnz_v, len_trunc, coeff_trunc, trott = parts

    # If you might have an optional note appended:
    if len(parts) < 9:
        raise ValueError(f"Unexpected filename format: {path.name}")

    nf        = int(parts[0])
    dt        = float(parts[1])
    n         = int(parts[2])
    init_len  = int(parts[3])
    nnz_h     = int(parts[4])
    nnz_v     = int(parts[5])
    len_trunc = int(parts[6])
    coeff_trunc = float(parts[7])   # works with "1e-10"
    trott     = int(parts[8])
    note      = "_".join(parts[9:]) if len(parts) > 9 else ""

    return {
        "path": path,
        "nf": nf,
        "dt": dt,
        "n": n,
        "init_len": init_len,
        "nnz_h": nnz_h,
        "nnz_v": nnz_v,
        "len_trunc": len_trunc,
        "coeff_trunc": coeff_trunc,
        "trott": trott,
        "note": note,
    }

'''
folder = Path("ta0220")   # e.g. Path("data/output")

files = sorted(folder.glob("*.csv"))  # or "*.txt" if that's your extension

metas = [parse_name(p) for p in files]

plt.figure()

for m in metas:
    data = np.loadtxt(m["path"], delimiter=",")
    x = np.arange(1, len(data[:])+1)
    y = data[:]

    label = f"coeff={m['coeff_trunc']:.0e}, trott={m['trott']}, L={m['len_trunc']}"
    plt.plot(x, y, marker='o', linestyle='--', label=label)

plt.legend(fontsize=8)
plt.xlabel("x")
plt.ylabel("y")
plt.yscale('log')  
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
'''


# ... keep your parse_name() ...

folder = Path("ta0220")
files = sorted(folder.glob("*.csv"))
metas = [parse_name(p) for p in files]

# --- color map by coeff_trunc ---
coeffs = sorted({m["coeff_trunc"] for m in metas})
cmap = plt.get_cmap("tab10")  # good for up to ~10 distinct groups
coeff_to_color = {c: cmap(i % cmap.N) for i, c in enumerate(coeffs)}

# --- style map by trott (optional) ---
trotts = sorted({m["trott"] for m in metas})
linestyles = ["-.", "-", "--", ":"]
trott_to_ls = {t: linestyles[i % len(linestyles)] for i, t in enumerate(trotts)}

plt.figure()

for m in metas:
    data = np.loadtxt(m["path"], delimiter=",")
    y = np.asarray(data).ravel()
    x = np.arange(1, len(y) + 1)

    plt.plot(
        x, y,
        color=coeff_to_color[m["coeff_trunc"]],
        linestyle=trott_to_ls[m["trott"]],
        marker="o",
        linewidth=1.5,
        markersize=3,
        label=f"coeff={m['coeff_trunc']:.0e}, trott={m['trott']}"
    )

plt.xlabel("timestep")
plt.ylabel(r'$\left\|Tr(\rho O^{\ell}_{\mathrm{MP}}) - Tr(\rho O_{\mathrm{trott}})\right\|_{2}$')
plt.yscale("log")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# If you have many curves, legend can get huge.
# A nicer alternative is a compact legend per coeff (see below).
plt.legend(fontsize=8, ncol=1)
plt.tight_layout()
plt.show()
