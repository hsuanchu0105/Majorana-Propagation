import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

folder = Path("len_distr_0224")      
pattern = "timestep*.csv"                  # or e.g. "counts_*.csv"

files = sorted(folder.glob(pattern))

def extract_step(p: Path) -> int:
    """
    Try to extract timestep from filename.
    Works for names like:
      step_12.csv, counts_step12.csv, ..._t=12_..., ..._12.csv
    If no number found, returns 0 (so sorting still works).
    """
    nums = re.findall(r"\d+", p.stem)
    return int(nums[-1]) if nums else 0

# sort by timestep number (recommended)
files = sorted(files, key=extract_step)

blocks = []
for p in files:
    arr = np.loadtxt(p, delimiter=",")
    # Ensure 2D (if file has a single row, loadtxt can return 1D)
    arr = np.atleast_2d(arr)
    blocks.append(arr)

counts_all = np.vstack(blocks) if blocks else np.zeros((0, 0), dtype=int)

print("Loaded:", len(files), "files")
print("Concatenated counts_all shape:", counts_all.shape)  # (total_snapshots, nf2)

nf2 = counts_all.shape[1]
t = np.arange(counts_all.shape[0])  # snapshot index across all timesteps

plt.figure()
for j in range(nf2):
    plt.plot(t, counts_all[:, j], label=f"len={j+1}")

plt.xlabel("snapshot index")
plt.ylabel("count")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# legend only if nf2 small
if nf2 <= 12:
    plt.legend(fontsize=8, ncol=2)

plt.tight_layout()
plt.show()