import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import date
import os

cn = 5

mp_color = "tab:blue"
rmp_color = "tab:orange"

d = np.load("analysis/ta0320/93915132_1922132_913395238_errors.npz")
err = d["err"]
err_rot = d["err_rot"]

'''
# for concacentation 
d2 = np.load("analysis/ta0313/393164_92333132_5238_errors.npz")

err1 = d1["err"]          # shape (2, 10)
err_rot1 = d1["err_rot"]

err2 = d2["err"][-3:, :]     # take last 3 cases from second file
print(err2.shape)
err_rot2 = d2["err_rot"][-3:, :]

err = np.concatenate([err1, err2], axis=0)          # shape (cn, 5)
err_rot = np.concatenate([err_rot1, err_rot2], axis=0)
'''

# err, err_rot: shape (cn, n_samples)
# Avoid log(0)
eps = 1e-300
log_err   = np.log(np.clip(err, eps, None))
log_err_r = np.log(np.clip(err_rot, eps, None))

x = np.arange(1, cn + 1)
w = 0.18  # horizontal offset between the two violins

fig, ax = plt.subplots()

# Matplotlib wants a list of 1D arrays, one per violin
data_mp = [log_err[i, :] for i in range(cn)]
data_r  = [log_err_r[i, :] for i in range(cn)]

vp1 = ax.violinplot(
    data_mp,
    positions=x - w,
    widths=0.30,
    showmeans=True,
    showmedians=False,
    showextrema=False,
)

vp2 = ax.violinplot(
    data_r,
    positions=x + w,
    widths=0.30,
    showmeans=True,
    showmedians=False,
    showextrema=False,
)





# Make the two sets visually distinct (no custom colors required)
for body in vp1["bodies"]:
    body.set_facecolor(mp_color)
    body.set_edgecolor("black")
    body.set_alpha(0.35)
for body in vp2["bodies"]:
    body.set_facecolor(rmp_color)
    body.set_edgecolor("black")
    body.set_alpha(0.55)

    

ax.set_xticks(x)
ax.set_xlabel("Case")
ax.set_ylabel("log(Relative error)")
ax.grid(True, which="both", linestyle="--", linewidth=0.5)
ax.set_title("Relative error distributions (violin)")

# Simple legend proxies
ax.plot([], [], "-", label="MP", alpha=0.35)
ax.plot([], [], "-", label="RMP", alpha=0.75)


# legend that matches violin fills
handles = [
    mpatches.Patch(facecolor=mp_color, edgecolor="black", alpha=0.35, label="MP"),
    mpatches.Patch(facecolor=rmp_color, edgecolor="black", alpha=0.55, label="RMP"),
]
ax.legend(handles=handles, loc="upper right")

plt.tight_layout()
dir_ = f"analysis/plot{date.today():%m%d}/"
note = "_vp"
filename =  dir_  +  note  + ".png"
os.makedirs(os.path.dirname(filename), exist_ok=True)
plt.savefig(filename, dpi=200, bbox_inches="tight")
plt.show()

