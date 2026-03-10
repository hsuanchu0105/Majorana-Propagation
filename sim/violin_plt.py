import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

cn = 5

mp_color = "tab:blue"
rmp_color = "tab:orange"

d = np.load("analysis/ta0306/393921_71596732_1225638_errors.npz")
err = d["err"]
err_rot = d["err_rot"]
# err, err_rot: shape (cn, n_samples)
# Avoid log(0)
eps = 1e-300
log_err   = np.log10(np.clip(err, eps, None))
log_err_r = np.log10(np.clip(err_rot, eps, None))

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




# after you create vp1 and vp2 from ax.violinplot(...)



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
ax.set_ylabel("log10(Relative error)")
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
plt.show()

