import numpy as np
from scipy.linalg import expm
from src.MajProp import * 

# test for tensor V contraction 
def majorana_matrices_rot(R, nf):
    rot_Maj = []
    A = np.array(majorana_matrices(nf))
    for i in range(nf * 2):
        rot_Maj.append(np.tensordot(R[i], A, axes=(0, 0)))
    return rot_Maj


nf = 6
nf2 = 12
h = np.zeros((nf2, nf2))

h[2][0] = -0.8640217711605882
h[5][1] = -0.039327907713503585
h[5][2] = -0.5583484414930555
h[5][3] = 0.397408193713201
h[8][2] = -0.8521909977163846
h[9][6] = -0.8402103310959215
h[9][7] = -0.09762133497071956
h[11][0] = -0.8614782934095213
h[11][10] = 0.46809750664769667

for i in range(nf2):
    for j in range(i+1, nf2):
        h[i][j] = -h[j][i]


V = np.zeros((12, 12, 12, 12))

V[0][1][3][9] = -0.7319375671312152
V[0][1][4][5] = 0.886824359840124
V[0][3][5][7] = 0.4784136494071112
V[0][6][9][10] = -0.12878381918615278
V[1][2][8][9] = -0.14131477740532583
V[2][3][4][8] = 0.1853154548499245
V[2][3][4][10] = -0.7257990377879862
V[2][7][8][10] = -0.535180162536246
V[2][7][9][10] = -0.7185340975860877
V[4][5][8][9] = -0.2874212054575591
V[5][6][10][11] = -0.2726582598103642


dt = 0.01
R = expm(4 * h * dt)
#print("h = ", h)
#print("R = ", R)
Rt = np.transpose(R)
#print(np.allclose(R, np.transpose(R))) # Why do changing R into Rt not change the result?
V1 = np.einsum("jklm, jn -> nklm", V, Rt)
V2 = np.einsum("nklm, ko -> nolm", V1, Rt)
V3 = np.einsum("nolm, lp -> nopm", V2, Rt)
V4 = np.einsum("nopm, mq -> nopq", V3, Rt)

# before contraction 
Vmtx = np.zeros((2**nf, 2**nf), dtype = complex)
v_ind = np.nonzero(V)
    
Maj_mtx = majorana_matrices(6)

for i in range(len(v_ind[0])):
    j = v_ind[0][i]
    k = v_ind[1][i]
    l = v_ind[2][i]
    m = v_ind[3][i]
    Vmtx += V[j][k][l][m] * (Maj_mtx[j] @ Maj_mtx[k] @ Maj_mtx[l]@ Maj_mtx[m])


Vmtx_r = np.zeros((2**nf, 2**nf), dtype = complex)
v_ind_rot = np.nonzero(V4)

Rot_Maj_mtx = majorana_matrices_rot(R, 6)


for i in range(len(v_ind_rot[0])):
    j = v_ind_rot[0][i]
    k = v_ind_rot[1][i]
    l = v_ind_rot[2][i]
    m = v_ind_rot[3][i]
    Vmtx_r += V4[j][k][l][m] * (Rot_Maj_mtx[j] @ Rot_Maj_mtx[k] @ Rot_Maj_mtx[l] @ Rot_Maj_mtx[m])


print(Vmtx)
print(Vmtx_r)
print(np.allclose(Vmtx, Vmtx_r))