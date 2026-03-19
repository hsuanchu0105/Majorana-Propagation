from src.MajProp import *
from src.Gate import *
from src.RMP import *
from src.Err_anlys import *
from src.TE import *
from src.Plot import *
import time 





dt = 0.01
n = 2 #num of timestep

#number of fermionic mode 
nf = 6
nf2 = 2 * nf


h = np.zeros((nf2, nf2))

seed_h = 12345
seed_v = 25910

size1 = 6
size2= 7
rng1 = np.random.default_rng(seed_h)
vals = rng1.uniform(-1, 1, size=size1)
'''
h[2][0] = -0.8640217711605882
h[5][1] = -0.039327907713503585
h[5][2] = -0.5583484414930555
h[5][3] = 0.397408193713201
h[8][2] = -0.8521909977163846
h[9][6] = -0.8402103310959215
h[9][7] = -0.09762133497071956
h[11][0] = -0.8614782934095213
h[11][10] = 0.46809750664769667
'''

h[2][0] = -0.8
h[5][1] = -0.03

for i in range(nf2):
    for j in range(i+1, nf2):
        h[i][j] = -h[j][i]

#print(h)


rng2 = np.random.default_rng(seed_v)
vals2 = rng2.uniform(-1, 1, size=size2)
V = np.zeros((nf2, nf2, nf2, nf2))
'''
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
'''



V[0][3][5][7] = 0.4
V[0][6][9][10] = -0.1





bin1 = np.array([1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1])
#bin1 = np.array([1, 1, 0, 0, 1, 1, 0, 0])
M1= MajoranaOp(nf2, bin1)
N1 = Node(bin1, 1j**M1.rb())

bin2 = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0])
#bin2 = np.array([0, 0, 1, 1, 0, 0, 0, 0])
M2= MajoranaOp(nf2, bin2)
N2 = Node(bin2, 1j**M2.rb())

bin3 = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1])
#bin3 = np.array([0, 0, 1, 1, 1, 1, 0, 0])
M3= MajoranaOp(nf2, bin3)
N3 = Node(bin3, 1j**M3.rb())

bin4 = np.array([0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1])
M4= MajoranaOp(nf2, bin4)
N4 = Node(bin4, 1j**M4.rb())

bin5 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0])
M5= MajoranaOp(nf2, bin5)
N5 = Node(bin5, 1j**M5.rb())

bin6 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1])
M6= MajoranaOp(nf2, bin6)
N6 = Node(bin6, 1j**M6.rb())

Init_Node = [N1, N2, N3, N4, N5, N6]
init_len = len(Init_Node)
init_maj = [M1, M2, M3, M4, M5, M6]

#Init_Node = [N1]
#init_len = len(Init_Node)
#init_maj = [M1]

h_ind = np.nonzero(h)
v_ind = np.nonzero(V)



#sps = sparsity(h_ind, v_ind, nf2)
#print("Delta = ", sps)
trott_order = 1

# the comparison can be written into more efficient way

    


#print(np.allclose(len(U), (n+1) * np.count_nonzero(h) + 2 * n * np.count_nonzero(V)))

#tc_st = 2
#tc_end = 9
#rel_mp_global = np.zeros(tc_end - tc_st)
#rel_rot_global = np.zeros(tc_end - tc_st)

ts_st = 1
mp = np.zeros(n - ts_st)
rmp = np.zeros(n - ts_st)
ana = np.zeros(n - ts_st)


#rel_mp = np.zeros(n - ts_st)
#rel_rmp = np.zeros(n - ts_st)

#coef_st = -2
#coef_end = -10
#rel_mp_global = np.zeros(coef_st - coef_end)
#rel_rot_global = np.zeros(coef_st - coef_end)

for ts in range(1, n):
#for tc_len in range(tc_st, tc_end):
#for coef_tr in range(coef_st, coef_end, -1):
    #trunc_param = np.array([tc_len, 1e-6])
    U = []
    AppendH(U, h, h_ind, dt, trott_order, nf2)
    AppendV(U, V, v_ind, dt, trott_order, nf2)

    for i in range(ts-1):
        AppendH(U, h, h_ind, 2 * dt, trott_order, nf2)
        AppendV(U, V, v_ind, dt, trott_order, nf2)

    AppendH(U, h, h_ind, dt, trott_order, nf2)
    print("gate count", len(U))

    trunc_param = np.array([8, 1e-6])
    #trunc_param = np.array([4, 10**(coef_tr)])

    

    #print("Fermionic gate:", U)
    #print('\t')
    rho_st = np.zeros(nf, dtype = int)
    c = np.zeros(nf, dtype = int)

    tic = time.perf_counter()
    Output_Node = MajoranaPropagation(trunc_param, Init_Node, len(U), U)
    toc = time.perf_counter()
    print(f"Majorana Propagation : {toc - tic:0.4f} seconds")

    # Rotated Majorana Propogation
    #tic = time.perf_counter()
    Node_out = twofourMajStrEvo(nf, h, V, ts, dt, Init_Node, trunc_param, trott_order)
    #toc = time.perf_counter()
    #print(f"Rotated Evolution : {toc - tic:0.4f} seconds")

    for node in Output_Node:
        #print(sum(node.b))
        pass


    obexp = np.zeros(2**nf, dtype = complex)
    rexp = np.zeros(2**nf, dtype = complex)
    dexp = np.zeros(2**nf, dtype = complex)

    # transform into binary representation |n> = |n_1 n_2 n_3 \cdots n_{nf} >
    for i in range(2**nf):
        rem = i
        for j in range(nf):
            c[j] = rem / 2**(nf - j - 1)
            rem = rem - c[j] * 2**(nf - j - 1)

        for j in range(nf):
            rho_st[j] = c[j]
        #print("input Fock state = ", rho_st)
        
        
        obexp[i] = ExpectVal(Output_Node, len(Output_Node), rho_st)
        #print("Expectation value by Majorana Propagation = ", obexp[i])

        rexp[i] = Rotated_ExpectVal(Node_out, h, dt, ts, rho_st, nf)
        #print("Expectation value by Rotated Majorana Propagation = ", rexp[i])

        dexp[i] = DirectExp(init_len, init_maj, V, h, ts * dt, i, nf, nf2)
        


    
    #Trotterization(dexp, init_len, init_maj, U, nf)

    # 2-norm 
    
    #rel_mp_global[tc_len - tc_st] = np.linalg.norm(obexp - dexp) / np.linalg.norm(dexp)
    #rel_rot_global[tc_len - tc_st] = np.linalg.norm(rexp - dexp) / np.linalg.norm(dexp)
    #rel_mp_global[coef_tr - coef_st] = np.linalg.norm(obexp - dexp) / np.linalg.norm(dexp)
    #rel_rot_global[coef_tr - coef_st] = np.linalg.norm(rexp - dexp) / np.linalg.norm(dexp)
    mp[ts - ts_st] = np.linalg.norm(obexp)
    rmp[ts - ts_st] = np.linalg.norm(rexp)
    ana[ts - ts_st] = np.linalg.norm(dexp)
    #rel_mp[ts - ts_st] = np.linalg.norm(obexp - dexp)/np.linalg.norm(dexp)
    #rel_rmp[ts - ts_st] = np.linalg.norm(rexp - dexp)/np.linalg.norm(dexp)
    
    ErrorPrint(dexp, obexp, rexp, 2)


# truncation method + dt + n + init_maj_len + nonzero_term_h + nonzero+term_v + note(option) 

dir_ = f"analysis/plot{date.today():%m%d}/"
note = ""
filename =  dir_ + "_" + str(nf) + "_" + str(dt) + "_" + str(n) + "_" + str(init_len)+ "_" + str(np.count_nonzero(h)) + "_" + str(np.count_nonzero(V)) + note  + ".png"



x = np.arange(1, n)
comp1 = mp
comp2 = rmp
comp3 = ana 
comps = [comp1, comp2, comp3]
x_label = "time steps"
y_label = "time dynamics"
labels = ["MP", "RMP", "Analytic"]

comp_plot(
    x,
    comps,
    x_label,
    y_label,
    filename,
    labels,
    saveOpt=True,
    logx=False,
    logy=True,
    markers=None,
    linestyles=None,
)

'''
ts_len = np.arange(ts_st, n)  
plt.figure()
plt.plot(ts_len, rel_mp , marker='o', linestyle='-', label='MP')
plt.plot(ts_len, rel_rmp , marker='o', linestyle='-', label='RMP')



#plt.xlabel('truncation length')
plt.xlabel('timestep')
plt.ylabel('relative error (global)')
#plt.ylabel(r'$\left\|O(t)\right\|_{2}$')
plt.yscale('log')  
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()


os.makedirs(os.path.dirname(filename), exist_ok=True)
plt.savefig(filename, dpi=200, bbox_inches="tight")
plt.show()


trunc_met = "lt"
dir = "plot0212/"
note = "_td4"
filename =  dir + trunc_met + "_" + str(nf) + "_" + str(dt) + "_" + str(n) + "_" + str(init_len)+ "_" + str(np.count_nonzero(h)) + "_" + str(np.count_nonzero(V)) + note  + ".png"

plt.figure()
plt.plot(ts_len, mp , marker='o', linestyle='-', label='MP')
plt.plot(ts_len, rmp , marker='o', linestyle='-', label='RMP')
plt.plot(ts_len, ana , marker='o', linestyle='-', label='ANA')
#plt.xlabel('truncation length')
plt.xlabel('timestep')
#plt.ylabel('relative error (global)')
plt.ylabel(r'$\left\|O(t)\right\|_{2}$')
plt.yscale('log')  
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()


os.makedirs(os.path.dirname(filename), exist_ok=True)
plt.savefig(filename, dpi=200, bbox_inches="tight")
plt.show()
'''