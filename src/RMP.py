import numpy as np
from scipy.linalg import expm
from .MajProp import *

# First change H' into new basis, then write in the form of fermionic gates
def BasisChange(N, h, V, dt, trott_order, trunc_param, bdry):
	# N: number of Fermionic mode
	# h: free-fermion Hamiltonian coefficient (2N * 2N matrix)
    # V: 4-leg tensor 
    # dt: time per timestep
    # trott_order: trotterization order 
	# output: tensor after contraction with R^T, resulting fermionic gate
    
    #print("input non zero V = ", V.nonzero())

    # find a way to check the correctness of contraction
    if(bdry):
        R = expm(2 * h * dt)
    else:
        R = expm(4 * h * dt)
    #print("h = ", h)
    #print("R = ", R)
    Rt = np.transpose(R)
    #print(np.allclose(R, np.transpose(R))) # Why do changing R into Rt not change the result?
    V1 = np.einsum("jklm, jn -> nklm", V, Rt)
    V2 = np.einsum("nklm, ko -> nolm", V1, Rt)
    V3 = np.einsum("nolm, lp -> nopm", V2, Rt)
    V4 = np.einsum("nopm, mq -> nopq", V3, Rt)
	
    coef_sh = V4.shape

    #print("non zero terms of rotated V = ", V4.nonzero())


    U = []
    if(trott_order == 2):
        for k in range(coef_sh[0]):
            for l in range(coef_sh[1]):
                for m in range(coef_sh[2]):
                    for n in range(coef_sh[3]):
                        b = np.zeros(2 * N)
                        if(V4[k][l][m][n] > trunc_param[1]): # maybe apply a threshold for coefficient
                            
                            theta = V4[k][l][m][n] * dt  #second order trotterization
                            b[k] += 1
                            b[l] += 1
                            b[m] += 1
                            b[n] += 1
                            for q in range(2 * N):
                                b[q] = b[q] % 2
                            
                            theta = theta * perm_parity(k, l , m, n)
                            U.append([theta, b])

        for i in range(len(U)-1, -1, -1):
            U.append(U[i])

    elif(trott_order == 1):
        for k in range(coef_sh[0]):
            for l in range(coef_sh[1]):
                for m in range(coef_sh[2]):
                    for n in range(coef_sh[3]):
                        b = np.zeros(2 * N)
                        if(V4[k][l][m][n] > trunc_param[1]):
                            
                            theta = 2 * V4[k][l][m][n] * dt  
                            b[k] += 1
                            b[l] += 1
                            b[m] += 1
                            b[n] += 1
                            for q in range(2 * N):
                                b[q] = b[q] % 2
                            
                            theta = theta * perm_parity(k, l , m, n)
                            U.append([theta, b])

    
    return V4, U

def twofourMajStrEvo(N, h, V, n, dt, Init_Node, trunc_param, trott_order, histsave = False):
	# N: number of Fermionic mode
	# h: free-fermion Hamiltonian coefficient (2N * 2N matrix)
    # V: 4-leg tensor 
    # n: number of timesteps
    # dt: evolution time each timestep 
	# Initial Node: 
	# output: coefficient of majorana operator after evolution
	
    Node_next = Init_Node

    bdry = False
    for i in range(n):
        if(i==0):
            bdry = True
        V, U = BasisChange(N, h, V, dt, trott_order, trunc_param, bdry) #coefficient in new basis
        #print("V_update= ", V)
        #print("Fermionic gate (V)", U)
        #print(f"gate count at step {i}: {len(U)}")
        Node_next = MajoranaPropagation(trunc_param, Node_next, len(U), U, histsave, "timestep" + str(i))
        #print(len(Node_next))

    return Node_next

def Rotated_ExpectVal(NodeList, h, dt, tstep_num, rho, nf):
    
    R0 = expm(2 * h * dt)
    R = expm(4 * h * dt)
    Rm = R0
    for i in range(tstep_num - 1):
        Rm = R @ Rm
    Rm = R0 @ Rm

    #Rm = expm(4 * h * dt)

    Exp = 0
    #'''
    for node in NodeList:
        if(np.sum(node.b) % 2 == 0):
            #print(node)
            bin = node.b
            coeff = node.c
            m = int(np.sum(bin)/2)
            #print("m=", m)
            for combo in combinations(list(range(nf)), m):
                J = list(combo)
                Js = [x for j in J for x in (2*j, 2*j + 1)]
                A = np.nonzero(bin)[0] 
                Q = Rm[np.ix_(A, Js)]
                Exp+= coeff * np.linalg.det(Q) * np.prod(1j * (1 - 2 * rho[J] ))
                #contr = coeff * np.linalg.det(Q) * np.prod(1j * (1 - 2 * rho[J] ))
                #print(Js, A, contr)
        #print('\t')
    return Exp
    #'''