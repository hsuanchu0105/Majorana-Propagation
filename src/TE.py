import numpy as np
from scipy.linalg import expm
from .Op import *

def Trotterization(exp, init_len, init_maj, U, nf):


    for i in range(2**nf):
        test = np.eye(2**nf)[i]
        rho = np.reshape(test, (2**nf, 1))
        rhoT = np.transpose(rho)

        #print(rho)
        
        H = Maj_to_mtx(init_len, init_maj, nf)
        #print(H)
        #Expect_dir = np.cos(theta) * np.trace(rho @ rhoT @ Mb) + np.sin(theta) * 1j * np.trace(rho @ rhoT @ Mbj @ Mb) + np.cos(theta) * np.trace(rho @ rhoT @ Mc) + np.sin(theta) * 1j * np.trace(rho @ rhoT @ Mbj @ Mc)
        
        for k in range(len(U)):
            M = MajoranaOp(len(U[k][1]), U[k][1]) 
            Mbj = Maj_to_mtx(1, [M], nf)
            theta = U[k][0]
            #print(theta)
            H = expm(1j * theta  *  Mbj/2) @ H @ expm(-1j * theta  *  Mbj/2)
            #print(H)
        
        #print(H)
        #print("diff ", H - H.conj().T)
        Expect_dir = np.trace(rho @ rhoT @ H)
        exp[i] = Expect_dir


def ExpectVal(Input_Node, lenN, rho):
    Expect = 0
    for i in range(lenN):
        Pair = Input_Node[i].BinPair()

        if(Pair[-1] != -1):
            while(len(Pair) < len(rho)):
                Pair = np.append(Pair, 0)
            #print("Pairs = ", Pair)
            PairedOne = np.inner(Pair, rho)           # { i | |n_i> = 1 and (b_{2i}, b_{2i+1}) is paired }
            Expect += ((-1)**PairedOne) * (1j**sum(Pair))* Input_Node[i].c  
            
    
    return Expect    


def DirectExp(init_len, init_maj, v, h, t, idx, nf, nf2):

    Maj_mtx = majorana_matrices(nf)
    
    #for i in range(2**nf):
    test = np.eye(2**nf)[idx]
    rho = np.reshape(test, (2**nf, 1))
    rhoT = np.transpose(rho)

    F = np.zeros((2**nf, 2**nf), dtype = complex)
    V = np.zeros((2**nf, 2**nf), dtype = complex)

    h_ind = np.nonzero(h) #tuple 
    v_ind = np.nonzero(v)
    
    for i in range(len(h_ind[0])):
        m = h_ind[0][i]
        k = h_ind[1][i]
        F += 1j * h[m][k] * (Maj_mtx[m] @ Maj_mtx[k]) 
        
    for i in range(len(v_ind[0])):
        j = v_ind[0][i]
        k = v_ind[1][i]
        l = v_ind[2][i]
        m = v_ind[3][i]
        V += v[j][k][l][m] * (Maj_mtx[j] @ Maj_mtx[k] @ Maj_mtx[l]@ Maj_mtx[m])
        
    H = Maj_to_mtx(init_len, init_maj, nf)
    H = expm(1j * (V+F) * t) @ H @ expm(-1j * (V+F) * t)

    Expect_dir = np.trace(rho @ rhoT @ H, dtype = complex)
    return Expect_dir
        #print("Expectation value by direct exponential = ", Expect_dir)