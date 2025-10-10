import cmath 
import numpy as np

class MajoranaOp:
    
    def __init__(self, N, b):
        self.N = N              # 2N in paper
        self.b = b
    def rb(self):
        w = sum(self.b)
        if(w % 4 == 0 or (w-1) % 4 ==0):
            return 0
        else:
            return 1

#Majorana Propagation 
def MajoranaPrg(Min, theta_ex, b_ex):
    w = sum(Min.b)              # 1 norm 
    w_ex = sum(b_ex)
    neg_cnt = 0                 #shuffle count 
    cons_len = len(b_ex)        #considered length
    if(len(b_ex) > Min.N):
        cons_len = Min.N
    for i in range(cons_len):
        if(b_ex[i]==1):
            shade = [0] * b_ex[i] + [1] * (Min.N - b_ex[i])
            neg_cnt += np.inner(Min.b, shade)
    
    bsum = np.add(Min.b,b_ex)
    bout = [x % 2 for x in bsum]

    sign = 1
    if(neg_cnt % 2 == 1):
        sign = -1
    
    imag = Min.rb() + MajoranaOp(len(b_ex), b_ex).rb() + 1

    c1 = cmath.cos(theta_ex)
    c2 = cmath.sin(theta_ex) *  (1j ** imag) * sign 

    return c1,  c2 , bout 

b0 = [1, 1, 0, 1, 1, 0, 0, 0, 1, 0]
M0 = MajoranaOp(10, b0)
theta1 = 0.3
b1 = [0, 0, 0, 1, 0, 0, 0, 0, 1, 1]
coeff1, coeff2, bnew = MajoranaPrg(M0, theta1, b1)

print(coeff1, coeff2, bnew)

