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
    neg_cnt = 0                             #negative sign from anti-commutivity 
    cons_len = min(len(b_ex), Min.N)        #considered length
    
    for i in range(cons_len):
        if(b_ex[i]==1):
            shade = [0] * i + [1] * (Min.N - i)
            neg_cnt += np.inner(Min.b, shade)
    

    if(len(b_ex) < Min.N):
        long_arr = Min.b
        short_arr = b_ex
    else:
        long_arr = b_ex
        short_arr = Min.b

    short_padded = np.zeros_like(long_arr)

    # Copy the elements of the smaller array into the padded array
    short_padded[:short_arr.shape[0]] = short_arr

    # Add the two arrays of the same size
    bsum = short_padded + long_arr
    bout = [x % 2 for x in bsum]

    sign = 1
    if(neg_cnt % 2 == 1):
        sign = -1
    
    imag = Min.rb() + MajoranaOp(len(b_ex), b_ex).rb() + 1

    c1 = cmath.cos(theta_ex)
    c2 = cmath.sin(theta_ex) *  (1j ** imag) * sign 

    return c1,  c2 , bout 

b0 = np.array([1, 1, 0, 1, 1, 0, 0, 0, 1, 0])
#b0 = np.array([1, 1])
M0 = MajoranaOp(len(b0), b0)
theta1 = cmath.pi/3
b1 = np.array([0, 0, 0, 1, 0, 0, 0, 0])
#b1 = np.array([1])
coeff1, coeff2, bnew = MajoranaPrg(M0, theta1, b1)

print(coeff1, coeff2, bnew)

