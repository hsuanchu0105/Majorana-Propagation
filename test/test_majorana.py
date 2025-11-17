import unittest
import numpy as np
import cmath 
import sys
sys.path.insert(1, '/mnt/c/Users/faceb/OneDrive/Desktop/TUM_CSE/Guided_research/code')
import sim 

"""
a1 = np.array([1, 1])
a2 = np.array([1, 0, 1, 0])
M1 = MajoranaOp(2, a1)
M2 = MajoranaOp(4, a2)
N1 = Node(a1, 1j**M1.rb())
N2 = Node(a2, 1j**M2.rb())

#Init_Node = [N1, N2]
#init_len = 2
#init_maj = [M1, M2]

a2 = np.array([0, 1, 1, 0, 1, 0])
M2 = MajoranaOp(6, a2)
N2 = Node(a2, 1j**M2.rb())

Init_Node = [N2]
init_len = 1
init_maj = [M2]

theta1 = cmath.pi/7
theta2 = cmath.pi/6
theta3 = cmath.pi/3
b1 = np.array([1, 0, 0, 1, 1, 1])
b2 = np.array([0, 1, 1, 1, 0, 0])
b3 = np.array([0, 0, 1, 0, 0, 0])

#U_wid = 3
#U = [[theta1, b1], [theta2, b2], [theta3, b3]]


theta1 = 0.45
theta2 = 0.53
theta3 = 1.05
b1 = np.array([1, 1, 0, 0, 1, 0])
b2 = np.array([0, 0, 0, 1, 1, 1])
b3 = np.array([0, 1, 0, 1, 0, 1])
U_wid = 1
#U = [[theta1, b1], [theta2, b2], [theta3, b3]]
U = [[theta3, b3]]

#theta1 = 0.45

#b1 = np.array([1, 0, 0, 1, 1, 1])

#U_wid = 1
#U = [[theta1, b1]]
"""

'''
init_len = 1
print(Maj2.b)
H = Maj_to_mtx(init_len, [Maj2])

    #Expect_dir = np.cos(theta) * np.trace(rho @ rhoT @ Mb) + np.sin(theta) * 1j * np.trace(rho @ rhoT @ Mbj @ Mb) + np.cos(theta) * np.trace(rho @ rhoT @ Mc) + np.sin(theta) * 1j * np.trace(rho @ rhoT @ Mbj @ Mc)

for k in range(len(U)):
    M = MajoranaOp(len(U[k][1]), U[k][1]) 
    Mbj = Maj_to_mtx(1, [M])
    #print("Mbj = ", Mbj)
    theta = U[k][0]
    #print(theta)
    H = expm(1j * theta  *  Mbj/2) @ H @ expm(-1j * theta  *  Mbj/2)
    print(H)

print("difference 2 = ", Maj_output - H)
'''

class TestMajorana(unittest.TestCase):
    def test_Majorana_Prop1(self):
        b0 = np.array([1, 1])
        M0 = sim.MajoranaOp(len(b0), b0)
        theta1 = cmath.pi/3
        b1 = np.array([1, 0])
        N0 = sim.Node(b0, 1j**(M0.rb()))
        coeff1, coeff2, bnew = sim.M1Prg(N0, theta1, b1)
        self.assertTrue(np.allclose(coeff1, 1j * cmath.cos(theta1)), msg="coeff1 does not match reference")
        self.assertTrue(np.allclose(coeff2, cmath.sin(theta1)), msg="coeff2 does not match reference")
        self.assertTrue(np.allclose(bnew, [0, 1]), msg="binary does not match reference")
    def test_Majorana_Prop2(self):
        b0 = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        M0 = sim.MajoranaOp(len(b0), b0)
        theta1 = cmath.pi/3
        b1 = np.array([1, 1])
        N0 = sim.Node(b0, 1j**(M0.rb()))
        coeff1, coeff2, bnew = sim.M1Prg(N0, theta1, b1)
        self.assertTrue(np.allclose(coeff1, cmath.cos(theta1)), msg="coeff1 does not match reference")
        self.assertTrue(np.allclose(coeff2, cmath.sin(theta1)), msg="coeff2 does not match reference")
        self.assertTrue(np.allclose(bnew, [0, 0, 0, 0, 1, 0, 1, 0]), msg="binary does not match reference")
    

if __name__ == '__main__':
    unittest.main()
