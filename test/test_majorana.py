import unittest
import numpy as np
import cmath 
import sys
sys.path.insert(1, '/mnt/c/Users/faceb/OneDrive/Desktop/TUM_CSE/Guided_research/code')
import sim 


class TestMajorana(unittest.TestCase):
    def test_Majorana_Prop(self):
        b0 = np.array([1, 1])
        M0 = sim.MajoranaOp(len(b0), b0)
        theta1 = cmath.pi/3
        b1 = np.array([1, 0])
        coeff1, coeff2, bnew = sim.MajoranaPrg(M0, theta1, b1)
        self.assertTrue(np.allclose(coeff1, cmath.cos(theta1)), msg="coeff1 does not match reference")
        self.assertTrue(np.allclose(coeff2, cmath.sin(theta1)), msg="coeff2 does not match reference")
        self.assertTrue(np.allclose(bnew, [0, 1]), msg="binary does not match reference")


if __name__ == '__main__':
    unittest.main()
