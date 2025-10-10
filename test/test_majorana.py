import unittest
import numpy as np
import cmath 

class TestMajorana(unittest.TestCase):
    def test_Majorana_Prop(self):
        b0 = [1, 1]
        M0 = MajoranaOp(2, b0)
        theta1 = cmath.pi/3
        b1 = [1, 0]
        coeff1, coeff2, bnew = MajoranaPrg(M0, theta1, b1)
