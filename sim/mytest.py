import numpy as np
import unittest
from scipy.linalg import expm
"""
class Node:
    def __init__(self, c = 1):
        self.c = c


class LinkedList:
    def __init__(self):
        self.head = None
        self.len = 0
    def insertNodeAtPosition(self, newNode, position):
        if position == 0:
            newNode.next = self.head
            self.head =  newNode
        elif position > 0:    
            currentNode = self.head
            for _ in range(position - 1):
                if currentNode is None:
                    break
                currentNode = currentNode.next

            newNode.next = currentNode.next
            currentNode.next = newNode
        self.len +=1 
    def deleteSpecificNode(self, nodeToDelete):
        if self.head == nodeToDelete:
            self.head = self.head.next

        else:
            currentNode = self.head
            while currentNode.next and currentNode.next != nodeToDelete:
                currentNode = currentNode.next

            if currentNode.next is None:
                currentNode = None
            else:
                currentNode.next = currentNode.next.next
        self.len -= 1
    def traverseAndPrint(self):
        currentNode = self.head
        while currentNode:
            print(currentNode.c, end=" -> ")
            currentNode = currentNode.next
        print("null")
    def __getitem__(self, position):
        if(position ==0):
            return self.head
        else:
            currentNode = self.head
            for i in range(position, 0, -1):
                currentNode = currentNode.next
            return currentNode
        

tl = LinkedList()
N1 = Node(5)
N2 = Node(6)
N3 = Node(7)
N4 = Node(-1)
tl.insertNodeAtPosition(N1, 0)
tl.traverseAndPrint()
tl.insertNodeAtPosition(N2, 1)
tl.traverseAndPrint()
tl.insertNodeAtPosition(N3, 1)
tl.traverseAndPrint()
tl.insertNodeAtPosition(N4, 3)
tl.traverseAndPrint()
print(tl[1].c)
tl.deleteSpecificNode(N2)
tl.traverseAndPrint()
"""
# Pauli gates 
X = np.array([[0, 1], [1, 0]])
Y = 1j * np.array([[0, -1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)

m1 = np.kron(np.kron(X, I), I)
m2 = np.kron(np.kron(Y, I), I)
m3 = np.kron(np.kron(Z, X), I)
m4 = np.kron(np.kron(Z, Y), I)
m5 = np.kron(np.kron(Z, Z), X)
m6 = np.kron(np.kron(Z, Z), Y)

Maj_mtx = [m1, m2, m3, m4, m5, m6]

class MajoranaOp:   
    def __init__(self, N, b):
        self.b = b 
        self.N = N                          # 2N in paper
    def rb(self):
        w = sum(self.b)
        if(w % 4 == 0 or w % 4 == 1):
            return 0
        else:
            return 1
    
def Maj_to_mtx(len, MajIniList):
    

    mtx = np.zeros((2 ** 3, 2**3))
    for i in range(len):
        MajOp = MajIniList[i]
        x = np.eye(2**3) * (1j ** MajOp.rb())
        for j in range(MajOp.N):
            if(MajOp.b[j]==1):
                x = x @ Maj_mtx[j]
        mtx = mtx + x
    return mtx 


class TestMajorana(unittest.TestCase):
    def test_matrix_transfer(self):
        b1 = np.array([1, 1, 0, 0, 1, 0])
        b2 = np.array([0, 0, 0, 1, 1, 1])
        b3 = np.array([0, 1, 0, 1, 0, 1])
        M1 = MajoranaOp(6, b1)
        M2 = MajoranaOp(6, b2)
        M3 = MajoranaOp(6, b3)
        mtx1 = Maj_to_mtx(1, [M1])
        mtx2 = Maj_to_mtx(1, [M2])
        mtx3 = Maj_to_mtx(1, [M3])
        #print(mtx2)
        #print("check2 = " , 1j* m4 @ m5 @ m6)
        #print("diff = " , mtx2 - 1j* m4 @ m5 @ m6)

        #print("difference = ", mtx - 1j * m1 @ m3)
        self.assertTrue(np.allclose(mtx1, 1j* m1@ m2 @ m5), msg="M1 transfer to matrix error")
        self.assertTrue(np.allclose(mtx2, 1j* m4 @ m5 @ m6), msg="M2 transfer to matrix error")
        self.assertTrue(np.allclose(mtx3, 1j* m2 @ m4 @ m6), msg="M3 transfer to matrix error")

    def test_case_1(self):
        b = np.array([1, 0, 1, 0])
        M = MajoranaOp(4, b)
        Min = Maj_to_mtx(1, [M])
        b1 = np.array([1, 1, 0, 0, 1, 0])
        b2 = np.array([0, 0, 0, 1, 1, 1])
        b3 = np.array([0, 1, 0, 1, 0, 1])
        M1 = MajoranaOp(6, b1)
        M2 = MajoranaOp(6, b2)
        M3 = MajoranaOp(6, b3)
        theta1 = 0.45
        theta2 = 0.53
        theta3 = 1.05
        mtx1 = Maj_to_mtx(1, [M1])
        mtx2 = Maj_to_mtx(1, [M2])
        mtx3 = Maj_to_mtx(1, [M3])
        print(mtx1)
        print(mtx2)
        print(mtx3)
        U1 = expm(1j * theta1 * mtx1/2)
        U2 = expm(1j * theta2 * mtx2/2)
        U3 = expm(1j * theta3 * mtx3/2)
        test = np.eye(8)[0]
        rho = np.reshape(test, (8, 1))
        rhoT = np.transpose(rho)
        #print(U1 @ Min @ U1.conj().T)
        #print(U2 @ U1 @ Min @ U1.conj().T @ U2.conj().T )
        H = U3 @ U2 @ U1 @ Min @ U1.conj().T @ U2.conj().T  @ U3.conj().T
        #print(H)
        Expectation = np.trace(rho @ rhoT @  H)
        print(Expectation)


if __name__ == '__main__':
    unittest.main()