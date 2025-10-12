import cmath 
import numpy as np


class Node:
    def __init__(self, bin, coeff = 1):
        self.b = bin
        self.c = coeff
        
"""
Majorana Operator
"""
class MajoranaOp(Node):   
    def __init__(self, N, b, c = 1):
        Node.__init__(self, b, c)
        self.N = N              # 2N in paper
    def rb(self):
        w = sum(self.b)
        if(w % 4 == 0 or w % 4 == 1):
            return 0
        else:
            return 1


        
"""
Majorana Propagation (1 Fermionic gate)
"""
def M1Prg(Min, theta_ex, b_ex):
    neg_cnt = 0                             #negative sign from anti-commutivity 
    cons_len = min(len(b_ex), Min.N)        #considered length
    
    for i in range(cons_len):
        if(b_ex[i]==1):
            shade = [0] * (i + 1) + [1] * (Min.N - i - 1)
            neg_cnt += np.inner(Min.b, shade)
    
    if(neg_cnt % 2 == 1):
        sign = -1

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

    
    
    imag = Min.rb() + MajoranaOp(len(b_ex), b_ex).rb() + 1

    c1 = cmath.cos(theta_ex)
    c2 = cmath.sin(theta_ex) *  (1j ** imag) * sign 

    return c1,  c2 , bout 

class LinkedList:
    def __init__(self):
        self.head = None
    def insertNodeAtPosition(self, newNode, position):
        if position == 1:
            newNode.next = self.head
            self.head =  newNode
        
        currentNode = self.head
        for _ in range(position - 2):
            if currentNode is None:
                break
            currentNode = currentNode.next

        newNode.next = currentNode.next
        currentNode.next = newNode
        
    def deleteSpecificNode(self, nodeToDelete):
        if self.head == nodeToDelete:
            self.head = None

        currentNode = self.head
        while currentNode.next and currentNode.next != nodeToDelete:
            currentNode = currentNode.next

        if currentNode.next is None:
            currentNode = None
        else:
            currentNode.next = currentNode.next.next

        

class MajoranaPropagation:
    length_trunc = 4
    coeff_thres = 1e-4
    L = 3
    N1 = Node(np.array([1, 1]))
    N2 = Node(np.array([1, 0, 1, 0]))
    PpgList = LinkedList()
    PpgList.insertNodeAtPosition(N1, 1)
    
    theta1 = cmath.pi/3
    theta2 = cmath.pi/4
    theta3 = cmath.pi/6
    b1 = np.array([1, 0, 0, 1, 1, 1])
    b2 = np.array([0, 0, 1, 1])
    b3 = np.array([0, 0, 1, 0, 0, 0, 1, 1,1 ,0])
    U = [[theta1, b1], [theta2, b2], [theta3, b3]]

    
    lv_st = 1
    lv_end = 2
    for i in range(L):
        M1Prg()


