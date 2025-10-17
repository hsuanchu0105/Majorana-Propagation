import cmath 
import numpy as np

"""
Node used for Majorana Propagation. Each node contains one binary representation and coefficient (default as 1)
"""
class Node:
    def __init__(self, b, c = 1):
        self.b = b
        self.c = c
        self.rb = 0
        self.N = len(b)
    #return 0 if is not paired, else return number of pairs
    def IsPaired(self):
        Pair = 0
        for i in range(0, self.N, 2):
            if (self.b[i] + self.b[i+1]) == 1:
                return 0
            elif(self.b[i] + self.b[i+1]) == 2:
                Pair += 1
        return Pair
"""
Majorana Operator 
"""
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
    
        
"""
Majorana Propagation for 1 Fermionic gate
"""
def M1Prg(Nin, theta_ex, b_ex):
    neg_cnt = 0                             #negative sign from anti-commutivity 
    cons_len = min(len(b_ex), len(Nin.b))        #considered length
    
    #sign added by multiplication of two Majorana operators (nodes)
    for i in range(cons_len):
        if(Nin.b[i]==1):
            shade = [0] * (i + 1) + [1] * (len(b_ex) - i - 1)
            neg_cnt += np.inner(b_ex, shade)
    
    sign = 1
    if(neg_cnt % 2 == 1):
        sign = -1

    # put two binaries into same length 
    if(len(b_ex) < len(Nin.b)):
        long_arr = Nin.b
        short_arr = b_ex
    else:
        long_arr = b_ex
        short_arr = Nin.b

    short_padded = np.zeros_like(long_arr)
    short_padded[:short_arr.shape[0]] = short_arr

    
    bsum = short_padded + long_arr
    bout = np.array([x % 2 for x in bsum])

    imag = MajoranaOp(len(b_ex), b_ex).rb() + 1

    c1 = Nin.c * cmath.cos(theta_ex)
    c2 = Nin.c * cmath.sin(theta_ex) *  (1j ** imag) * sign 
    #print(imag, sign, c2)

    return c1,  c2 , bout 

"""
LinkedList for recording Majorana Propagation 
"""
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
            print( currentNode.b, ",", f"{currentNode.c.real:.3f}{currentNode.c.imag:+.3f}j", end=" -> ")
            currentNode = currentNode.next
        print("null")
    def PrintFrom(self, start, end):
        currentNode = self.head
        for i in range(start):
            currentNode = currentNode.next
        for i in range(start, end+1):
            print( currentNode.b, ",", f"{currentNode.c.real:.3f}{currentNode.c.imag:+.3f}j", end=" -> ")
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
      
"""
Main function of Majorana Propagation 
"""
def MajoranaPropagation(trunc, Nin, lenU, U):

    #parameters for truncation
    length_trunc = trunc[0]
    coeff_thres = trunc[1]

    # initial Majorana operator
    PpgList = LinkedList()
    for i in range(len(Nin)):
        PpgList.insertNodeAtPosition(Nin[i], i)
    
    #parameters of Fermionic circuit U
    L = lenU                   #length of U
    
    
    lv_st = 0               #start of current level 
    lv_end = 1              #end of current level 
    current_pos = 1
    print("length threshold = ", length_trunc, ", coefficient threshold = ", coeff_thres)
    print("Level 0 :")
    PpgList.PrintFrom(lv_st, lv_end)

    for i in range(L):
        for j in range(lv_st, lv_end + 1):
            if(len(PpgList[j].b) < len(U[i][1])):
                long_arr = U[i][1]
                short_arr = PpgList[j].b
            else:
                long_arr = PpgList[j].b
                short_arr = U[i][1]

            short_padded = np.zeros_like(long_arr)
            short_padded[:short_arr.shape[0]] = short_arr

            if(np.inner(short_padded, long_arr) % 2 == 0): #if M_b and M_{b_j} commute
                #pass
                N = Node(PpgList[j].b, PpgList[j].c)
                PpgList.insertNodeAtPosition(N, current_pos + 1)
                current_pos += 1
            else:
                coeff1, coeff2, bnew = M1Prg(PpgList[j], U[i][0], U[i][1])
                #print(coeff2)
                #print(PpgList[j].b)
                Nl = Node(PpgList[j].b, coeff1) 
                Nr = Node(bnew, coeff2)
                PpgList.insertNodeAtPosition(Nl, current_pos + 1)
                if(sum(bnew) <= length_trunc and np.abs(coeff2) > coeff_thres): #length truncation and coefficient truncation 
                    PpgList.insertNodeAtPosition(Nr, current_pos + 2)
                    current_pos += 1
                current_pos += 1
        #PpgList.traverseAndPrint()
        #print("length = ", PpgList.len)
        lv_st = lv_end + 1
        lv_end = current_pos
        print("Level", i+1, ":")
        PpgList.PrintFrom(lv_st, lv_end)

    #compute expectation value
    Expect = 0
    for i in range(lv_st, lv_end + 1):
        if(PpgList[i].IsPaired()):
            Paircnt = PpgList[i].IsPaired()
            print("Paired nodes = ", PpgList[i].b)
            print("number of pairs = ", Paircnt)
            Expect += (1j**Paircnt) * PpgList[i].c 

    print("Expectation value calculated by Majorana Propagation = ", Expect)      



trunc_param = np.array([4, 1e-4])
b1 = np.array([1, 1])
M1 = MajoranaOp(2, b1)
N1 = Node(b1, 1j**M1.rb())
b2 = np.array([1, 0, 1, 0])
M2 = MajoranaOp(4, b2)
N2 = Node(b2, 1j**M2.rb())
Init_Node = [N1, N2]
theta1 = cmath.pi/3
theta2 = cmath.pi/6
theta3 = cmath.pi/3
b1 = np.array([1, 0, 0, 1, 1, 1])
b2 = np.array([0, 0, 1, 1])
b3 = np.array([0, 0, 1, 0, 0, 0, 1, 1, 1 , 0])

U = [[theta1, b1], [theta2, b2], [theta3, b3]]
MajoranaPropagation(trunc_param, Init_Node, 3, U)