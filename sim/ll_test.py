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