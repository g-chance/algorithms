class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

class SLL:
    def __init__(self):
        self.head = None
    def addToFront(self, val):
        new_node = Node(val)
        new_node.next = self.head
        # print(new_node.next)
        self.head = new_node
        return self
    def removeFromBack(self):
        runner = myList.head
        while runner.next.next != None:
            runner = runner.next
        runner.next = None
    def addToBack(self, val):
        new_node = Node(val)
        runner = myList.head
        while runner.next != None:
            # if runner.next == None:
            runner = runner.next
        runner.next = new_node
    def insertAt(self, idx, val):
        new_node = Node(val)
        runner = self.head
        if idx == 0:
            self.addToFront(val)
        else:
            i = 0
            while i < idx-1:
                runner = runner.next
                i += 1
            new_node.next = runner.next
            runner.next = new_node
    def removeAt(self, idx):
        runner = self.head
        i = 0
        while i < idx - 1:
            runner = runner.next
            i += 1
        runner.next = runner.next.next
    def KthfromLast(self,num):
        runner = self.head
        runner2 = self.head
        count = 0
        while runner != None:
            runner = runner.next
            if count > num:
                runner2 = runner2.next
            count += 1
        return runner2.val
    
    def reverseList(self):
        newhead = None
        runner2 = self.head.next
        while runner2 != None:
            self.head.next = newhead
            newhead = self.head
            self.head = runner2
            runner2 = runner2.next
        self.head.next = newhead
        while self.head != None:
            print(self.head.val)
            self.head = self.head.next
        return self

class Stack:
    def __init__(self):
        self.head = None
    def pop(self):
        val = self.head.val
        self.head = self.head.next
        return val
    def push(self,val):
        new_node = Node(val)
        new_node.next = self.head
        self.head = new_node
        return self
    def reverseStack(self):
        if self.head != None:
            newstack = Stack()
            while self.head != None:
                newstack.push(self.pop())
        self.head = newstack.head
        return self


myStack = Stack()
myStack.push(5)
myStack.push(4)
myStack.push(3)
myStack.push(2)
myStack.push(1)

runner = myStack.head
while runner != None:
    print("myStack", runner.val)
    runner = runner.next

myStack.reverseStack()

runner = myStack.head
while runner != None:
    print("myStack Reversed", runner.val)
    runner = runner.next

print(myStack)

# myList = SLL()
# myList.addToFront(7)
# myList.addToFront(9)
# myList.addToFront(12)
# myList.addToFront(1)
# myList.addToFront(12)

# runner = myList.head
# while runner != None:
#     print("myList", runner.val)
#     runner = runner.next
# print("Kth", myList.KthfromLast(3))

# print("*"*50)

# print(myList.reverseList())

# runner = myList.head
# while runner != None:
#     print(runner.val)
#     runner = runner.next
# print("REMOVE FROM BACK")
# myList.removeFromBack()
# runner = myList.head
# while runner != None:
#     print(runner.val)
#     runner = runner.next
# print("ADD TO BACK")
# myList.addToBack(30)
# runner = myList.head
# while runner != None:
#     print(runner.val)
#     runner = runner.next
# print("INSERT AT")
# myList.insertAt(5,50)
# runner = myList.head
# while runner != None:
#     print(runner.val)
#     runner = runner.next
# print("REMOVE AT")
# myList.removeAt(3)
# runner = myList.head
# while runner != None:
#     print(runner.val)
#     runner = runner.next