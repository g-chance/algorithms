// Names: Aaron Gray, Anmol Duggal, Yusuf Ozgen, Greg Chance

// doubly linked node
class DLNode {
    
    constructor(data) {
        this.data = data;
        this.next = null;
        this.prev = null;
    }
}

class DLL {

    constructor() {
        this.head = null;
        this.tail = null;
    }

    push(data) {
        var node = new Node(data);
        if(this.head === null) {
            this.head = node;
            this.tail = node;
        } else {
            node.next = this.head;
            this.head.prev = node;
            this.head = node;
        }
    }
    append(data) {
        let newNode = new Node(data);
        if(this.head == null) {
            this.head = newNode;
            this.tail = newNnode;
        } else {
        this.tail.next = newNode;
        newNode.prev = this.tail;
        newNode = this.tail;
        }
    }
    deleteFirst() {
        if(this.head == null) {
            return null;
        } else if(this.head.next == null) {
            let popped = this.head;
            this.head = null;
            this.tail = null;
            return popped;
        } else {
            let popped = this.head;
            this.head = this.head.next;
            popped.next = null;
            this.head.prev = null;
            return popped;
        }
    }
    deleteLast() {
        if(this.tail = null) {
            return null;
        } else if(this.head == this.tail) {
            let popped = this.tail;
            this.head = null;
            this.tail = null;
            return popped;
        } else {
            let popped =this.tail;
            this.tail = this.tail.next;
            popped.prev = null;
            this.tail.next =null;
            return popped;
        }
    }
    delete(node) {
        let runner = this.head;
        while(runner != node) {
            if(runner == null) {
                return null;
            }
            runner = runner.next;
        }
        let popped = runner;
        runner.prev.next = runner.next;
        if(runner.next != null) {
        runner.next.prev = runner.prev;
        }
        runner.next = null;
        runner.prev = null;
        return popped;
    }
    frontToBackTraversal() {
        let runner = this.head;
        while(runner !== null) {
            console.log(runner);
            runner = runner.next;
        }
        return this;
    }
    backToFrontTraversal() {
        let runner = this.tail;
        while(runner !== null) {
            console.log(runner);
            runner = runner.prev;
        }
        return this;
    }
    insertBefore(node, data) {
        let runner = this.head;
        if(runner === null) {
            return null;
        }
        if(runner === node) {
            this.push(data);
        }
        while(runner) {
            if(runner === node) {
                var new_node = new Node(data);
                new_node.next = runner.next;
                new_node.prev = runner;
                runner.next = new_node;
                new_node.next.prev = new_node;
                return this;
            }
            runner = runner.next;
        }
        return null;
    }
    insertAfter(node, data) {
        let runner = this.head;
        while(runner !== null) {
            runner = runner.next
        }
        node.next = runner.next;
        node.prev = runner;
        runner.next = node;
        if(node.next != null) {
            node.next.prev = node;
        }
        return this;
    }
    isPalindrome() {
        let headrunner = this.head;
        let tailrunner = this.tail;
        if(headrunner == null) {
            return false;
        }
        if(headrunner == tailrunner) {
            return true;
        }
        while(headrunner != tailrunner) {
            if(headrunner.data != tailrunner.data) {
                return false;
            }
            headrunner = headrunner.next;
            tailrunner = tailrunner.prev;
            if(headrunner.prev == tailrunner) {
                return true;
            }
        }
        return true;
    }
    reverse() {
        let headrunner = this.head;
        let temphead = this.head;
        let pointer;
        while(headrunner != null) {
            pointer = headrunner.next;
            headrunner.next = headrunner.prev;
            headrunner.prev = pointer;
            headrunner = pointer;
            this.head = pointer;
        }
        this.tail = temphead;
        return this;
    }
    
    // return true or false if DLL1 is a rotation of DLL2

    // DLL1 = (1)  (2)  (3)  (4)
    // DLL2 = (4)  (1)  (2)  (3)
    isRotation(DLL1, DLL2) {
        let myArr = [];
        let DLL1runner = DLL1.head;
        let DLL2runner
        while(DLL1runner != null) {
            myArr.push(runner.data);
        }
        while(DLL2runner != null) {
            if(DLL2runner.data == myArr[0]) {
                break;
            }
            DLL2runner = DLL2runner.next
        }
        let i = 0
        while(i < myArr.length) {
            if(myArr[i] == DLL2runner.data) {
                if(DLL2runner.next == null) {
                    DLL2runner = DLL2.head;
                } else {
                DLL2runner= DLL2runner.next
                }
                i++;
            } else {
                return false
            }
        }
        return true;
    }

}