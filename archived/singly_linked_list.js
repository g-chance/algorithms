class Node {
    constructor(val) {
        this.val = val;
        this.next = null;
    }
}
class SLL {
    constructor() {
        this.head = null;
    }
    addToFront(val) {
        let newNode = new Node(val);
        let currentHead = this.head;
        newNode.next = currentHead;
        this.head = newNode;
        return this;
    }
    addArray(arr) {
        for (let i = 0; i < arr.length; i++) {
            this.addToFront(arr[i])
        }
        return this;
    }

    //Building on the previous Linked List algorithms, please complete
    //the following methods.

    //nthFromLast(n) {}
    //return the nth from last node in a given list,
    //else return the head.
    //ADVANCED CHALLENGE: Only do one traversal of your list.
    nthFromLast(n) {
        let runner = this.head;
        let slowrunner = this.head;
        if (this.head === null) {
            return "List is empty";
        }
        let count = 0;
        while (runner.next !== null) {
            runner = runner.next
            if (count >= n) {
                slowrunner = slowrunner.next
            }
            count++;
        }
        return n+" from last has a value of: "+slowrunner.val;
    }
    //prepend(target, data){}
    //if the list has a node with the data of target, create
    //a new node (with the data of data) and attach it onto the list
    //before the target node
    prepend(target, val) {
        let runner1 = this.head;
        let runner2 = this.head;
        if (this.head === null) {
            return "List is empty";
        }
        runner1 = runner1.next
        while (runner1 != null && runner1.next != null) {
            if (runner1.val === target) {
                let new_node = new Node(val);
                runner2.next = new_node;
                new_node.next = runner1;
                return this;
            }
            runner1 = runner1.next;
            runner2 = runner2.next;
        }
        return "Target not found"

    }

    //reverse(){}
    //given a linked list, reverse the order of the nodes.
    //there is an optimal way to solve this algorithm, but first
    //try solving it any way that you can.
    reverse() {
        let firstrunner = this.head;
        let secondrunner = this.head;
        if (this.head === null) {
            return "List is empty";
        }
        firstrunner = firstrunner.next
        this.head.next = null;
        while (firstrunner != null && firstrunner.next != null) {
            secondrunner = firstrunner;
            firstrunner = firstrunner.next;
            secondrunner.next = this.head;
            this.head = secondrunner;
        }
        if (firstrunner == null) {
            return "List is too small"
        }
        firstrunner.next = this.head;
        this.head = firstrunner;
        return this;
    }

    //contains(data){}
    //given data, traverse the current list and return true if that
    //data is present in our list, else return false

    containsData(val) {
        let runner = this.head;
        if(runner === null) {
            return false;
        }
        while(runner !== null) {
            if(runner.val === val) {
                return true;
            }
            runner = runner.next;
        }
        return false;
    }
    //count(){}
    //return the total number of nodes in the current list

    count(){
        let runner = this.head;
        let count = 0;
        if (runner === null){
            return count;
        }
        while (runner !== null){
            runner = runner.next;
            count= count +1;
        }
        return count;
    }
    //middle(){}
    //if odd, return the middle node of a list. return the node, not
    //the value, if even return head

    middle(){
        let fastRunner =this.head;
        let slowRunner =this.head;
        while (fastRunner.next !== null){
            //even
            if(fastRunner.next.next === null) {
                return this.head;
            }
            //odd
            fastRunner=fastRunner.next.next;
            slowRunner=slowRunner.next;
        }
        return slowRunner;
    }
    //BONUS
    //rotateFrontAndBack(){}
    //if the total number of nodes in a linked list are 2 or more,
    //swap the head with the last node on the list.
    
    rotateFrontandBack() {
        let runner = this.head;
        let tempHead = this.head;
        while (runner.next.next !== null){
            runner = runner.next
        }
        runner.next.next = this.head.next;
        this.head = runner.next;
        runner.next = tempHead;
        tempHead.next = null;
        return this
    }

    delete(val) {
        let runner = this.head;
        if (runner.val == val){
            this.head = this.head.next
            return this
        }
        while(runner.next !== null && runner.next.val !== val) {
            runner = runner.next;
        }
        console.log(runner.next)
        if (runner.next === null) {
            return false
        }
        runner.next = runner.next.next
    }

    //void append(target, data){}
    //if target is within the current list, create a new node
    //with the data of data, and append it directly after target
    append(target, val) {
        let runner = this.head;
        while(runner.next !== null && runner.val !== target) {
            runner = runner.next;
        }
        let new_node = new Node(val);
        if (runner.next == null){
            runner.next = new_node
            new_node.next = null
        }
        else {
            let temp_node = runner.next;
            runner.next = new_node;
            new_node.next = temp_node;
        }
    }

}
    //ADVANCED BONUS
    //recursiveContains(data, runner){}
    //try to refactor contains with a function that calls itself


myList = new SLL();
myList.addArray([8,2,3,5,9,6,4])

var runner = myList.head
while(runner !== null) {
    console.log("first",runner.val)
    runner = runner.next
}

console.log(myList.nthFromLast(2))
// console.log(myList.containsData(3))
// console.log(myList.count())
// console.log(myList.middle())
// console.log(myList.rotateFrontandBack())

myList.reverse()
// myList.prepend(5,100)

var runner = myList.head
while(runner !== null) {
    console.log("reversed",runner.val)
    runner = runner.next
}

// var runner = myList.head
// while(runner !== null) {
//     console.log("reversed",runner.val)
//     runner = runner.next
// }

// myList.append(2,11)

// var runner = myList.head
// while(runner !== null) {
//     console.log("appended",runner.val)
//     runner = runner.next
// }