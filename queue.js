// first in, first out

// a queue is like the line to get into costco
// if you're at the back, it's a bummer
// if you're at the front, you got there first

// a queue only implements the following methods:
// enqueue, dequeue, front, isEmpty, size

// queues are not indexed and you can't jump
// to values in the middle or center of queues

// the only way to see all the values
// is to dequeue in a loop

class Node {
    constructor(data) {
        this.data = data;
        this.next = null;
    }
}

class Queue {
    constructor() {
        this.back = null;
        this.front = null;
        this.length = 0
    }

    // it is possible to implement all of these methods
    // without linearly iterating through the queue. like
    // the stack, we want these methods to have the same runtime
    // no matter how large we make the queue

    enqueue(data){
        let newNode = Node(data);
        if(this.back == null && this.front == null) {
            this.back == newNode;
            this.front == newNode;
            this.length += 1;
        } else {
            this.back.next = newNode; //someone asking "is this the end of the line?" "
            this.back = newNode; //they hand them the sign saying END OF THE LINE BUDDY
            this.length += 1;
        }
    }
    // add a node to the back

    dequeue() {
        if(this.back == null && this.front == null) {
            return null;
        }
        this.front = this.front.next;
        this.length -= 1;
        return this.front
    }
    // remove and return a node from the front

    front() {
        let temp = this.front;
        return temp;
    }
    // return the front node, not removing it

    isEmpty() {
        if(this.back == null && this.front == null){
            return true;
        } else {
            return false;
        }
    }
    // is the queue empty, true or false?

    size() {
        return this.length;
    }
    // how many nodes are in our queue?
}