class Node {
    constructor(data){
        this.data = data;
        this.next = null;
    }
}

class Stack {
    constructor(){
        this.top = null;
        this.length = 0;
    }

    push(data){
        let newNode = new Node(data);
        newNode.next = this.top;
        this.top = newNode;
        this.length += 1;
    }

    pop(){
        if(this.top == null) {
            return null;
        }
        let temp = this.top;
        this.top = this.top.next;
        this.length -= 1;
        return temp;
    }

    isEmpty(){
        if(this.top == null) {
            return true;
        }
        return false;
    }

    peek(){
        return this.top
    }

    size(){
        return this.length;
    }

}

const reverse = stack => {
    let newStack = new Stack();
    while (stack.isEmpty() == false){
        let popped = stack.pop();
        newStack.push(popped.data);
    }
    return newStack;
}


let myStack = new Stack();

myStack.push(3);
myStack.push(4);
myStack.push(5);
myStack.push(6);
myStack.push(7);

// myStack.pop();

myStack = reverse(myStack)

let runner = myStack.top;
while(runner != null) {
    console.log(runner.data);
    runner=runner.next;
}

console.log(myStack.isEmpty());
console.log(myStack.peek());
console.log(myStack.size());