class Node {
    constructor(data){
        this.data = data;
        this.next = null;
    }
}

class minStack {
    constructor(){
        this.top = null;
        this.size = 0;
        this.min = [];

        // this.mins = [] - what about if we store an array??
    }
    push(node) {
        if(this.top===null){
            this.top=node;
            this.min.push(node.data)
            this.size++
            return this;
        }
        else{
            let temp=this.top;
            this.top=node;
            this.top.next=temp;
            if(node.data < this.min[this.min.length-1]) {
                this.min.push(node.data)
            }
            this.size++
            return this;
        }
    }
    pop(){
        if(this.top==null){
            return null;
        }else if(this.top.next==null){
            this.min.pop()
            this.top=null;

            return this;
        }else{
            if(this.top.data === this.min[this.min.length-1]) {
                let popped = this.min.pop();
                console.log("in pop method - popped",popped)
                console.log("in pop method - this.min",this.min)
            }
            let temp=this.top;
            this.top=this.top.next;
            temp.next=null;
            return this;
        }
    }
    peek(){
        return this.top;
    }
    isEmpty(){
        return this.top === null;
    }
    size(){
        return this.size;
    }
    getMin(){
        return this.min[this.min.length-1]

    }

}

myStack = new minStack();

myStack.push(new Node(1))
myStack.push(new Node(-2))
myStack.push(new Node(3))
myStack.push(new Node(-1))
myStack.push(new Node(0))
myStack.pop()
myStack.pop()
myStack.pop()
myStack.pop()

console.log("myStack.min",myStack.min)

let runner = myStack.top
while(runner != null) {
    console.log("myStack",runner.data)
    runner = runner.next
}
console.log("min",myStack.getMin())