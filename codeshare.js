// given an unsorted array of integers, each integer is repeated twice except for one of them. find and return the integer that is not duplicated
// try without constraints
// try with memory constraint -> dont use additional structures (look up bitwise)

let arr = [1,2,2,1,3,4,4,3,5,6,6,7,7];
function findLoner(arr) {
    
    for(let i=0;i<arr.length;i++) {
        for(let j=0;j<arr.length;j++) {
            if(i == j) {
                continue;
            }
            if(arr[i] == arr[j]) {
                break;
            }
            else if(j == arr.length-1) {
                return arr[i]
            }
        }
    }
}
console.log(findLoner(arr));

// With a sorted list
function findLonerSorted(arr){
    arr.sort()
    for (i=0; i <arr.length; i+=2){
        if( arr[i] == arr[i+1]){
            continue
        }
        else {
            return arr[i]
        }
    }
}
console.log("sorted", findLonerSorted(arr));

// With bitwise
function findLonerBW(arr) {
    result = arr[0]
    for(let i=1;i<arr.length;i++) {
        result = result ^ arr[i];
    }
    return result
}
console.log("bitwise", findLonerBW(arr));