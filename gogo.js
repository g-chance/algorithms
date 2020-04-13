function reverseStr(str) {
    var reverseArray = [];                      // here is where we store each substring from the original str
    var endIdx = str.length;					// this keeps track of the end of the current substring
    var temp;									// this is where we hold the substring before we push it into reverseArray

    for (var i = str.length - 1; i > -1; i--) {
        if (str[i] == ' ') {
            temp = str.substring(i + 1, endIdx);
            reverseArray.push(temp);
            endIdx = i;
        }
    }
    temp = str.substring(i, endIdx)
    reverseArray.push(temp);

    let newStr = reverseArray.join(' ');
    return newStr;
}

console.log(reverseStr("this is funny"))