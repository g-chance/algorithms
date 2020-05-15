

//  ===== REVERSE STRING II =====
let s = "abcdefghijk";
// Output: "bacdfeg"

var reverseStr = function(s, k) {
    if(k == 1) {
        return s;
    }
    let newStr = "";
    let i = 0;
    while(s.length-i > k) {
        console.log(i);
        for(let j=(i+k)-1; j>=i; j--) {
            newStr += s[j];
        }
        for(j=i+k; j<i+k*2 && j<s.length; j++) {
            newStr += s[j]
        }
        i += k*2
    }
    for(let j=s.length-1; j>=i; j--) {
        newStr += s[j];
    }
    return newStr
};
console.log(reverseStr(s, 3));

//  ===== FIND ALL NUMBERS DISAPPEARED IN AN ARRAY =====

// let nums = [4,3,2,7,8,2,3,1];

//     // Space InPlace Modification
// var findDisappearedNumbers = function(nums) {
//     for(let i=0;i<nums.length;i++) {
//         new_index = Math.abs(nums[i])-1
//         if(nums[new_index] > 0) {
//             nums[new_index] *= -1;
//         }
//     }
//     let newArr = [];
//     for(let j=0;j<nums.length;j++) {
//         if(nums[j] > 0) {
//             newArr.push(j+1);
//         }
//     }
//     return newArr;
// }
// console.log(findDisappearedNumbers(nums));

    // With double loop
// var findDisappearedNumbers = function(nums) {
//     let myArr = [];
//     for(let i=1;i<=nums.length;i++) {
//         for(let j=0;j<nums.length;j++) {
//             if(i == nums[j]) {
//                 break;
//             } else if(j == nums.length-1) {
//                 myArr.push(i);
//             }
//         }
//     }
//     return myArr;
// }
// console.log(findDisappearedNumbers(nums))

    // With dictionary
// var findDisappearedNumbers = function(nums) {
//     let myDict = {};
//     let newArr = [];
//     for(let i=0;i<nums.length;i++) {
//         if(!(nums[i] in myDict)) {
//             myDict[nums[i]] = nums[i];
//         }
//     }
//     console.log(myDict)
//     for(let i=1;i<=nums.length;i++) {
//         if(!(i in myDict)) {
//             newArr.push(i);
//         }
//     }
//     return newArr;
// };
// console.log(findDisappearedNumbers(nums));

    //  ===== REVERSE INTEGER =====

// let x = -210;
//     // Using modulus
// function reverse(x) {
//     rev = 0;
//     while(x != 0) {
//         let pop = x % 10;
//         x = Math.trunc(x/10);
//         rev = rev * 10 + pop;
//     }
//     if(rev > 2**31 || rev < 2**31*(-1)) {
//         return 0;
//     }
//     return rev;
// }
// console.log(reverse(x));

    // By converting to string
// function reverse(x) {
//     let newString = "";
//     if(x<0) {
//         newString += "-";
//     }
//     x = x.toString();
//     for(let i=x.length-1;i>=0;i--) {
//         newString += x[i];
//     }
//     if(parseInt(newString) > 2**32) {
//         return 0;
//     }
//     return parseInt(newString);
// };
// console.log(reverse(x));


//  ==================== TWOSUM ====================

// var twoSum = function(nums, target) {
//     for(let i=0;i<nums.length-1;i++) {
//         for(let j=i;j<nums.length-1;j++) {
//             if(nums[j] > target) {
//                 break;
//             } else if(nums[j] + nums[j+1] == target) {
//                 console.log(nums[j])
//                 return (j)+","+(j+1);
//             }
//         }
//     }
//     return false;
// };
// console.log(twoSum([2,7,11,15], 9))