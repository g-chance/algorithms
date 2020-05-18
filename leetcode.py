#   ==================== ADD TWO NUMBERS ====================

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    def addTwoNumbers(self, l1, l2):
        def getTotal(list, total):
            myList = []
            runner = list
            while runner != None:
                myList.append(runner.val)
                runner = runner.next
            for i in reversed(range(len(myList))):
                total += myList[i]*10**(i)
            return total
        total = 0
        total = getTotal(l1, total)
        total = getTotal(l2, total)
        l3 = ListNode(total % 10)
        total //= 10
        runner = l3
        while total != 0:
            runner.next = ListNode(total % 10)
            runner = runner.next
            total //= 10
        return l3

list_one = ListNode(2)
runner = list_one
runner.next = ListNode(4)
runner = runner.next
runner.next = ListNode(3)

list_two = ListNode(5)
runner = list_two
runner.next = ListNode(6)
runner = runner.next
runner.next = ListNode(4)

list_three = ListNode()
print(list_three.addTwoNumbers(list_one,list_two).val)

# runner = list_three.head
# while runner != None:
#     print("list_three",runner.val)
#     runner = runner.next

# class SLL:
#     def __init__(self):
#         self.head = None
#     def addToFront(self, val):
#         new_node = ListNode(val)
#         new_node.next = self.head
#         self.head = new_node
#         return self

#   ==================== DISTRIBUTE CANDIES TO PEOPLE ====================

# candies = 40
# num_people = 4

# def distributeCandies(candies: int, num_people: int):
#     myList = [0] * num_people
#     count = 0
#     while candies > 0:
#         for i in range(num_people):
#             if (candies - (1+count)) > 0:
#                 myList[i] += (1+count)
#                 candies -= (1+count)
#                 count += 1
#             else:
#                 myList[i] += candies
#                 candies -= candies
#     return myList
# print(distributeCandies(candies, num_people))


#   ==================== REMOVE ELEMENT ====================

# nums = [3,3,3,2] 
# val = 3

# def removeElement(nums, val: int) -> int:
#     i = 0
#     for j in range(len(nums)):
#         if nums[j] != val:
#             nums[i] = nums[j]
#             i += 1
#     print(nums)
#     return i
# print(removeElement(nums, val))

# def removeElement(nums, val: int) -> int:
#     for i in range(len(nums)):
#         if nums[i] == val:
#             for j in range(i+1,len(nums)):
#                 if nums[j] != val:
#                     temp = nums[i]
#                     nums[i] = nums[j]
#                     nums[j] = temp
#                     break
#             else:
#                 length = 0
#                 for num in nums:
#                     if num != val:
#                         length += 1
#                 print(nums)
#                 return length
#     return len(nums)
# print(removeElement(nums, val))


#   ==================== PLUS ONE ====================

# nums = [9,9,9]

# def plusOne(digits):
#     count = 0
#     for i in range(len(digits)-1,-1,-1):
#         if digits[i] == 9:
#             digits[i] = 0
#             count += 1
#         else:
#             digits[i] += 1
#             return digits
#     if count == len(digits):
#         digits[i] = 1
#         digits.append(0)
#     return digits
# print(plusOne(nums))

    # Build the digit, add 1 to it, then rebuild the list
# def plusOne(digits):
#     digit = 0
#     for i in range(len(digits)):
#         digit += digits[i]*(10**(len(digits)-i-1))
#     digit += 1
#     print(digit)
#     digits = []
#     while digit != 0:
#         digits = [digit % 10] + digits
#         digit //= 10
#     return digits
# print(plusOne(nums))

    # Convert list to str, str to int, int back to str, then rebuilding the list
# def plusOne(digits):
#     myStr = ""
#     for digit in digits:
#         myStr += str(digit)
#     digit = int(myStr)
#     digit += 1
#     myStr = str(digit)
#     digits = [int(char) for char in myStr]
#     return digits
# print(plusOne(nums))


#   ==================== SINGLE NUMBER ====================

# nums = [2,3,3,4,1,1,2]

# def singleNumber(nums) -> int:
#     s_num = 0
#     for num in nums:
#         s_num ^= num
#     return s_num
# print(singleNumber(nums))

    # Works but is too slow
# def singleNumber(nums) -> int:
#     for i in range(len(nums)):
#         if(nums[i] != None):
#             for j in range(i+1,len(nums)):
#                 if nums[i] == nums[j]:
#                     nums[i] = nums[j] = None
#     for k in nums:
#         if k != None:
#             return k
# print(singleNumber(nums))

    # With dictionary
# def singleNumber(nums) -> int:
#     myDict = {}
#     for num in nums:
#         if num in myDict:
#             del myDict[num]
#         else:
#             myDict[num] = num
#     for key in myDict:
#         return key
# print(singleNumber(nums))


#   ==================== ROTATE ARRAY ====================

# nums = [-1,-100,3,99]
# k = 7
# def rotate(nums, k: int) -> None:
#     temp_list = []
#     if k > len(nums):
#         k -= len(nums)
#     for i in range(len(nums)-k):
#         temp_list.append(nums[i])
#     for i in range(len(nums)):
#         if i < k:
#             nums[i] = nums[len(nums)-k+i]
#         else:
#             nums[i] = temp_list[i-k]
#     print(nums)
#     return None

# rotate(nums, k)


#   ==================== COUNT PRIMES ====================

# def countPrimes(n: int) -> int:
#     primes_list = [0] * n
#     count = 0
#     if n < 3:
#         return 0
#     for i in range(2,n):
#         if primes_list[i] == 0:
#             count += 1
#             for j in range(i*i,n,i):
#                 primes_list[j] = 1
#     return count

    # Leet code didn't accept this because it was too slow, but it does work
# def countPrimes(n: int) -> int:
#     if n < 3:
#         return 0
#     primes_list = [2]
#     for i in range(3,n,2):
#         for j in range(0,len(primes_list)//2):
#             if i % primes_list[j] == 0:
#                 print("break","i",i,"j",j)
#                 break
#         else:
#             print("else",i)
#             primes_list.append(i)
#     return len(primes_list)

# print(countPrimes(100))


#   ==================== MOVE ZEROES ====================

# nums = [0,1,0,3,12]

# def moveZeroes(nums) -> None:
#     for i in range(len(nums)-1,-1,-1):
#         if(nums[i] == 0):
#             for j in range(i,len(nums)-1):
#                 nums[j] = nums[j+1]
#             nums[len(nums)-1] = 0
#     print(nums)

# def moveZeroes(nums) -> None:
#     myList = []
#     for i in range(len(nums)):
#         if nums[i] == 0:
#             continue
#         else:
#             myList.append(nums[i])
#     for i in range(len(nums)):
#         if(i<len(myList)):
#             nums[i] = myList[i]
#         else:
#             nums[i] = 0
#     print(nums)

# moveZeroes(nums)


#   ==================== BULLS AND COWS ====================

# def getHint(secret: str, guess: str) -> str:
#     myDict = {}
#     count_a = 0
#     count_b = 0
#     guessesLeft = ""
#     for i in range(len(secret)):
#         if secret[i] == guess[i]:
#             count_a += 1
#         elif secret[i] in myDict:
#             myDict[secret[i]] += 1
#             guessesLeft += guess[i]
#         else:
#             myDict[secret[i]] = 1
#             guessesLeft += guess[i]
#     for char in guessesLeft:
#         if char in myDict and myDict[char] != 0:
#             count_b += 1
#             myDict[char] -= 1
#     print(myDict)
#     return str(count_a)+"A"+str(count_b)+"B"
# print(getHint("11", "10"))


#   ==================== REVERSE STRING III ====================

# str = "Hello I am nuggets"

# def reverseWords(s: str) -> str:
#     strList = s.split()
#     newList = []
#     print(strList)
#     for word in strList:
#         newStr = ""
#         for i in range(len(word)-1,-1,-1):
#             newStr += word[i]
#         newList.append(newStr)
#     return " ".join(newList)

# def reverseWords(s: str) -> str:
#     newStr = ""
#     count = 0
#     for i in range(len(s)):
#         if s[i] == " ":
#             for j in range(i,-1+count,-1):
#                 if(s[j] == " "):
#                     count += 1
#                     continue
#                 newStr += s[j]
#                 count += 1
#             newStr += " "
#     for j in range(len(s)-1,-1+count,-1):
#         newStr += s[j]
#     return newStr

# print(reverseWords(str))