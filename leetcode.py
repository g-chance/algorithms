#   ================== REVIEW ====================
#   ==================== 53 ====================
#   ================== REVIEW ====================

    # After looking at discussion
# def maxSubArray(nums) -> int:
#     max = nums[0]
#     temp = nums[0]
#     for i in range(1, len(nums)):
#         temp += nums[i]
#         if nums[i] > temp:
#             temp = nums[i]
#         if temp > max:
#             max = temp
#     return max

# def maxSubArray(nums) -> int:
#     max_sum = -(2**32)
#     for i in range(len(nums)):
#         for j in range(i,len(nums)):
#             if sum(nums[i:j+1]) > max_sum:
#                 max_sum = sum(nums[i:j+1])
#     return max_sum

# def maxSubArray(nums) -> int:
#     max_sum = nums[0]
#     i = 0
#     while True:
#         for j in range(len(nums)):
#             if sum(nums[i:j+1]) > 0:
#                 if sum(nums[i:j+1]) > max_sum:
#                     max_sum = sum(nums[i:j+1])
#                     print(max_sum)
#             elif j < len(nums)-1 and nums[j] > max_sum:
#                 max_sum = nums[j]
#                 i = j
#                 print(max_sum)
#             else:
#                 i = j
#         if j == len(nums)-1:
#             break
#     return max_sum

print(maxSubArray([1]))


#   ==================== 38 ====================

# def countAndSay(n: int) -> str:
#     if n == 1:
#         return "1"
#     my_str = ""
#     freq = 1
#     num = countAndSay(n-1)
#     for i in range(len(num)-1):
#         val = str(num[i])
#         if num[i] == num[i+1]:
#             freq += 1
#         else:
#             my_str += str(freq)+val
#             freq = 1
#     val = num[-1]
#     my_str += str(freq)+val
#     return my_str 

# def countAndSay(n: int) -> str:
#     if n == 1:
#         return str(n)
#     num = "11"
#     for i in range(2,n):
#         frequency = 1
#         my_str = ""
#         for j in range(len(num)-1):
#             value = num[j]
#             if num[j] == num[j+1]:
#                 frequency += 1
#             else:
#                 my_str += str(frequency)+str(value)
#                 frequency = 1
#         value = num[-1]
#         my_str += str(frequency)+str(value)
#         num = my_str
#     return num

# print(countAndSay(5))


#   ==================== Remove Duplicates ====================

# def removeDuplicates(nums) -> int:
#     if len(nums) < 2:
#         return len(nums)
#     i = 0
#     while True:
#         if nums[i] == nums[i+1]:
#             nums.pop(i)
#         else:
#             i += 1
#         if i >= len(nums)-1:
#             return len(nums)
#     return len(nums)

# print(removeDuplicates([1,1,1]))


#   ==================== 21 ====================

#     # Not functional - just here for reference
# def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
#     dummy = l3 = ListNode(0)
#     while l1 and l2:
#         if l1.val <= l2.val:
#             l3.next = ListNode(l1.val)
#             l1 = l1.next
#         else:
#             l3.next = ListNode(l2.val)
#             l2 = l2.next
#         l3 = l3.next
#     l3.next = l1 or l2
#     return dummy.next


#   ==================== 20 ====================

    # After looking at discussion - doing it with a dictionary
# def isValid(s: str) -> bool:
#     my_dict = { "}": "{", "]": "[", ")": "(" }
#     stack = []
#     for i, c in enumerate(s):
#         if c in my_dict.values():
#             stack.append(c)
#         elif c in my_dict:
#             if stack and my_dict[c] != stack.pop():
#                 return False
#             elif not stack:
#                 return False
#     return True

    # How I did it
# def isValid(s: str) -> bool:
#     my_stack = []
#     for i, c in enumerate(s):
#         if c == ")" or c == "}" or c == "]":
#             if len(my_stack) == 0:
#                 return False
#         if c == ")" and my_stack[-1] != "(":
#             return False
#         if c == "}" and my_stack[-1] != "{":
#             return False
#         if c == "]" and my_stack[-1] != "[":
#             return False
#         elif c == "(" or c == "{" or c == "[":
#             my_stack.append(c)
#             continue
#         my_stack.pop()
#     if len(my_stack) > 0:
#         return False
#     return True

# print(isValid("(()())"))

#   ==================== Namespace / Scope Practice ====================

# a = "butts"
# print("global",a)

# def outer_func():
#     # global a
#     # nonlocal a
#     a = "poops"
#     print("outer",a)
#     def middle_scope():
#         # global a
#         # nonlocal a
#         a = "cacos"
#         print("middle",a)
#         def inner():
#             # global a
#             # nonlocal a
#             a = "dingding"
#             print("inner",a)
#             # global b
#             # b = "stuff"
#         inner()
#         print("middle",a)
#     middle_scope()
#     print("outer",a)
# outer_func()

# print("global",a)
# # print("global",b)

#   ==================== Longest Common Prefix ====================


# def longestCommonPrefix(strs) -> str:
#     if len(strs) == 1:
#         return strs[0]
#     my_str = ""
#     j = 0
#     while strs:
#         for i in range(len(strs)-1):
#             if j >= len(strs[i]) or j >= len(strs[i+1]):
#                 return my_str
#             if strs[i][j] == strs[i+1][j]:
#                 continue
#             return my_str
#         my_str += strs[0][j]
#         j += 1
#     return my_str

# print(longestCommonPrefix(["flower", "flow"]))


#   ==================== 13 ====================

# def romanToInt(s: str) -> int:
#     romans = {
#         "I" : 1, "V" : 5, "X" : 10, "L" : 50, "C" : 100, "D" : 500, "M" : 1000
#     }
#     count = 0
#     for i in range(len(s)-1):
#         if romans[s[i]] < romans[s[i+1]]:
#             count -= romans[s[i]]
#         else:
#             count += romans[s[i]]
#     return count + romans[s[i+1]]

# print(romanToInt("MCMXCIV"))


#   :===================,1221 ====================

# def balancedStringSplit(s: str) -> int:
#     count = 0
#     splits = 0
#     for char in s:
#         if char == "L":
#             count -=1
#         if char == "R":
#             count += 1
#         if count == 0:
#             splits += 1
#     return splits

# print(balancedStringSplit("RLLLLRRRLR"))


#   ==================== 1408 ====================

# def diStringMatch(S: str):
#     my_arr = [0]*(len(S)+1)
#     count_d = 0
#     for i in range(len(S)):
#         if S[i] == 'D':
#             count_d += 1
#     my_arr[0] = count_d
#     inc = dec = count_d
#     for i in range(len(S)):
#         if S[i] == 'I':
#             inc += 1
#             my_arr[i+1] = inc
#         else:
#             dec -= 1
#             my_arr[i+1] = dec
#     return my_arr

# print(diStringMatch("IDIDI"))


#   ==================== 1408 ====================

# def stringMatching(words):
#     my_arr = []
#     for i in range(len(words)):
#         for j in range(len(words)):
#             if i != j:
#                 if words[i] in words[j] and words[i] not in my_arr:
#                     my_arr += [words[i]]
#     return my_arr

# print(stringMatching(words = ["leetcoder","leetcode","od","hamlet","am"]))


#   ==================== 387 ====================

# def missingNumber(nums) -> int:
#     my_arr = [0]*(len(nums)+1)
#     for i in range(len(nums)):
#         my_arr[abs(nums[i])] = 1
#     print(my_arr)
#     for i, num in enumerate(my_arr):
#         if num == 0:
#             return i
#     return

# def missingNumber(nums) -> int:
#     total = sum([i for i in range(len(nums)+1)])
#     missing_total = 0
#     for num in nums:
#         missing_total += num
#     return total - missing_total

# print(missingNumber([9,6,4,2,3,5,7,0,1]))


#   ==================== 387 ====================

# def firstUniqChar(s: str) -> int:
#     my_dict = {}
#     my_set = set()
#     for i, char in enumerate(s):
#         if char not in my_set:
#             my_dict[char] = i
#             my_set.add(char)
#         elif char in my_dict:
#             del my_dict[char]
#     if not my_dict:
#         return -1
#     for key in my_dict:
#         return my_dict[key]

# def firstUniqChar(s: str) -> int:
#     if not s:
#         return -1
#     my_str = list(s).reverse()
#     for i in range(len(s)-1,-1,-1):
#         popped = my_str.pop(i)
#         if popped not in my_str:
#             return len(s)-i
#         my_str.append(popped)
#     return -1

# def firstUniqChar(s: str) -> int:
#     if not s:
#         return -1
#     for i, char in enumerate(s):
#         if s.count(char) == 1:
#             return i
#     return -1

# def firstUniqChar(s: str) -> int:
#     if not s:
#         return -1
#     if len(s) == 1:
#         return 0
#     for i in range(len(s)-1):
#         if s[i] not in s[:i]+s[i+1:]:
#             return i
#     if s[i+1] not in s[:i+1]:
#         return i+1
#     return -1

# print(firstUniqChar("leetcode"))

#   ==================== 160 ====================

    # Got this from discussion.. genius
# def getIntersectionNode(headA, headB):
#     runnera = headA
#     runnerb = headB
#     while runnera != runnerb:
#         if runnera == None:
#             runnera = headB
#         if runnerb == None:
#             runnerb = headA
#         runnera = runnera.next
#         runnerb = runnerb.next
#     return runnera

# def getIntersectionNode(headA, headB):
#     my_arr = []
#     runnera = headA
#     runnerb = headB
#     while runnera != None and runnerb != None:
#         print(runnera.val, runnerb.val)
#         if runnera in my_arr:
#             return runnera
#         my_arr.append(runnera)
#         if runnerb in my_arr:
#             return runnerb
#         my_arr.append(runnerb)
#         runnera = runnera.next
#         runnerb = runnerb.next
#     runner = None
#     print(runnera,runnerb)
#     if runnera:
#         runner = runnera
#         print("runnera")
#     elif runnerb:
#         runner = runnerb
#         print("runnerb")
#     while runner != None:
#         if runner in my_arr:
#             return runner
#         runner = runner.next
#     return None

# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

# list1 = ListNode(1)
# runner = list1
# runner.next = ListNode(3); runner = runner.next
# runner.next = ListNode(5); runner = runner.next
# temp = runner.next = ListNode(7); runner = runner.next
# # runner.next = ListNode(9); runner = runner.next
# list2 = ListNode(2)
# runner = list2
# runner.next = ListNode(4); runner = runner.next
# runner.next = temp; runner = runner.next
# print(list2.next.next.val)

# print(getIntersectionNode(list1,list2).val)

#   ==================== 819 ====================

# import re

    # With REGEX
# def mostCommonWord(paragraph: str, banned) -> str:
#     alnum = re.compile(r"\w+")
#     word_list = alnum.findall(paragraph)
#     word_dict = {}
#     count = 0
#     result = ""
#     for word in word_list:
#         print(word.lower())
#         if word.lower() not in banned and word.lower() not in word_dict:
#             word_dict[word.lower()] = 1
#         elif word.lower() not in banned:
#             word_dict[word.lower()] +=1
#     print(word_dict)
#     for word, val in word_dict.items():
#         if val > count:
#             count = val
#             result = word
#     return result


# def mostCommonWord(paragraph: str, banned) -> str:
#     word_list = []
#     new_idx = 0
#     for i, char in enumerate(paragraph):
#         if not char.isalpha():
#             word_list.append(paragraph[new_idx:i])
#             new_idx = i+1
#         elif i == len(paragraph)-1:
#             word_list.append(paragraph[new_idx:i+1])
#     print(word_list)
#     word_dict = {}
#     result = ""
#     count = 0
#     for i, word in enumerate(word_list):
#         print(word)
#         if not word.isalpha():
#             new_word = ""
#             for char in word:
#                 if char.isalpha():
#                     new_word += char
#             word = new_word
#         if word.lower() not in banned and word.lower() not in word_dict and not word == "":
#             word_dict[word.lower()] = 1
#         elif word.lower() not in banned and not word == "":
#             word_dict[word.lower()] += 1
#     for word,val in word_dict.items():
#         if val > count:
#             result = word
#             count = val
#     return result

# print(mostCommonWord(paragraph = "Bob hit a ball, the hit BALL flew far after it was hit.", banned = ["hit"]))

#   ==================== 83 ====================

# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

# def deleteDuplicates(head: ListNode) -> ListNode:
#     runner = head
#     while runner != None and runner.next != None:
#         if runner.val == runner.next.val:
#             runner.next = runner.next.next
#         else:
#             runner = runner.next
#     return head

# list1 = ListNode(1)
# runner = list1
# runner.next = ListNode(1); runner = runner.next
# runner.next = ListNode(2); runner = runner.next
# runner.next = ListNode(2); runner = runner.next
# runner.next = ListNode(3); runner = runner.next
# print(list1.next.val)

# print(deleteDuplicates(list1))
# new_list = deleteDuplicates(list1)
# print(new_list.val,new_list.next.val,new_list.next.next.val)

#   ==================== 1047 ====================

# def removeDuplicates(S: str) -> str:
#     stack = []
#     for char in S:
#         if len(stack) > 0 and stack[len(stack)-1] == char:
#             stack.pop()
#         else:
#             stack.append(char)
#     return "".join(stack)

# def removeDuplicates(S: str) -> str:
#     while True:
#         for i in range(len(S)-1):
#             if S[i] == S[i+1]:
#                 S = S[:i]+S[i+2:]
#                 break
#         else:
#             if len(S) < 2 or len(S) == 2 and S[0] != S[1]:
#                 return S
#             if len(S) == 2 and S[0] == S[1]:
#                 return ""
#             return S

# def removeDuplicates(S: str) -> str:
#     if len(S) < 2 or len(S) == 2 and S[0] != S[1]:
#         return S
#     if len(S) == 2 and S[0] == S[1]:
#         return ""
#     new_str = ""
#     for i in range(len(S)-1):
#         if S[i] == S[i+1]:
#             new_str += S[:i]+S[i+2:]
#             return removeDuplicates(new_str)
#     return S

# print(removeDuplicates("abbbabaaa"))


#   ==================== 219 ====================

# def containsNearbyDuplicate(nums, k: int) -> bool:
#     my_dict = {}
#     for i in range(len(nums)):
#         if nums[i] in my_dict:
#             my_dict[nums[i]] += [i]
#             if abs(my_dict[nums[i]][0] - my_dict[nums[i]][1]) <= k:
#                 return True
#             my_dict[nums[i]].pop(0)
#         else:
#             my_dict[nums[i]] = [i]
#     return False

# print(containsNearbyDuplicate([1,2,3,1,2,3], 2))


#   ==================== 217 ====================

# def containsDuplicate(nums) -> bool:
#     myDict = {}
#     for num in nums:
#         if num in myDict:
#             return True
#         myDict[num] = 0
#     return False

# def containsDuplicate(nums) -> bool:
#     nums.sort()
#     for i in range(len(nums)-1):
#         if nums[i] == nums[i+1]:
#             return True
#     return False

# print(containsDuplicate([1,2,3,1]))


#   ==================== 500 ====================

# def findWords(words):
#     words_arr = []
#     keyboard = {
#         "row1": set("qwertyuiop"),
#         "row2": set("asdfghjkl"),
#         "row3": set("zxcvbnm")
#     }
#     for word in words:
#         for row in keyboard:
#             print(set(word.lower()) - keyboard[row])
#             print(keyboard[row])
#             if set(word.lower()) - keyboard[row] == set():
#                 words_arr.append(word)
#     return words_arr

# def findWords(words):
#     words_arr = []
#     keyboard = {
#         "row1": "qwertyuiop",
#         "row2": "asdfghjkl",
#         "row3": "zxcvbnm"
#     }
#     for word in words:
#         for row in keyboard:
#             if word[0].lower() in keyboard[row]:
#                 for char in word:
#                     if char.lower() not in keyboard[row]:
#                         break
#                 else:
#                     words_arr.append(word)
#     return(words_arr)

# print(findWords(["Hello", "Alaska", "Dad", "Peace"]))

#   ==================== 997 ====================

# def findJudge(N: int, trust) -> int:
#     graph = [0]*(N+1)
#     for i, j in trust:
#         graph[i] -= 1
#         graph[j] += 1
#     for p in graph:
#         if p == N-1:
#             return p
#     return -1

# def findJudge(N: int, trust) -> int:
#     trust_dict = {}
#     for i in range(len(trust)):
#         if trust[i][0] not in trust_dict:
#             trust_dict[trust[i][0]] = [trust[i][1]]
#         else:
#             trust_dict[trust[i][0]] += [trust[i][1]]
#     print(trust_dict)
#     for i in range(1, N+1):
#         if i not in trust_dict:
#             count = 0
#             for key, val in trust_dict.items():
#                 print(val)
#                 if i not in val:
#                     break
#                 count += 1
#             else:
#                 if count == N-1:
#                     return i
#     return -1

# print(findJudge(4, [[1,3],[1,4],[2,3]]))


#   ==================== 389 ====================

# def findTheDifference(s: str, t: str) -> str:
#     dict_s = {}
#     for char in s:
#         if char in dict_s:
#             dict_s[char] += 1
#         else:
#             dict_s[char] = 1
#     for char in t:
#         if char not in dict_s:
#             return char
#         dict_s[char] -= 1
#         if dict_s[char] < 0:
#             return char

# def findTheDifference(s: str, t: str) -> str:
#     list_s = list(s)
#     list_t = list(t)
#     list_s.sort()
#     list_t.sort()
#     print(list_s,list_t)
#     for i in range(len(list_s)):
#         if list_s[i] != list_t[i]:
#             return list_t[i]
#     return list_t[len(t)-1]

# print(findTheDifference("abcd","abedc"))

#   ==================== 1200 ====================

# def minimumAbsDifference(arr):
    # arr.sort()
    # min_dist = 10**6
    # min_list = []
    # print(arr)
    # for i in reversed(range(1,len(arr))):
    #     if abs(arr[i] - arr[i-1]) < min_dist:
    #         min_dist = abs(arr[i] - arr[i-1])
    # print(min_dist)
    # for i in range(len(arr)-1):
    #     if abs(arr[i+1] - arr[i]) == min_dist:
    #         min_list.append([arr[i],arr[i+1]])
    # print(min_list)
    # return min_list

# print(minimumAbsDifference([4,2,1,3]))


#   ==================== 463 ====================

# def islandPerimeter(grid) -> int:
#     def checkSides(grid, i, j):
#         sum = 4
#         if j < len(grid[0])-1 and grid[i][j+1] == 1:
#             sum -= 2
#         if i < len(grid)-1 and grid[i+1][j] == 1:
#             sum -= 2
#         return sum
#     perim = 0
#     for i in range(len(grid)):
#         for j in range(len(grid[0])):
#             if grid[i][j] == 1:
#                 perim += checkSides(grid,i,j)
#     return perim
# print(islandPerimeter(
#     [[0,1,0,0],
#     [1,1,1,0],
#     [0,1,0,0],
#     [1,1,0,0]]
# ))


#   ==================== 125 ====================

# def isPalindrome(s: str) -> bool:
#     l = 0
#     r = len(s)-1
#     while l < r:
#         while l < len(s) and not s[l].isalnum():
#             l += 1
#         while r > -1 and not s[r].isalnum():
#             r -= 1
#         if l == len(s):
#             return True
#         if s[l].lower() != s[r].lower():
#             return False
#         l += 1
#         r -= 1
#     return True

# def isPalindrome(s: str) -> bool:
#     new_str = ""
#     for char in s:
#         if char.isalnum():
#             new_str += char.lower()
#     print(new_str)
#     for i in range(len(new_str)//2):
#         if new_str[i] != new_str[len(new_str)-i-1]:
#             return False
#     return True

# print(isPalindrome(",a,aa,"))

#   ========================================

# def validPalindrome(s: str) -> bool:
#     count = 0
#     l = [char for char in s]
#     for i in range(len(l)//2):
#         if l[i] != l[len(l)-i-1]:
#             l1 = l[i+1:len(l)-i]
#             l2 = l[i:len(l)-i-1]
#             for j in range(len(l1)//2):
#                 if l1[j] != l1[len(l1)-j-1]:
#                     if l2:
#                         break
#                     return False
#             else: return True
#             for k in range(len(l2)//2):
#                 if l2[k] != l2[len(l2)-k-1]:
#                     return False
#             return True
#     return True

# print(validPalindrome("abca"))


#   ========================================

# def isPalindrome(x: int) -> bool:
#     s = str(x)
#     for i in range(len(s)//2):
#         if s[i] == s[len(s)-i-1]:
#             pass
#         else:
#             return False
#     return True

# def isPalindrome(x: int) -> bool:
#     if x < 0:
#         return False
#     if x % 10 == 0:
#         return False
#     poop = x
#     reverse = 0
#     while x != 0:
#         reverse = reverse*10 + (x % 10)
#         x //= 10
#     print(reverse)
#     print(x)
#     if poop == reverse:
#         return True
#     return False

# print(isPalindrome(50505))


#   ========================================

    # With math (slower)
# def findNumbers(nums) -> int:
#     count = 0
#     for i in range(1,7,2):
#         for num in nums:
#             if num >= 10**i and num < 10**(i+1):
#                 print("i",i)
#                 print(10**0)
#                 count += 1
#     return count

# def findNumbers(nums) -> int:
#     count = 0
#     for num in nums:
#         count += len(str(num)) % 2 == 0
#     return count
# print(findNumbers([12,345,2,6,7896]))


#   ========================================

# def subtractProductAndSum(n: int) -> int:
#     mult = 1
#     add = 0
#     while n != 0:
#         mult *= (n % 10)
#         add += (n % 10)
#         n //= 10
#     return mult - add
# print(subtractProductAndSum(234))


#   ==================== GEN STR W/ CHARS THAT HAVE ODD COUNTS ====================

# def generateTheString(n: int) -> str:
#     my_str = ""
#     if n % 2 == 1:
#         my_str += "a"*n
#     else:
#         my_str += "a"*(n-1)
#         my_str += "b"
#     return my_str

# print(generateTheString(6))


#   ==================== MATRIX CELLS IN DISTANCE ORDER ====================

# R = 2
# C = 3
# r0 = 1
# c0 = 2
# def allCellsDistOrder(R: int, C: int, r0: int, c0: int):
#     matrix = []
#     ordered = []
#     myDict = {}
#     for r in range(R):
#         for c in range(C):
#             matrix += [[r, c]]
#     print(matrix)
#     for cell in matrix:
#         dist = abs(r0-cell[0])+abs(c0-cell[1])
#         if dist in myDict:
#             myDict[dist] += [cell]
#         else:
#             myDict[dist] = [cell]
#     i = 0
#     while i in myDict:
#         ordered += myDict[i]
#         i += 1
#     print(myDict)
#     print(ordered)

# allCellsDistOrder(R,C,r0,c0)


#   ==================== ODD CELLS ====================

    # matrix = [[0]*m]*n

# def oddCells(n, m, indices) -> int:
#     matrix = [[0]*m for i in range(n)]
#     for pair in indices:
#         matrix[pair[0]] = [x+1 for x in matrix[pair[0]]]
#         for j in range(len(matrix)):
#             matrix[j][pair[1]] += 1
#     count = 0
#     for row in matrix:
#         for col in row:
#             if col % 2 == 1:
#                 count += 1
#     return count

# print(oddCells(2,2,[[1,1],[0,0]]))


#   ==================== MAX NUM OF BALLOONS ====================

# def maxNumberOfBalloons(text: str) -> int:
#     text_dict = {}
#     for char in text:
#         if char in "balloon":
#             if char in text_dict:
#                 text_dict[char] += 1
#             else:
#                 text_dict[char] = 1
#     print(text_dict)
#     min = 2**64
#     for char in "balon":
#         if char not in text_dict:
#             return 0
#         elif char == "l" or char == "o":
#             text_dict[char] //= 2
#         if text_dict[char] < min:
#             min = text_dict[char]
#     return min

# def maxNumberOfBalloons(text: str) -> int:
#     balloon_dict = {}
#     text_dict = {}
#     for char in "balloon":
#         if char in balloon_dict:
#             balloon_dict[char] += 1
#         else:
#             balloon_dict[char] = 1
#     for char in text:
#         if char in balloon_dict:
#             if char in text_dict:
#                 text_dict[char] += 1
#             else:
#                 text_dict[char] = 1
#     count = 0
#     while True:
#         for key in balloon_dict:
#             if key not in text_dict:
#                 return 0
#             text_dict[key] -= balloon_dict[key]
#             if text_dict[key] < 0:
#                 return count
#         count += 1
#     return count

# print(maxNumberOfBalloons("nlaebolko"))


#   ==================== NUM STEPS TOP REDUCE A NUMBER TO 0 ====================

# num = 14

# def numberOfSteps (num: int) -> int:
#     count = 0
#     while num != 0:
#         if num % 2 == 0:
#             num /= 2
#             count += 1
#         else:
#             num -= 1 
#             count += 1
#     return count

# print(numberOfSteps(num))

#   ==================== strStr ====================

# haystack = "mississippi"
# needle = "pi"

    # Most thoughtful solution
# def strStr(haystack: str, needle: str) -> int:
#     if len(needle) == 0:
#         return 0
#     for i in range(len(haystack)):
#         if haystack[i] == needle[0] and i+len(needle) <= len(haystack):
#             if haystack[i:i+len(needle)] == needle:
#                 print(haystack[i:i+len(needle)], i)
#                 print(needle)
#                 return i
#     return -1

    # Using .find (fastest)
# def strStr(haystack: str, needle: str) -> int:
#     if needle == "":
#         return 0
#     if needle in haystack:
#         return haystack.find(needle)
#     return -1

    # This breaks when the inputs are really long
# def strStr(haystack: str, needle: str) -> int:
#     if needle == "":
#         return 0
#     if len(needle) > len(haystack):
#         return -1
#     i = 0
#     while i < len(haystack):
#         needleCount = len(needle)
#         if haystack[i] == needle[0]:
#             print(haystack[i])
#             print(i)
#             for j in range(i, len(haystack)):
#                 if haystack[j] == needle[j-i]:
#                     needleCount -= 1
#                 else:
#                     print(i)
#                     break
#                 if needleCount == 0:
#                     return i
#         i += 1
#     return -1

# print(strStr(haystack,needle))


#   ==================== ADD TWO NUMBERS ====================

# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
#     def addTwoNumbers(self, l1, l2):
#         def getTotal(list, total):
#             myList = []
#             runner = list
#             while runner != None:
#                 myList.append(runner.val)
#                 runner = runner.next
#             for i in reversed(range(len(myList))):
#                 total += myList[i]*10**(i)
#             return total
#         total = 0
#         total = getTotal(l1, total)
#         total = getTotal(l2, total)
#         l3 = ListNode(total % 10)
#         total //= 10
#         runner = l3
#         while total != 0:
#             runner.next = ListNode(total % 10)
#             runner = runner.next
#             total //= 10
#         return l3

# list_one = ListNode(2)
# runner = list_one
# runner.next = ListNode(4)
# runner = runner.next
# runner.next = ListNode(3)

# list_two = ListNode(5)
# runner = list_two
# runner.next = ListNode(6)
# runner = runner.next
# runner.next = ListNode(4)

# list_three = ListNode()
# print(list_three.addTwoNumbers(list_one,list_two).val)


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