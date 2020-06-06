#   ==================== 414 ======================

    def thirdMax(self, nums: List[int]) -> int:
        m1 = m2 = m3 = -2**32
        se = set()
        for i in range(len(nums)):
            if nums[i] not in se:
                if nums[i] > m1:
                    m3, m2, m1 = m2, m1, nums[i]
                elif nums[i] > m2:
                    m3, m2 = m2, nums[i]
                elif nums[i] > m3:
                    m3 = nums[i]
            se.add(nums[i])
        if len(se) < 3:
            return max(se)
        return m3


#   ==================== 409 ======================

def checkRecord(s: str) -> bool:
    if "A" in s and "LLL" in s:
        return False
    return True

print(checkRecord("PPALLL"))


#   ==================== 409 ======================

# def longestPalindrome(s: str) -> int:
#     dic = {}
#     tot = 0
#     for c in s:
#         dic[c] = 1 if c not in dic else dic[c] + 1
#     odd = True
#     for val in dic.values():
#         if val & 1 == 0:
#             tot += val
#         elif val & 1 == 1:
#             maxOdd = val
#             tot += val-1
#     tot += 1 if odd else 0
#     return tot

# print(longestPalindrome("aababajds"))


#   ==================== 1323 ======================

# def maximum69Number (num: int) -> int:
#     lenNum = 0
#     idx = -1
#     temp = num
#     while temp != 0:
#         if temp % 10 == 6:
#             idx = lenNum
#         temp //= 10
#         lenNum += 1
#     if idx > -1:
#         num += 3*(10**idx)
#     return num

    # Converting to String
# def maximum69Number (num: int) -> int:
#     st = str(num)
#     for i in range(len(st)):
#         if st[i] == '6':
#             st = st[:i] + '9' + st[i+1:]
#             break
#     return st

# print(maximum69Number(6996))


# def removeOuterParentheses(S: str) -> str:
#     if not S:
#         return ""
#     temp, count = [], 0
#     for i in range(len(S)):
#         if S[i] == '(':
#             count += 1
#             temp += S[i] if count > 1 else ""
#         else:
#             count -= 1
#             temp += S[i] if count > 0 else ""
#     S = ''.join(temp)
#     return S

# print(removeOuterParentheses("(()())(())(()(()))"))


# def toHex(num: int) -> str:
#     if num == 0: 
#         return '0'
#     if num < 0: 
#         num = num + 2**32
#     a = ord('a')
#     hexVals = {i:f'{i}' if i < 10 else chr(a+(i-10)) for i in range(16)}

#     hexNum = ""
#     while num != 0:
#         hexNum += hexVals[num % 16]
#         num //= 16
#     return hexNum[::-1]

# print(toHex(-1))


#   ==================== 1436 ======================

# def destCity(paths) -> str:

#     dic = {a:b for a,b in paths}
#     for val in dic.values():
#         if val not in dic:
#             return val

# print(destCity([["B","C"],["D","B"],["C","A"]]))


#   ==================== 709 ======================

# def toLowerCase(self, str: str) -> str:
#     return

# print(toLowerCase())


# def readBinaryWatch(num: int):

#     poss = [[h,m] for h in range(12) for m in range(60)]
#     ans = []
#     for pair in poss:
#         if (bin(pair[0])+bin(pair[1])).count('1') == num:
#             ans += [f"{int(pair[0])}:{int(pair[1]):02}"]
#     return ans

# print(readBinaryWatch(3))


# def isSubsequence(s: str, t: str) -> bool:#     i = 0
#     for char in s:
#         while i < len(t):
#             if char == t[i]:
#                 i += 1
#                 break
#             i += 1
#         else:
#             return False
#     return True

# print(isSubsequence(s = "acb", t = "ahbgdc"))


#   ==================== 383 ======================

# def canConstruct(ransomNote: str, magazine: str) -> bool:
#     mag = {}
#     for c in magazine:
#         if c not in mag:
#             mag[c] = 1
#         else:
#             mag[c] += 1
#     for c in ransomNote:
#         if c not in mag or mag[c] - 1 < 0:
#             return False
#         mag[c] -= 1
#     return True

# print(canConstruct('aabb', 'aaababa'))


# def guessNumber(n: int) -> int:
#     l, r = 0, n
#     m = (l+r) // 2
#     while l <= r:
#         if guess(m) == 0:
#             return m
#         elif guess(m) == -1:
#             m = r-1
#         elif guess(m) == 1:
#             m = l+1
#         m = (l+r) // 2
#     return False

# print(guessNumber(10))


#   ==================== 168 ======================

# def convertToTitle(n: int) -> str:
#     uni = ord('A')
#     vals = {i-uni:chr(i) for i in range(uni, uni+26)}
#     col = ""
#     while n != 0:
#         col += vals[(n-1) % 26]
#         n = (n-1) // 26
#     return col[::-1]

# print(convertToTitle(28))


#   ==================== 1266 ======================

# def minTimeToVisitAllPoints(points) -> int:
#     start = points[0]
#     seconds = 0
#     for point in points:
#         if start != point:
#             diff1 = abs(point[0] - start[0])
#             diff2 = abs(point[1] - start[1])
#             seconds += diff1
#             if diff2 > diff1:
#                 seconds = (seconds - diff1) + diff2
#         start = point
#     return seconds


# def minTimeToVisitAllPoints(points) -> int:
#     start = points[0]
#     seconds = 0
#     for point in points:
#         while start != point:
#             if point[0] > start[0]:
#                 start[0] += 1
#             elif point[0] < start[0]:
#                 start[0] -= 1
#             if point[1] > start[1]:
#                 start[1] += 1
#             elif point[1] < start[1]:
#                 start[1] -= 1
#             seconds += 1
#     return seconds

# print(minTimeToVisitAllPoints([[1,1],[3,4],[-1,0]]))


#   ==================== 292 ======================

# def canWinNim(n: int) -> bool:
    
#     return

# print(canWinNim)


#   ==================== 290 ======================

# def wordPattern(pattern: str, str: str) -> bool:
#     dic, lis, se = {}, str.split(), set()
#     if len(lis) != len(pattern):
#         return False
#     for i, c in enumerate(pattern):
#         if c not in dic and lis[i] not in se:
#             dic[c] = lis[i]
#             se.add(lis[i])
#         elif c not in dic or dic[c] != lis[i]:
#             return False
#     return True

# print(wordPattern(pattern = "abba", str = "dog dog dog dog"))


#   ==================== 278 ======================

    # def firstBadVersion(self, n):
        
    #     l, r, = 1, n
    #     m = (l + r) // 2
    #     v = 0
    #     while l <= r:
    #         if isBadVersion(m):
    #             v = m
    #             r = m - 1
    #             m = (l + r) // 2
    #         else:
    #             l = m + 1
    #             m = (l + r) // 2
    #     return v


#   ==================== 1290 ======================

#     # My solution
# def getDecimalValue(self, head: ListNode) -> int:
#     binStr = ""
#     while head:
#         binStr += str(head.val)
#         head = head.next
#     return int(binStr, 2)

#     # This demonstrates how to use math to convert from binary to decimal (found in discussion)
# def getDecimalValue(self, head: ListNode) -> int:
#     dec = 0
#     while head:
#         dec = dec * 2 + head.val
#         head = head.next
#     return dec


#   ==================== 263 ======================

# def isUgly(num: int) -> bool:
#     if num < 1:
#         return False
#     for n in [2, 3, 5]:
#         while num % n == 0:
#             num /= n
#     return num == 1

# print(isUgly(14))


#   ==================== 258 ======================

# def addDigits(num: int) -> int:
#     while num > 9:
#         temp = 0
#         while num != 0:
#             temp += (num % 10)
#             num //= 10
#         num = temp
#     return num

# print(addDigits(38))


#   ==================== 1450 ======================

# def busyStudent(startTime, endTime, queryTime: int) -> int:
#     return sum(queryTime >= i[0] and queryTime <= i[1] for i in zip(startTime, endTime))

# print(busyStudent(startTime = [4], endTime = [4], queryTime = 5))


#   ====================  ======================

# def canBeEqual(target, arr) -> bool:
#     for i in range(len(target)-1):
#         for j in range(i, len(arr)):
#             if target[i] == arr[j]:
#                 arr = arr[:i] + arr[i:j+1][::-1] + arr[j+1:]
#                 print(arr)
#                 break
#     return target == arr

# print(canBeEqual(target = [1,2,3,4], arr = [2,4,1,3]))


#   ====================  ======================

# def binaryTreePaths(self, root: TreeNode) -> List[str]:
    
#     def find_paths(node, st):
#         if not node:
#             return None
#         if node.left == node.right:
#             return [st + str(node.val)]
        
#         st += str(node.val)+"->"
#         left = find_paths(node.left, st)
#         right = find_paths(node.right, st)
        
#         if left and right:
#             return left + right
#         elif left: 
#             return left
#         return right
    
#     return find_paths(root, "")


#   ==================== 938 ======================

# def rangeSumBST(root, L: int, R: int) -> int:
#     if not root:
#         return 0
    
#     def count_em_up(node):
#         if not node:
#             return 0
#         left = count_em_up(node.left); 
#         right = count_em_up(node.right)
#         if node.val >= L and node.val <= R:
#             return node.val + left + right
#         return left + right
    
#     return count_em_up(root)

#     # Using nonlocal
# def rangeSumBST(self, root: TreeNode, L: int, R: int) -> int:
#     if not root:
#         return 0
#     count = 0

#     def count_em_up(node):
#         nonlocal count
#         if not node:
#             return
#         if node.val >= L and node.val <= R:
#             count += node.val
#         count_em_up(node.left); count_em_up(node.right)

#     count_em_up(root)

#     return count


#   ==================== 235 ======================

# def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':

    # def find_ancestors(node, pointer):
    #     if not node:
    #         return [False, pointer]
    #     if pointer and node.val == p.val or pointer and node.val == q.val:
    #         return [True, pointer]
    #     elif node.val == p.val or node.val == q.val:
    #         pointer = node
    #     left = find_ancestors(node.left, pointer)
    #     right = find_ancestors(node.right, pointer)
    #     if left[0] == True:
    #         return left
    #     if right[0] == True:
    #         return right
    #     if left[1] and right[1] and left[1].val != right[1].val:
    #         return [True, node]
    #     if left[1]:
    #         return left
    #     return right

    # return find_ancestors(root, None)[1]



# def maxProduct(nums) -> int:
#     if len(nums) < 2:
#         return (nums[0]-1) * (nums[1]-1)
#     max1, max2 = nums[0], nums[1]
#     if nums[0] < nums[1]:
#         max1, max2 = nums[1], nums[0]
#     for i in range(2,len(nums)):
#         if nums[i] > max1:
#             max2, max1 = max1, nums[i]
#         elif nums[i] > max2:
#             max2 = nums[i]
#     return (max1-1) * (max2-1)

# print(maxProduct([3,4,5,2]))


# def createTargetArray(nums, index):
    # target = []
    # for i, num in enumerate(nums):
    #     target.insert(index[i], num)
    # return target

# print(createTargetArray(nums = [0,1,2,3,4], index = [0,1,2,2,1]))


# def isIsomorphic(s: str, t: str) -> bool:
#     dic, se = {}, set()
#     for i, char in enumerate(s):
#         if char not in dic and t[i] in se:
#             return False
#         elif char not in dic:
#             dic[char] = t[i]
#             se.add(t[i])
#         elif dic[char] != t[i]:
#             return False
#     return True

# print(isIsomorphic(s = "abb", t = "acd"))


#   ==================== 1313 ======================

# def decompressRLElist(nums):
#     lis = []
#     for i in range(0,len(nums),2):
#         sub_list = [nums[i+1] for j in range(nums[i])]
#         lis += sub_list
#     return lis

# def decompressRLElist(nums):
#     lis = [nums[i+1] for i in range(0,len(nums),2) for j in range(nums[i])]
#     return lis

# print(decompressRLElist(nums = [1,2,3,4]))


#   ==================== 1365 ======================

# def smallerNumbersThanCurrent(nums):
    # lis = [0 for num in nums]
    # for i in range(len(nums)-1):
    #     for j in range(i+1, len(nums)):
    #         if nums[i] > nums[j]:
    #             lis[i] += 1
    #         elif nums[i] < nums[j]:
    #             lis[j] += 1
    # return lis

#     count = [0] * 102
#     for num in nums:
#         count[num+1] += 1    
#     print(count)
#     for i in range(1, 102):
#         count[i] += count[i-1]
#     print(count)

# print(smallerNumbersThanCurrent(nums = [8,1,2,2,3]))


#   ================== REVIEW =====================
#   ==================== 168 ======================
#   =================== REVIEW ====================


# def convertToTitle(n: int) -> str:
#     dic = {i-ord("A"):chr(i) for i in range(ord("A"),ord("A")+26)}
#     print(dic)
#     col = ""
#     while n != 0:
#         col += dic[(n-1) % 26]
#         n = (n-1) // 26
#     return col[::-1]

# print(convertToTitle(27))


# def numJewelsInStones(J: str, S: str) -> int:
#     return

# print(numJewelsInStones())


#   ==================== 167 ======================

# def twoSum(numbers, target: int):
#     i = 0
#     while i < len(numbers)-1:
#         l, r = i+1, len(numbers)-1
#         id1 = numbers[i]
#         while l <= r:
#             m = (l + r) // 2
#             if id1 + numbers[m] == target:
#                 return [i+1, m+1]
#             elif id1 + numbers[m] > target:
#                 r = m - 1
#             elif id1 + numbers[m] < target:
#                 l = m + 1
#         i += 1

    # for i in range(len(numbers)-1):
    #     for j in range(i+1, len(numbers)):
    #         if numbers[i] + numbers[j] == target:
    #             return [i+1, j+1]
    #         if numbers[i] + numbers[j] > target:
    #             break

# print(twoSum([2,7,11,15], 9))


#   ================== REVIEW =====================
#   ==================== 111 ======================
#   =================== REVIEW ====================

    # # Won't run b/c it requires TreeNode class, struggled with this alot but I'm proud of my solution
    # def minDepth(self, root: TreeNode) -> int:
        
    #     if not root:
    #         return 0
    #     if root.left == root.right:
    #         return 1
        
    #     def find_leaf(node):
    #         if not node:
    #             return 0
            
    #         left = find_leaf(node.left)
    #         right = find_leaf(node.right)
            
    #         if left == 0 or right == 0:
    #             return 1 + max(left, right)
            
    #         return 1 + min(left, right)
        
    #     return find_leaf(root)


#   =================== REVIEW =====================
#   ===================== 110 ====================
#   ==================== REVIEW ====================

#     # After looking at discussion -- much better solution
# def isBalanced(self, root: TreeNode) -> bool:

#     def checker(node):
#         if not node:
#             return 0
        
#         left = checker(node.left)
#         right = checker(node.right)
        
#         if left == -1 or right == -1 or abs(left - right) > 1:
#             return -1
        
#         return 1 + max(left, right)

#     return checker(root) != -1

#     # How I did it first (took me 6 hours, see above for better solution)
# def isBalanced(self, root: TreeNode) -> bool:
#     if not root:
#         return True

#     def go_diggin(left, right):
#         if left == right:
#             return True
        
#         depth = 0
        
#         def find_depth(node, depth):
#             if not node:
#                 return depth
#             depth += 1
#             if node.left == node.right:
#                 return depth
#             return max(find_depth(node.left, depth), find_depth(node.right, depth))
            
#         diff = abs(find_depth(left, depth) - find_depth(right, depth))
#         if diff > 1:
#             return False
        
#         if left and right:
#             return go_diggin(left.left, left.right) and go_diggin(right.left, right.right)
#         elif left:
#             return go_diggin(left.left, left.right)
#         elif right:
#             return go_diggin(right.left, right.right)

#     return go_diggin(root.left, root.right)


#   ==================== 107 ====================

#     # Won't run b/c it requires TreeNode class, but the solution I can up with made me \
#     # feel like a genius so I pasted it here for the record
# def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
#     if not root:
#         return root
#     my_dict = {0: [root.val]}
#     level = 1
#     left = root.left
#     right = root.right
#     def bottoms_up(l, r, lv):
#         if l == r:
#             return None
#         nonlocal my_dict
#         if lv not in my_dict:
#             my_dict[lv] = []
#         if l:
#             my_dict[lv] += [l.val]
#             bottoms_up(l.left, l.right, lv+1)
#         if r:
#             my_dict[lv] += [r.val]
#             bottoms_up(r.left, r.right, lv+1)
#     bottoms_up(left, right, level)
#     my_list = [val for val in my_dict.values()]
#     my_list.reverse()
#     return my_list


#   ==================== 100 ====================

#     # I literally feel like a genius for implementing a try/except clause in this. I also passed \
#     # this on my first attemp. It won't run here because it required a TreeNode class but I had \
#     # to paste it here for the record
# def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
#     if not p and not q:
#         return True
#     if not p or not q or p.val != q.val:
#         return False
    
#     def lets_fuckin_go(l, r):
#         if l == r:
#             return True
#         try:
#             if l.val == r.val:
#                 return lets_fuckin_go(l.left, r.left) and lets_fuckin_go(l.right, r.right)
#         except:
#             return False
        
#     return lets_fuckin_go(p.left, q.left) and lets_fuckin_go(p.right, q.right)


#   ==================== 58 ====================

# def lengthOfLastWord(s: str) -> int:
#     my_list = s.split()
#     print(my_list)
#     for word in reversed(my_list):
#         if word:
#             return len(word)
#     return 0

# print(lengthOfLastWord("Hello World"))


#   ==================== 35 ====================

# def searchInsert(nums, target: int) -> int:
#     for i, num in enumerate(nums):
#         if num == target or num > target:
#             return i
#     else:
#         return len(nums)
#     return 0

# print(searchInsert([7,7,7,7], 5))


#   ==================== 202 ====================

# def isHappy(n: int) -> bool:
#     my_set = set()
#     while True:
#         num1 = 0
#         while n:
#             num1 += (n % 10)**2
#             n //= 10
#         if num1 == 1:
#             return True
#         if num1 in my_set:
#             return False
#         my_set.add(num1)
#         n = num1

# print(isHappy(1))


#   ================== REVIEW ====================
#   ==================== 198 ====================
#   =================== REVIEW ====================

    # Got from discussion -- UNDERSTAND THIS
# def rob(nums) -> int:
#     prev = curr = 0
#     for num in nums:
#         temp = prev # This represents the nums[i-2]th value
#         prev = curr # This represents the nums[i-1]th value
#         curr = max(num + temp, prev) # Here we just plug into the formula
#     return curr

    # My solution (took me 4+ hours) == not as good as solution from discussion
# def rob(nums) -> int:
    # if not nums:
    #     return 0
    # if len(nums) < 2:
    #     return max(nums)
    # left = nums[0]
    # right = nums[1]
    # maxx = max(left, right)
    # i = 0
    # for i in range(2,len(nums)-1,2):
    #     if maxx == left:
    #         right = left + nums[i+1]
    #         left += nums[i]
    #         maxx = max(left, right)
    #     elif maxx == right:
    #         left += nums[i]
    #         if left < right:
    #             left = right
    #         right += nums[i+1]
    #         maxx = max(left, right)
    # if i+2 == len(nums)-1:
    #     if left + nums[i+2] > right:
    #         maxx = left + nums[i+2]
    # return maxx

# print(rob([8,9,9,4,10,5,6,9,7,9]))


#   ==================== 191 ====================

# def hammingWeight(n: int) -> int:
#     bin_str = f'{n:b}'
#     return bin_str.count("1")

# print(hammingWeight(11))


#   ================== REVIEW ====================
#   ==================== 190 ====================
#   =================== REVIEW ====================

# def reverseBits(n: int) -> int:
#     bit_str = f'{n:b}'
#     bit_str = '0'*(32-len(bit_str)) + bit_str
#     bit_str = bit_str[::-1]
#     return int(bit_str, 2)
# print(reverseBits(43261596))


#   ================== REVIEW ====================
#   ==================== 172 ====================
#   =================== REVIEW ====================

# def trailingZeroes(n: int) -> int:
#     count = 0
#     exp = 1
#     while 5**exp <= n:
#         count += (n // 5**exp)
#         exp += 1
#     return count

# def trailingZeroes(n: int) -> int:
#     my_num = n
#     for i in range(n-1,0,-1):
#         my_num *= i
#     print(my_num)
#     count = 0
#     while my_num != 0:
#         if my_num % 10 != 0:
#             return count
#         count += 1
#         my_num //= 10
#     return count

# print(trailingZeroes(30))


#   ==================== 171 ====================

# def titleToNumber(s: str) -> int:
#     st = ord("A")
#     my_dict = {chr(i):i-st+1 for i in range(st, st+26)}
#     tot = my_dict[s[len(s)-1]]
#     for i in range(len(s)-1):
#         tot += (my_dict[s[i]] * 26 ** (len(s)-i-1))
#     return tot

# print(titleToNumber("A"))


#   ==================== 169 ====================

# def majorityElement(nums) -> int:
#     maj = len(nums) / 2
#     my_dict = {}
#     for num in nums:
#         if num not in my_dict:
#             my_dict[num] = 1
#         else:
#             my_dict[num] += 1
#         if my_dict[num] > maj:
#             return num
#     return False

# print(majorityElement([2,2,1,1,1,2,2]))


# def maxProfit(prices) -> int:
#     if not prices:
#         return 0
#     maxx = minn = prices[0]
#     prof = 0
#     for i in range(1, len(prices)):
#         if prices[i] < minn:
#             if maxx - minn > 0:
#                 prof += (maxx - minn)
#             minn = maxx = prices[i]
#         elif prices[i] > maxx:
#             maxx = prices[i]
#         elif prices[i] < maxx:
#             print(minn, maxx)
#             if maxx - minn > 0:
#                 prof += (maxx - minn)
#             minn = maxx = prices[i]
#     if maxx - minn > 0:
#         prof += maxx - minn
#     return prof

# print(maxProfit([7,1,5,3,6,4]))


#   ==================== 118 ====================

# def maxProfit(prices) -> int:
#     if not prices:
#         return 0
#     min = max = prices[0]
#     prof = 0
#     for i in range(1,len(prices)):
#         if prices[i] < min:
#             min = max = prices[i]
#         elif prices[i] > max:
#             max = prices[i]
#         if max - min > prof:
#             prof = max - min
#     return prof

# print(maxProfit([7,1,5,3,6,4]))


# def generate(numRows: int):
#     my_list = []
#     if numRows == 0:
#         return my_list
#     if numRows == 1:
#         return my_list + [[1]]
#     if numRows == 2:
#         return my_list + [[1], [1, 1]]
#     my_list += [[1], [1, 1]]
#     for i in range(2, numRows):
#         temp = [1]
#         prev = my_list[len(my_list)-1]
#         for j in range(len(prev)):
#             temp.append(sum(prev[j:j+2]))
#         my_list.append(temp)
#     return my_list

# print(generate(1))

#   ====================  ====================


    # Failed sorted array to Binary Search Tree conversion
# if not nums:
#     return None
# if len(nums) == 1:
#     return TreeNode(nums[0])
# mid = len(nums) // 2
# root = TreeNode(nums[mid])
# node = root
# my_stack = []
# for i in range(mid+1, len(nums)):
#     if i > mid+1 and i % 2 == (mid+1) % 2:
#         node.right = TreeNode(my_stack.pop())
#         node.right.left = TreeNode(my_stack.pop())
#         node = node.right
#     my_stack.append(nums[i])
# while my_stack:
#     node.right = TreeNode(my_stack.pop())
#     if my_stack:
#         node.right.left = TreeNode(my_stack.pop())
# node = root

# for j in range(mid-1, -1, -1):
#     if j < mid-1 and j % 2 == (mid-1) % 2:
#         print(my_stack)
#         node.left = TreeNode(my_stack.pop())
#         node.left.right = TreeNode(my_stack.pop())
#         node = node.left
#     my_stack.append(nums[j])
# while my_stack:
#     node.left = TreeNode(my_stack.pop())
#     if my_stack:
#         node.left.right = TreeNode(my_stack.pop())
# return root

#   ==================== BST Symmetry ====================


    # Failed Binary Tree symmetry test
# if not root.left and not root.right:
#     return True
# if not root.left or not root.right:
#     return False

# def left_root(root):
#     if not root.left and not root.right:
#         return True
#     if not root.left or not root.right:
#         return False
#     if root.left.val != root.val*2 - 1 or root.right.val != root.val*2:
#         return False
#     return left_root(root.left) and left_root(root.right)

# def right_root(root):
#     if not root.left and not root.right:
#         return True
#     if not root.left or not root.right:
#         return False
#     if root.right.val != root.val*2 - 1 or root.left.val != root.val*2:
#         return False
#     return right_root(root.left) and right_root(root.right)

#   ==================== 70 ====================

# def climbStairs(n: int) -> int:
#     if n < 3:
#         return n
#     st1 = 1
#     st2 = 2
#     tot = 0
#     for i in range(3,n+1):
#         tot = st1 + st2
#         st1 = st2
#         st2 = tot
#     return tot

# print(climbStairs(5))


#   ==================== 69 ====================

# def mySqrt(x: int) -> int:
#     if x == 1:
#         return 1
#     l = 1
#     r = x // 2
#     m = (l + r) // 2
#     while l < r:
#         if m * m == x:
#             return int(m) // 1
#         elif x > m * m and x < (m+1)*(m+1):
#             return int(m) // 1
#         elif m * m > x:
#             r = m - 1
#         else:
#             l = m + 1
#         m = (l + r) // 2
#     return int(m) // 1

# print(mySqrt(1))


#   ==================== 840 ====================

    # With List comprehension
# def numMagicSquaresInside(grid) -> int:
#     count = 0
#     magic = [grid[i][j:j+3]+grid[i+1][j:j+3]+grid[i+2][j:j+3] 
#         for i in range(len(grid)) if i+2 < len(grid) 
#         for j in range(len(grid[i])) if j+2 < len(grid[i])]
#     for row in magic:
#         if len(set(row)) == 9 and sum(row) == sum(i for i in range(10)) and 0 not in row:
#             if sum(row[:3]) == sum(row[3:6]) == sum(row[6:9]) == \
#             sum(row[::3]) == sum(row[1::3]) == sum(row[2::3]) == \
#             sum(row[::4]) == sum(row[2:8:2]):
#                 count += 1
#     return count

# def numMagicSquaresInside(grid) -> int:
#     for i, row in enumerate(grid):
#         magic = []
#         if i+2 < len(grid):
#             for j in range(len(row)):
#                 if j + 2 < len(row):
#                     magic = grid[i][j:j+3]+grid[i+1][j:j+3]+grid[i+2][j:j+3]
#                     if len(set(magic)) == 9 and sum(magic) == sum(i for i in range(10)) and 0 not in magic:
#                         if sum(magic[:3]) == sum(magic[3:6]) == sum(magic[6:9]) == \
#                         sum(magic[::3]) == sum(magic[1::3]) == sum(magic[2::3]) == \
#                         sum(magic[::4]) == sum(magic[2:8:2]):
#                             count += 1
#     return count

# print(numMagicSquaresInside([[4,3,8,4],[9,5,1,9],[2,7,6,2]]))

# def isPerfectSquare(num: int) -> bool:
#     if num == 1:
#         return True
#     l = 1
#     r = num // 2
#     m = (l+r) // 2
#     while l < r:
#         print("l",l,"r",r,"m",m)
#         if m*m == num:
#             return True
#         elif m*m > num:
#             r = m-1
#         else:
#             l = m+1
#         m = (l+r) // 2
#     # if m*m == num:
#     #     return True
#     return False

# print(isPerfectSquare(598))


#   ==================== 242 ====================

# def isAnagram(s: str, t: str) -> bool:
#     my_dict = {}
#     for char in s:
#         if char not in t:
#             return False
#         if char not in my_dict:
#             my_dict[char] = 1
#         else:
#             my_dict[char] += 1
#     for char in t:
#         if char not in s:
#             return False
#         my_dict[char] -= 1
#         if my_dict[char] < 0:
#             return False
#     for val in my_dict.values():
#         if val != 0: 
#             return False
#     return True

# print(isAnagram(s = "rat", t = "car"))


#   ==================== 412 ====================

# def fizzBuzz(n: int):
#     my_list = []
#     for i in range(1, n+1):
#         if i % 3 == 0 and i % 5 == 0:
#             my_list += ["FizzBuzz"]
#         elif i % 3 == 0:
#             my_list += ["Fizz"]
#         elif i % 5 == 0:
#             my_list += ["Buzz"]
#         else:
#             my_list += [str(i)]
#     return my_list
# print(fizzBuzz(15))


#   ==================== 350 ====================

# def intersect(nums1, nums2):
#     inter = []
#     checked = {}
#     for num in nums1:
#         if num in nums2 and num not in checked:
#             checked[num] = 1
#             m = min(nums1.count(num), nums2.count(num))
#             inter += [num for i in range(m)]
#     return inter

# def intersect(nums1, nums2):
#     inter_dict = {}
#     inter_list = []
#     for num in nums1:
#         if num in nums2:
#             if num not in inter_dict:
#                 inter_dict[num] = 1
#             else:
#                 inter_dict[num] += 1
#     for num in nums2:
#         if num in inter_dict and inter_dict[num] > 0:
#             inter_dict[num] -= 1
#             inter_list.append(num)
#     return inter_list

# print(intersect(nums1 = [1,2,2,1], nums2 = [2,2]))


#   ==================== 349 ====================

# def intersection(nums1, nums2):
#     set1 = set(nums1)
#     set2 = set(nums2)
#     return set1.intersection(set2)

    # This is wrong - I thought an intersection was where there was an identical overlap, so that's what this produces
# def intersection(nums1, nums2):
#     inter = []
#     i = 0
#     for i, num in enumerate(nums1):
#         if num in nums2:
#             i2 = nums2.index(num)
#             break
#     while i < len(nums1) and i2 < len(nums2):
#         if nums1[i] != nums2[i2]:
#             return inter
#         inter.append(nums1[i])
#         i += 1; i2 += 1
#     return set(inter)

# print(intersection(nums1 = [1,2,2,1], nums2 = [2,2]))


#   ==================== 1394 ====================

# def findLucky(arr) -> int:
#     lucky = -1
#     my_dict = {}
#     for num in arr:
#         if num not in my_dict:
#             my_dict[num] = 1
#         else:
#             my_dict[num] += 1
#     for key, val in my_dict.items():
#         print(key, val)
#         if key == val and key > lucky:
#             lucky = key
#     return lucky

# print(findLucky([2,2,2,3,3]))


#   ==================== 989 ====================

# def addToArrayForm(A, K: int):
#     dummy = []
#     car = 0
#     i = len(A)-1
#     while i > -1 or K or car:
#         num = 0 + car
#         if i > -1:
#             num += A[i]
#             i -= 1
#         if K:
#             num += K % 10
#             K //= 10
#         if num > 9:
#             dummy.append(num % 10)
#             car = num // 10
#         else:
#             dummy.append(num)
#             car = 0
#     dummy.reverse()
#     return dummy

# print(addToArrayForm(A = [9,9,9,9], K = 39))


#   ==================== 67 ====================

# def addBinary(a: str, b: str) -> str:
#     temp = 0
#     new_bin = ""
#     while a or b or temp:
#         if a: 
#             temp += int(a[-1])
#             a = a[:-1]
#         if b:
#             temp += int(b[-1])
#             b = b[:-1]
#         new_bin += str(temp % 2)
#         if temp > 1:
#             temp = 1
#         else:
#             temp = 0
#     return new_bin[::-1]

# def addBinary(a: str, b: str) -> str:
#     return bin(int(a, 2) + int(b, 2))[2:]

# print(addBinary("11","1"))


#   ==================== 415 ====================

# def addStrings(num1: str, num2: str) -> str:
#     n1 = 0
#     n2 = 0
#     for num in num1:
#         n1 = n1*10 + int(num)
#     for num in num2:
#         n2 = n2*10 + int(num)
#     return str(n1 + n2)

# print(addStrings("100","200"))


#   ==================== 88 ====================

# def merge(nums1, m: int, nums2, n: int) -> None:
#     while m != 0 and n != 0:
#         if nums1[m-1] >= nums2[n-1]:
#             nums1[m+n-1] = nums1[m-1]
#             m -= 1
#         else:
#             nums1[m+n-1] = nums2[n-1]
#             n -= 1
#     if m > 0:
#         rem = m; lis = nums1
#     elif n > 0:
#         rem = n; lis = nums2
#     while rem != 0:
#         nums1[rem] = lis[rem]
#         rem -= 1

# print(merge(nums1 = [1,2,3,0,0], m = 3, nums2 = [5,6], n = 2))


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

# print(maxSubArray([1]))


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