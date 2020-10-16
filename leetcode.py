#   ====================  ====================

def lengthOfLIS(nums) -> int:
    dp = [1 for i in range(len(nums))]
    ans = 1
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
        ans = max(ans, dp[i])
    return ans

print(lengthOfLIS([10,22,9,33,21,50,41,60,80]))


#   ====================  ====================

# def isRectangleOverlap(rec1, rec2) -> bool:
#     if max(rec1[0], rec2[0]) < min(rec1[2], rec2[2]) and \
#         max(rec1[1], rec2[1]) < min(rec1[3], rec2[3]):
#         return True
#     return False

# print(isRectangleOverlap(rec1 = [0,0,2,2], rec2 = [1,1,3,3]))


#   ====================  ====================

# def numKLenSubstrNoRepeats(S: str, K: int) -> int:
#     if len(S) < K:
#         return 0
    
#     count = 0

#     for i in range(len(S) - K+1):
#         if len(set(S[i:i+K])) == K:
#             count += 1
#     return count

# print(numKLenSubstrNoRepeats(S = "havefunonleetcode", K = 5))


#   ====================  ====================
# from collections import Counter
# from heapq import heapify, heappop, heappush

# def leastInterval(tasks, n: int) -> int:

#     # time, tasksHeap = 0, list(map(lambda x: (-x[1], x[0]), Counter(tasks).items()))
#     # heapify(tasksHeap)

#     time, tasksHeap = 0, []
#     for x, y in Counter(tasks).items():
#         heappush(tasksHeap, (-y, x))

#     idle, queued = n + 1, []
#     while tasksHeap:
#         count, task = heappop(tasksHeap)
#         time, count, idle = time + 1, count + 1, idle - 1
#         if count:
#             queued.append((count, task))
#         if (not tasksHeap and queued) or (not idle):
#             time, idle = time + idle, n + 1
#             while queued:
#                 heappush(tasksHeap, queued.pop())

#     return time

# print(leastInterval(["A","A","A","B","B","B", "C","C","C", "D", "D", "E"], 2))


#   ====================  ====================
# from collections import defaultdict

# def mostVisitedPattern(username, timestamp, website):
#     userVisits = defaultdict(list)
#     for i in range(len(username)):
#         userVisits[username[i]].append((timestamp[i],website[i]))
    
#     patterns = defaultdict(int)
#     for user in userVisits:
#         userVisits[user].sort()
#         se = set()
#         sites = userVisits[user]
#         for i in range(len(sites)-2):
#             for j in range(i+1, len(sites)-1):
#                 for k in range(j+1, len(sites)):
#                     seq = (sites[i][1], sites[j][1], sites[k][1])
#                     if seq not in se:
#                         patterns[seq] += 1
#                     se.add(seq)

#     return next(iter(sorted(patterns, key=lambda x: (-patterns[x], x))))

# print(mostVisitedPattern(
# ["h","eiy","cq","h","cq","txldsscx","cq","txldsscx","h","cq","cq"],
# [527896567,334462937,517687281,134127993,859112386,159548699,51100299,444082139,926837079,317455832,411747930],
# ["hibympufi","hibympufi","hibympufi","hibympufi","hibympufi","hibympufi","hibympufi","hibympufi","yljmntrclw","hibympufi","yljmntrclw"]
# ))


#   ================== REVIEW ====================
#   ==================== 131 ====================
#   ==================== REVIEW ==================

# def partition(s: str):
#     res = []
    
#     def isPal(s):
#         if s == s[::-1]:
#             return True
#         return False
    
#     def splitString(s, path=[]):
#         if not s:
#             res.append(path)
#             return
#         for i in range(1, len(s)+1):
#             sub = s[:i]
#             if isPal(sub):
#                 splitString(s[i:], path+[sub])
                
#     splitString(s)
    
#     return res

# print(partition("aab"))


#   ====================  ====================

# def minSubArrayLen(s: int, nums) -> int:
#     ans = 2**32

#     su = count = i = j = 0
#     for i in range(len(nums)):
#         su, count = su + nums[i], count + 1
#         while su >= s:
#             su, j = su - nums[j], j + 1
#             ans, count = min(ans, count), count - 1
#     return ans if ans < 2**32 else 0

# from collections import deque
# def minSubArrayLen(s: int, nums) -> int:
#     nums = deque(nums)
#     ans = 2**32

#     su = count = i = 0
#     while i < len(nums):
#         su, count = su + nums[i], count + 1
#         if su >= s:
#             while su >= s:
#                 su, i = su - nums.popleft(), i - 1
#                 if su >= s:
#                     count -= 1
#             ans, count = min(ans, count), count - 1
#         i += 1
#     return ans if ans < 2**32 else 0

# print(minSubArrayLen(s = 7, nums = [2,3,1,2,4,3]))


#   ====================  ====================

# def smallestDivisor(nums, threshold: int) -> int:
    
#     mn, mx = 1, max(nums)
#     div = (mn + mx) // 2
#     ans = 10**6
    
#     while mn <= mx:
#         sum = 0
#         for num in nums:
#             sum += (num // div) + 1 if num % div else num // div
#         if sum > threshold:
#             mn = div + 1
#             div = (mn + mx) // 2
#         elif sum <= threshold:
#             ans = min(ans, div)
#             mx = div - 1
#             div = (mn + mx) // 2

#     return ans

# print(smallestDivisor(nums = [1,2,5,9], threshold = 6))


#   ====================  ====================

# def minPathSum(grid) -> int:
#     dp = [[2**32 for i in range(len(grid[0]))] for j in range(len(grid))]
#     dp[0][0] = grid[0][0]

#     for x in range(len(grid)):
#         for y in range(len(grid[0])):
#             if 0 <= x-1 < len(grid):
#                 dp[x][y] = min(dp[x][y], dp[x-1][y] + grid[x][y])
#             if 0 <= y-1 < len(grid[0]):
#                 dp[x][y] = min(dp[x][y], dp[x][y-1] + grid[x][y])

#     return dp[-1][-1]

# print(minPathSum([
#     [7,1,3,5,8,9,9,2,1,9,0,8,3,1,6,6,9,5],
#     [9,5,9,4,0,4,8,8,9,5,7,3,6,6,6,9,1,6],
#     [8,2,9,1,3,1,9,7,2,5,3,1,2,4,8,2,8,8],
#     [6,7,9,8,4,8,3,0,4,0,9,6,6,0,0,5,1,4],
#     [7,1,3,1,8,8,3,1,2,1,5,0,2,1,9,1,1,4],
#     [9,5,4,3,5,6,1,3,6,4,9,7,0,8,0,3,9,9],
#     [1,4,2,5,8,7,7,0,0,7,1,2,1,2,7,7,7,4],
#     [3,9,7,9,5,8,9,5,6,9,8,8,0,1,4,2,8,2],
#     [1,5,2,2,2,5,6,3,9,3,1,7,9,6,8,6,8,3],
#     [5,7,8,3,8,8,3,9,9,8,1,9,2,5,4,7,7,7],
#     [2,3,2,4,8,5,1,7,2,9,5,2,4,2,9,2,8,7],
#     [0,1,6,1,1,0,0,6,5,4,3,4,3,7,9,6,1,9]
# ]))

#   ====================  ====================
# from collections import Counter

# def reorganizeString(S: str) -> str:
#     half = (len(S)+1) // 2
#     counts = Counter(S)
#     ans = [""]*len(S)
#     d = []
#     for key, val in sorted(counts.items(), key=lambda x: -x[1]):
#         if val > half:
#             return ""
#         d.extend([key]*val)
#     for i in range(0,half):
#         ans[i*2] = d[i]
#     for i in range(half, len(d)):
#         ans[1+((i-half)*2)] = d[i]
#     return ''.join(ans)

# print(reorganizeString(S = "zrhmhyevkojpsegvwolkpystdnkyhcjrdvqtyhucxdcwm"))


#   ====================  ====================

# def calPoints(ops) -> int:

#     score = 0
#     stk = []
#     for s in ops:
#         if s != 'C':
#             if s.isnumeric() or s[0] == '-':
#                 stk.append(int(s))
#             elif s == '+':
#                 stk.append(stk[-1] + stk[-2])
#             elif s == 'D':
#                 stk.append(stk[-1] * 2)
#             score += stk[-1]
#         else:
#             score -= stk.pop()
#     return score

# print(calPoints(["5","2","C","D","+"]))


#   ====================  ====================

# def heightChecker(heights) -> int:
#     ordered = sorted(heights)
#     ans = 0
#     for i, student in enumerate(ordered):
#         if student != heights[i]:
#             ans += 1
#     return ans

# print(heightChecker([1,2,1,2,1,1,1,2,1]))


#   ====================  ====================

# def isRobotBounded(instructions: str) -> bool:
#     start = [0,0]
#     directions = [[0,1], [1,0], [0,-1], [-1,0]]
#     i = move = 0
#     while i < 4:
#         for cmd in instructions:
#             if cmd == 'L':
#                 move = move - 1 if move > 0 else 3
#             elif cmd == 'R':
#                 move = move + 1 if move < 3 else 0
#             else:
#                 x, y = directions[move]
#                 start[0] += x
#                 start[1] += y
#         if start == [0,0]:
#             return True
#         i += 1
#     return start == [0,0]

# print(isRobotBounded("GLGLGGLGL"))


#   ====================  ====================

#         # With deque
# def maxCoins(self, piles: List[int]) -> int:
#     piles.sort()
#     piles = deque(piles)
    
#     me = 0
#     while len(piles) > 2:
#         piles.pop()
#         me += piles.pop()
#         if piles:
#             piles.popleft()
#     return me

#         # With 2 pointers
# def maxCoins(self, piles: List[int]) -> int:
#     piles.sort()
    
#     me = 0
#     start, end = 0, len(piles) - 1
#     while start < end:
#         if end-1 > 0:
#             me += piles[end-1]
#         start, end = start+1, end-2
#     return me


#   ====================  ====================
# import math

# def numPrimeArrangements(n: int) -> int:
#     total = 0
#     primes = [0]*(n+1)
#     for i in range(2, n+1):
#         if not primes[i]:
#             total += 1
#             for j in range(i, n+1, i):
#                 primes[j] = 1
#     return (math.factorial(total) * math.factorial(n-total)) % (10**9 + 7)

# print(numPrimeArrangements(n = 20))


#   ====================  ====================
# from collections import Counter

# def relativeSortArray(arr1, arr2):
#     counts = Counter(arr1)
#     ans = []
#     for num in arr2:
#         if num in counts:
#             ans += [num]*counts[num]
#             del counts[num]
#     for num in sorted(counts):
#         ans += [num]*counts[num]
#     return ans

# print(relativeSortArray(arr1 = [2,3,1,3,2,4,6,7,9,2,19], arr2 = [2,1,4,3,9,6]))


#   ====================  ====================
# from collections import Counter

#     # After looking at comments I discovered Counter() has similar functionality to set()! Using &= takes the..
#     # minimum element count from each dict, and removes it entirely if it doesn't exist in one of them!
# def commonChars(A):
#     counts = Counter(A[0])
#     for word in A:
#         counts &= Counter(word)
#     return list(counts.elements())

    # My first solution -- works and is fast, just not as elegant as the one above
# def commonChars(A):
#     common = set(A[0])
#     for i in range(1, len(A)):
#         common.intersection_update(A[i])
#     counts = Counter(A[0])
#     for word in A:
#         temp = Counter(word)
#         for c in temp:
#             if c in common:
#                 counts[c] = temp[c] if not c in counts else min(counts[c], temp[c])
#     ans = []
#     for key, val in counts.items():
#         ans += [key]*val
#     return ans

# print(commonChars(["bella","label","roller"]))


#   ====================  ====================
# from collections import deque

# def ladderLength(beginWord: str, endWord: str, wordList) -> int:
#     if not endWord in wordList:
#         return 0
#     wordList = set(wordList)
#     # dp = {beginWord: 1}
#     st = deque([[beginWord, 1]])
#     charSet = {c for word in wordList for c in word}
#     print(st)
    
#     while st:
#         word, count = st.popleft()
#         # for word in wordList:
#         for i in range(len(word)):
#             for c in charSet:
#                 if word[i] != c:
#                     temp = word[:i] + c + word[i+1:]
#                     if temp == endWord:
#                         return count+1
#                     if temp in wordList:
#                         wordList.remove(temp)
#                         # dp[temp] = dp[word]+1
#                         st.append([temp, count+1])
#     return 0

# def ladderLength(beginWord: str, endWord: str, wordList) -> int:
#     if not endWord in wordList:
#         return 0

#     dp = {beginWord: 1}
#     st = deque([beginWord])
    
#     while st:
#         cur = st.popleft()
#         for word in wordList:
#             for i in range(len(word)):
#                 if word[i] != cur[i]:
#                     temp = cur[:i] + word[i] + cur[i+1:]
#                     if temp in wordList and not temp in dp:
#                         dp[temp] = dp[cur]+1
#                         st.append(temp)
#         if endWord in dp:
#             return dp[endWord]
#     return 0

# print(ladderLength(
#     beginWord = "a",
#     endWord = "c",
#     wordList = ["a","b","c"]
# ))


#   ====================  ====================

#     # IM DOIN IT! ALL BY MASELF! DYNAMIC PROGRAMMING!!
# def numDecodings(s: str) -> int:
    
#     if s[0] == '0':
#         return 0
    
#     tab = [1]*len(s)
    
#     for i in range(1, len(s)):
#         tab[i] = tab[i-1]
#         if s[i] == '0' and (int(s[i-1]) > 2 or s[i-1] == '0'):
#             return 0
#         elif s[i] == '0':
#             continue
#         if 9 < int(s[i-1]) * 10 + int(s[i]) <= 26:
#             if i == len(s) - 1 or s[i+1] != '0':
#                 tab[i] += tab[i-2]

#     return tab[-1]

# print(numDecodings("20622"))


#   ====================  ====================

#     # My second (original) solution after looking up some examples, still only ~25% but ms is ~170 so good enough
# from collections import deque
# def calculate(s: str) -> int:
#     num = 0
#     mem, calc = deque(), []

#     for i in range(len(s)):
#         if s[i].isdigit():
#             num = num*10 + int(s[i])
#         if (i < len(s) - 1 and not s[i+1].isdigit() and calc) or (i == len(s) - 1 and calc):
#             op = calc.pop(); temp = calc.pop()
#             num = temp * num if op == '*' else temp // num
#         if i == len(s) - 1:
#             mem.append(num)
#             break
#         if s[i] in {'+','-'}:
#             mem.append(num); mem.append(s[i])
#             num = 0
#         if s[i] in {'*','/'}:
#             calc.append(num); calc.append(s[i])
#             num = 0

#     ans = mem.popleft()
#     while mem:
#         op = mem.popleft(); num = mem.popleft()
#         ans = ans + num if op == '+' else ans - num

#     return ans

#     # My first solution (5%, ~270ms)
# def calculate(s: str) -> int:

    # def getNum(s, i):
    #     num = ''
    #     while i < len(s) and not s[i] in '*/+-':
    #         num += s[i]
    #         i += 1
    #     return (num, i)

    # order = ['*/','+-']

    # s, curCalc = list(s.replace(' ','')), []
    # i = priority = 0
    # while True:
    #     print(curCalc)
    #     if i >= len(s):
    #         i, priority = 0, priority + 1
    #         s, curCalc = curCalc, []
    #     if priority == 2:
    #         break
    #     while i < len(s):
    #         if s[i] not in order[priority] and s[i] in '*/+-':
    #             curCalc.append(s[i])
    #             i += 1
    #         else:
    #             if i < len(s) and s[i] in order[priority] or i == len(s):
    #                 break
    #             num, i = getNum(s, i)
    #             curCalc.append(num)
    #     if i == len(s):
    #         continue
    #     num, j = getNum(s, i+1)
    #     if s[i] == '*':
    #         res = int(curCalc.pop()) * int(num)
    #     if s[i] == '/':
    #         res = int(curCalc.pop()) // int(num)
    #     if s[i] == '+':
    #         res = int(curCalc.pop()) + int(num)
    #     if s[i] == '-':
    #         res = int(curCalc.pop()) - int(num)
    #     curCalc.append(str(res))
    #     i = j

    # return int(''.join(s))

# print(calculate("1*2-3/4+5*6-7*8+9/10"))


#   ====================  ====================

#     # Time complexity is better than concatenating strings in each step of the loop as opposed to appending
#     # to an array and then joining them all at the end before joining the greater array.. I don't know why
#     # since concatenating strings is supposed to be slower than appening to an array and then doing a join..
# def convert(s: str, numRows: int) -> str:
#     if numRows == 1 or numRows >= len(s):
#         return s
#     grid = ['' for c in s]
#     numRows -= 1

#     x, direction = 0, 'D'
#     for c in s:
#         grid[x] += c
#         if direction == 'D':
#             x += 1
#         if direction == 'U':
#             x -= 1
#         if x == numRows:
#             direction = 'U'
#         if x == 0:
#             direction = 'D'

#     return ''.join(grid)

#     # My first solution.. which is far more interesting but unfortunately slow (5%)
# def convert(s: str, numRows: int) -> str:
#     if numRows == 1 or numRows >= len(s):
#         return s

#     grid = [['' for y in range((len(s)+1) // 2)] for x in range(numRows)]
#     numRows -= 1

#     x, y, direction = 0, 0, 'D'
#     for c in s:
#         grid[x][y] = c
#         if direction == 'D':
#             x += 1
#         if direction == 'U':
#             x, y = x - 1, y + 1
#         if x == numRows:
#             direction = 'U'
#         if x == 0:
#             direction = 'D'
#     return ''.join([''.join(x) for x in grid])

# print(convert(s = "IFEELLIKEADAMNGENIUS", numRows = 8))


#   ====================  ====================

# def exist(board, word: str) -> bool:

#     def checkWord(se, word, i, j):
#         if not word:
#             return True
#         for x, y in [[i+1,j], [i,j+1], [i-1,j], [i,j-1]]:
#             if (x,y) not in se and 0 <= x < len(board) and 0 <= y < len(board[0]):
#                 if board[x][y] == word[0]:
#                     if checkWord(se | {(i,j)}, word[1:], x, y):
#                         return True

#     for i in range(len(board)):
#         for j in range(len(board[0])):
#             if board[i][j] == word[0]:
#                 if checkWord(set(), word[1:], i, j):
#                     return True
#     return False

# print(exist([
#     ["C","A","A"],
#     ["A","A","A"],
#     ["B","C","D"]
# ], "AAB"))


#   ================== REVIEW ====================
#   ==================== 221 ====================
#   ==================== REVIEW ==================

#     # My second attempt after looking at a hint -- great runtime! (~70%)
# def maximalSquare(matrix) -> int:
#     dp = [[0 for y in range(len(matrix[0]))] for x in range(len(matrix))]

#     maximal = 0
#     for x in range(len(matrix)):
#         for y in range(len(matrix[0])):
#             temp = 0
#             if matrix[x][y] == '1':
#                 if 1 <= x and 1 <= y:
#                     temp = min(dp[x-1][y], dp[x][y-1], dp[x-1][y-1])
#             dp[x][y] = int(matrix[x][y]) + temp
#             maximal = max(maximal, dp[x][y])
#     return maximal * maximal

#     # My first solution without looking up hints, slow but I'm proud of it!
# from collections import deque
# def maximalSquare(matrix) -> int:

    # def findSquares(test):
    #     se = set()
    #     while test:
    #         x, y, level = test.popleft()
    #         for i, j in [[x+1, y], [x, y+1], [x+1, y+1]]:
    #             if not (i, j) in se:
    #                 if not 0 <= i < len(matrix) or not 0 <= j < len(matrix[0]) or matrix[i][j] != '1':
    #                     return level - 1
    #                 test.append([i, j, level+1])
    #                 se.add((i, j))
    #     return level

    # maximal = 0
    # for i in range(len(matrix)):
    #     for j in range(len(matrix[0])):
    #         if matrix[i][j] == '1':
    #             sq = findSquares(deque([[i, j, 2]]))
    #             maximal = max(maximal, sq*sq)
                # if maximal >= (len(matrix) - i) * (len(matrix) -  i):
                #     return maximal
    # return maximal

# print(maximalSquare([
#     ["0","1","1","1","1"],
#     ["1","1","1","1","1"],
#     ["1","1","1","1","1"],
#     ["1","1","1","1","1"],
# ]))


#   ================== REVIEW ====================
#   ==================== 322 ====================
#   ==================== REVIEW ==================

#     # Tabularization
# def coinChange(coins, amount: int) -> int:

#     coins.sort()
    
#     tab = [0] + [-1] * (amount)

#     for i in range(coins[0], len(tab)):
#         mn = 2 ** 32
#         for coin in coins:
#             newIdx = i - coin
#             if newIdx >= 0 and tab[newIdx] > -1:
#                 tab[i] = min(mn, tab[newIdx] + 1)
#                 mn = tab[i]

#     return tab[-1]

    # Memoization
# def coinChange(coins, amount: int) -> int:

#     memo = {}

#     def findMin(amount):
#         if amount < 0:
#             return -1
#         if amount == 0:
#             return 0
#         if amount in memo:
#             return memo[amount]

#         mn = 2 ** 32
#         for coin in coins:
#             x = findMin(amount - coin)
#             if x >= 0 and x < mn:
#                 mn = x+1
#         memo[amount] = -1 if mn == 2 ** 32 else mn
#         return memo[amount]

#     return findMin(amount)

# print('coinChange()', coinChange([397,417,24,44,235], 3383))


#   ================== REVIEW ====================
#   ==================== 200 ====================
#   ==================== REVIEW ==================

#     # Commented out shows my first instinct -- I was overcomplicating the shit out of it....
# def numIslands(grid) -> int:
    
#     if not grid:
#         return 0

#     # x = y = 0
#     # visited = set()
#     noIslands = 0

#     def findIsland(x, y):
#         grid[x][y] = 'X'
#         for i, j in [[x+1,y], [x,y+1], [x-1,y], [x,y-1]]:
#             if 0 <= i < len(grid) and 0 <= j < len(grid[0]) and grid[i][j] == '1':
#                 findIsland(i, j)

#     for i in range(len(grid)):
#         for j in range(len(grid[0])):
#             if grid[i][j] == '1':
#                 noIslands += 1
#                 findIsland(i, j)

#     # def dfs(x, y):
#     #     nonlocal noIslands
#     #     visited.add((x,y))
#     #     if grid[x][y] == '1':
#     #         noIslands += 1
#     #         findIsland(x, y)
#     #     for i, j in [[x+1,y], [x,y+1]]:
#     #         if not (i,j) in visited and 0 <= i < len(grid) and 0 <= j < len(grid[0]):
#     #             dfs(i, j)

#     # dfs(x, y)
#     return noIslands

# print(numIslands(grid = [
#     ["1","1","1"],
#     ["0","1","0"],
#     ["1","1","1"]
# ]))


#   ====================  ====================

# def toHexspeak(num: str) -> str:
#     se = {
#         "a": "A", 
#         "b": "B", 
#         "c": "C", 
#         "d": "D", 
#         "e": "E", 
#         "f": "F", 
#         '1': "I", 
#         '0': "O"}
#     toHex = hex(int(num))
#     print(toHex)
#     ans = []
#     for c in toHex.replace('0x', ''):
#         if c not in se:
#             return 'ERROR'
#         ans.append(se[c])
#     return ''.join(ans)

# print(toHexspeak(num = "747823223228"))


#   ====================  ====================

# def minStartValue(nums) -> int:
#     mn = runningSum = 0
#     for num in nums:
#         runningSum += num
#         if runningSum < mn:
#             mn = runningSum
#     return 1 - mn

# print(minStartValue(nums = [8,2,5,8,-3,4,2]))


#   ================== REVIEW ====================
#   ==================== 1013 ====================
#   ==================== REVIEW ==================

#     # From leetcode comments -- MUCH better than my first solution
# def canThreePartsEqualSum(A) -> bool:
#     average, remainder, part, cnt = sum(A) // 3, sum(A) % 3, 0, 0
#     for i, a in enumerate(A):
#         part += a
#         if part == average:
#             cnt += 1
#             part = 0
#         if cnt == 2 and i < len(A) - 1 and not remainder:
#             return True
#     # return not remainder and cnt >= 3
#     return False

#     # My first attemp (BARELY passes -- 5%)
# def canThreePartsEqualSum(A) -> bool:
#     sumsOne = {}
#     runningSum = 0
#     for i in range(len(A)):
#         runningSum += A[i]
#         if runningSum not in sumsOne:
#             sumsOne[runningSum] = i
#     runningSum = 0
#     se = set()
#     for i in range(len(A)-1,-1,-1):
#         runningSum += A[i]
#         if runningSum not in se and runningSum in sumsOne:
#             idx = sumsOne[runningSum]
#             if idx < i:
#                 test = A[idx+1:i]
#                 if len(test) > 0 and sum(test) == runningSum:
#                     return True
#         se.add(runningSum)
#     return False

# print(canThreePartsEqualSum([10,-10,10,-10,10,-10,10,-10]))


#   ====================  ====================

# def sortArrayByParity(A):
#     evens, odds = [], []
#     for num in A:
#         if num & 1 == 1:
#             odds.append(num)
#         else:
#             evens.append(num)
#     return evens + odds

# print(sortArrayByParity([3,1,2,4]))


#   ====================  ====================

# def countGoodTriplets(arr, a: int, b: int, c: int) -> int:
#     ans = 0
#     for i in range(len(arr)-2):
#         for j in range(i+1, len(arr)-1):
#             if abs(arr[i] - arr[j]) <= a:
#                 for k in range(j+1, len(arr)):
#                     if abs(arr[j] - arr[k]) <= b and abs(arr[i] - arr[k]) <= c:
#                         ans += 1
#     return ans

# print(countGoodTriplets(arr = [3,0,1,1,9,7], a = 7, b = 2, c = 3))


#   ====================  ====================

# def letterCombinations(digits: str):
    
#     phone = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
#     ans = []

#     def doTheThing(d, comb=[]):
#         if not d:
#             ans.append(''.join(comb))
#             return
#         for c in phone[d[0]]:
#             doTheThing(d[1:], comb+[c])
#     doTheThing(digits)

#     return ans

# print(letterCombinations(""))


#   ====================  ====================

#     # Turns out there were no constraints...
# def peakIndexInMountainArray(A) -> int:
#     for i in range(1, len(A) - 1):
#         if A[i] > A[i+1]:
#             return i

#     # My first solution assuming there were constraints (like maybe it wasn't actually a mountain)
# def peakIndexInMountainArray(A) -> int:
#     if len(A) < 3 or A[0] >= A[1]:
#         return False
#     change = 0
#     peak = 0
#     for i in range(1, len(A) - 1):
#         if not change and A[i] > A[i+1]:
#             change += 1
#             peak = i
#         if change and A[i] < A[i+1]:
#             return False
#         if A[i] == A[i + 1]:
#             return False
#     return peak

# print(peakIndexInMountainArray([0,1,2,1,0,-1]))


#   ====================  ====================

# def checkPossibility(nums) -> bool:
#     mod = 0
#     for i in range(1, len(nums)):
#         if nums[i] < nums[i-1]:
#             mod += 1
#             if i > 1:
#                 if nums[i-2] <= nums[i]:
#                     nums[i-1] = nums[i-2]
#                 elif nums[i-2] > nums[i]:
#                     nums[i] = nums[i-1]
#             else:
#                 nums[i-1] = nums[i]
#         if mod > 1:
#             return False
#     return True

# print(checkPossibility(nums = [-1,4,2,3]))


#   ==================== 1382 ====================

#     # My second solution after refactoring which is very good!
# def balanceBST(self, root: TreeNode) -> TreeNode:
    
#     vals = []

#     def traverse(node):
#         if not node:
#             return
#         traverse(node.left)
#         vals.append(node.val)
#         traverse(node.right)
#     traverse(root)

#     i = len(vals) // 2
#     balTree = TreeNode(vals[i])
    
#     def bal(node, lis):
#         if not lis:
#             return
#         i = len(lis) // 2
#         val = lis[i]

#         if val < node.val:
#             node.left = TreeNode(val)
#             bal(node.left, lis[:i]); bal(node.left, lis[i+1:])
#         else:
#             node.right = TreeNode(val)
#             bal(node.right, lis[:i]); bal(node.right, lis[i+1:])
#     bal(balTree, vals[:i]); bal(balTree, vals[i+1:])

#     return balTree

#     # My first solution which works but is very slow
# def balanceBST(self, root: TreeNode) -> TreeNode:
    
#     vals = []
#     balTree = None
    
#     def traverse(node):
#         if not node:
#             return
#         traverse(node.left)
#         vals.append(node.val)
#         traverse(node.right)
#     traverse(root)

#     def buildTree(node, val):
#         if not node:
#             nonlocal balancedTree
#             balancedTree = TreeNode(val)
#         elif val < node.val:
#             if not node.left:
#                 node.left = TreeNode(val)
#             else:
#                 buildTree(node.left, val)
#         else:
#             if not node.right:
#                 node.right = TreeNode(val)
#             else:
#                 buildTree(node.right, val)

#     def getVals(lis):
#         if not lis:
#             return
#         i = len(lis) // 2
#         val = lis[i]
#         buildTree(balancedTree, val)

#         getVals(lis[:i]); getVals(lis[i+1:])
#     getVals(nodeVals)

# return balancedTree


#   ====================  ====================

# def findContestMatch(n: int) -> str:

#     matches = list(range(1,n+1))

#     i, j, temp = 0, len(matches) - 1, []
#     while len(matches) > 1:
#         temp.append((matches[i],matches[j]))
#         i, j = i + 1, j - 1
#         if i > j:
#             matches, temp = temp, []
#             i, j = 0, len(matches) - 1

#     return str(matches[0]).replace(' ','')

# print(findContestMatch(8))


#   ====================  ====================

# def arrayPairSum(nums) -> int:
#     nums.sort()
#     out = 0
#     for i in range(0, len(nums), 2):
#         out += nums[i]
#     return out

# print(arrayPairSum([-1,5,-7,2]))


#   ====================  ====================

# def validWordAbbreviation(word: str, abbr: str) -> bool:

#     wIdx = aIdx = 0
#     while aIdx < len(abbr):
#         if wIdx >= len(word):
#             return False
#         if abbr[aIdx] == word[wIdx]:
#             wIdx += 1
#             aIdx += 1
#         elif abbr[aIdx].isdigit():
#             if abbr[aIdx] == '0':
#                 return False
#             temp = aIdx
#             while aIdx < len(abbr) and abbr[aIdx].isdigit():
#                 aIdx += 1
#             wIdx += int(abbr[temp:aIdx])
#         else:
#             return False

#     return wIdx == len(word) and aIdx == len(abbr)

# print(validWordAbbreviation(word = "a", abbr = "10"))


#   ================== REVIEW ====================
#   ==================== 394 ====================
#   ==================== REVIEW ==================

#     # My working attempt -- fast but not very readable. Second solution below from leetcode comments
#         # is much, much better
# def decodeString(s: str) -> str:
    
#     ans = ''
#     st, alSt = [], []

#     i = 0
#     while i < len(s):
#         if s[i].isalpha() and s[i] not in {'[', ']'}:
#             ans += s[i]; i += 1
#         elif s[i].isdigit():
#             point = i
#             while s[i].isdigit():
#                 i += 1
#             st.append(s[point:i])
#             i += 1; temp = ''
#             while st:
#                 while i < len(s) and s[i].isalpha() and s[i] not in {'[', ']'}:
#                     temp += s[i]; i += 1
#                 if s[i].isdigit():
#                     point = i
#                     while s[i].isdigit():
#                         i += 1
#                     st.append(s[point:i])
#                     alSt.append(temp)
#                     temp = ''
#                 elif s[i] == ']':
#                     temp *= int(st.pop())
#                     if alSt:
#                         temp = alSt.pop() + temp
#                 i += 1
#             ans += temp
#     return ans

#     # Amazing solution found in leetcode comments
# def decodeString(s: str) -> str:
#     stack = []; curNum = 0; curString = ''
#     for c in s:
#         if c == '[':
#             stack.append(curString)
#             stack.append(curNum)
#             curString = ''
#             curNum = 0
#         elif c == ']':
#             num = stack.pop()
#             prevString = stack.pop()
#             curString = prevString + num*curString
#         elif c.isdigit():
#             curNum = curNum*10 + int(c)
#         else:
#             curString += c
#     return curString

# print(decodeString("100[leetcode]"))

#     # First attempt.. I tried so hard.. and failed
# def decodeString(s: str) -> str:

#     def getStr(st):
#         nonlocal i
#         for t in range(len(st)):
#             if not st[t].isdigit():
#                 break
#         temp = ''
#         j = t + 1
#         while st[j].isalpha() and st[j] not in {'[', ']'}:
#             temp += st[j]
#             j += 1
#         i += j + 1
#         if st[j] == ']':
#             return temp * int(st[:t])
#         else:
#             return (temp + getStr(st[j:])) * int(st[:t])

#     ans = ''

#     i = 0
#     while i < len(s):
#         if s[i].isalpha() and s[i] not in {'[', ']'}:
#             ans += s[i]
#             i += 1
#         elif s[i].isdigit():
#             ans += getStr(s[i:])

#     return ans


#   ====================  ====================

#     # One of my least favorite algos ever.. the second solution below (found in leetcode comments) is a much cleaner
#         # solution than the first that I came up with below..
# def search(nums, target: int) -> int:
    
#     if len(nums) < 1:
#         return -1
#     if len(nums) == 1:
#         return 0 if nums[0] == target else - 1
    
#     l, r = 0, len(nums) - 1
    
#     while True:
#         m = (l + r) // 2
#         if nums[m] > nums[0]:
#             check = m + 1 if m < len(nums) - 1 else 0
#             if nums[m] > nums[check]:
#                 pivot = check
#                 break
#             l = m + 1
#         elif nums[m] < nums[0]:
#             if nums[m] < nums[m-1]:
#                 pivot = m
#                 break
#             r = m - 1
#         elif m == 0:
#             pivot = 0 if nums[m] < nums[m+1] else 1
#             break
    
#     if nums[pivot] <= target <= nums[-1]:
#         l, r = pivot, len(nums) - 1
#     else:
#         l, r = 0, pivot - 1
    
#     while l <= r:
#         m = (l + r) // 2
#         if nums[m] == target:
#             return m
#         if nums[m] > target:
#             r = m - 1
#         if nums[m] < target:
#             l = m + 1
    
#     return m if nums[m] == target else -1

# def search(nums, target):
#     if not nums:
#         return -1

#     low, high = 0, len(nums) - 1

#     while low <= high:
#         mid = (low + high) / 2
#         if target == nums[mid]:
#             return mid

#         if nums[low] <= nums[mid]:
#             if nums[low] <= target <= nums[mid]:
#                 high = mid - 1
#             else:
#                 low = mid + 1
#         else:
#             if nums[mid] <= target <= nums[high]:
#                 low = mid + 1
#             else:
#                 high = mid - 1

#     return -1

# print(search([1,2,3,4], 1))

#   ====================  ====================

# def repeatedStringMatch(A: str, B: str) -> int:

#     repeat = A
#     max_len = len(B) + len(A)
#     i = 1
#     while B not in repeat:
#         if len(repeat) > max_len:
#             return -1
#         repeat += A
#         i += 1
#     return i

# print(repeatedStringMatch(A = "bbc", B = "bbcbbcbbcb"))


#   ====================  ====================

#     # I overcomplicated this... second solution below is from comments and is much simpler...
# def isToeplitzMatrix(matrix) -> bool:
#     r, c = len(matrix), len(matrix[0])
#     for i in range(r):
#         for j in range(c):
#             if matrix[i][j] == -1:
#                 break
#             x, y = i, j
#             while 0 <= x+1 < r and 0 <= y+1 < c:
#                 if matrix[x][y] != matrix[x+1][y+1]:
#                     return False
#                 matrix[x][y] = -1
#                 x, y = x+1, y+1
#             matrix[x][y] = -1
#     return True

#     # Much simpler solution found in leetcode comments...
# def isToeplitzMatrix(m) -> bool:
#     for i in range(len(m) - 1):
#         for j in range(len(m[0]) - 1):
#             print(i, j)
#             if m[i][j] != m[i + 1][j + 1]:
#                 return False
#     return True

# print(isToeplitzMatrix([
#     [1,2,3,4],
#     [5,1,2,3],
#     [9,5,1,2]
# ]))


#   ====================  ====================

#     # My solution which is very, VERY good -- see next solution for another way to think about
#         # soliving this
# def spiralOrder(matrix):

#     change_di = {'R':'D', 'D':'L', 'L':'U', 'U':'R'}
#     seen = set()
#     ans = []

#     r, c, di = 0, 0, 'R'
#     while True:

#         def findValidCoords(x ,y, d):
#             t1, t2 = x, y
#             try:
#                 if d == 'R':
#                     x, y = x, y+1
#                 elif d == 'L':
#                     x, y = x, y-1
#                 elif d == 'D':
#                     x, y = x+1, y
#                 elif d == 'U':
#                     x, y = x-1, y
#                 if (x, y) in seen or not 0 <= x < len(matrix) or not 0 <= y < len(matrix[0]):
#                     raise IndexError
#                 return [x, y, d]
#             except IndexError:
#                 d = change_di[d]
#                 return findValidCoords(t1, t2, d)
        
#         seen.add((r, c))
#         ans.append(matrix[r][c])
#         if len(seen) == len(matrix[0]*len(matrix)):
#             return ans

#         r, c, di = findValidCoords(r, c, di)

#     # A genius solution I copied from leetcode to review as an alternative approach to this problem
# def spiralOrder(self, matrix):
#     if not matrix or not matrix[0]:
#         return []
#     ans = []
#     m, n = len(matrix), len(matrix[0])
#     u, d, l, r = 0, m - 1, 0, n - 1
#     while l < r and u < d:
#         ans.extend([matrix[u][j] for j in range(l, r)])
#         ans.extend([matrix[i][r] for i in range(u, d)])
#         ans.extend([matrix[d][j] for j in range(r, l, -1)])
#         ans.extend([matrix[i][l] for i in range(d, u, -1)])
#         u, d, l, r = u + 1, d - 1, l + 1, r - 1
#     if l == r:
#         ans.extend([matrix[i][r] for i in range(u, d + 1)])
#     elif u == d:
#         ans.extend([matrix[u][j] for j in range(l, r + 1)])
#     return ans

# print(spiralOrder([
#     [ 1, 2, 3 ],
#     [ 4, 5, 6 ],
#     [ 7, 8, 9 ]
# ]))


#   ====================  ====================

#     # YOU CAN USE BOOLEANS TO REPRESENT ACTUAL INTEGERS AND DO MATH WITH THEM!! WHAAAA?!!
# def maxArea(height) -> int:
    
#     ma = 0
#     l, r = 0, len(height) - 1
#     while l < r:
#         h = min(height[l], height[r])
#         area = h * (r - l)
#         ma = max(ma, area)
#         # if height[l] < height[r]:
#             # l += 1
#         l += height[l] == h
#         # elif height[l] > height[r]:
#             # r -= 1
#         r -= height[r] == h
#         # else:
#         #     l, r = l+1, r-1
#     return ma

#     # Brute Force
# def maxArea(height) -> int:
    # ma = 0
    # for i in range(len(height) - 1):
    #     for j in range(i+1, len(height)):
    #         area = min(height[i], height[j]) * (j - i)
    #         if area > ma:
    #             ma = area
    # return ma

# print(maxArea([1,8,6,2,5,4,8,3,7]))


#   ====================  ====================

# def isMonotonic(A) -> bool:
#     direction = 0
#     for i in range(len(A) - 1):
#         if A[i] > A[i+1]:
#             if direction == -1:
#                 return False
#             direction = 1
#         elif A[i] < A[i+1]:
#             if direction == 1:
#                 return False
#             direction = -1
#     return True

# print(isMonotonic([3,2,3,1]))


#   ====================  ====================

# from collections import defaultdict

# def findShortestSubArray(nums) -> int:

#     counts, dist, check = defaultdict(int), {}, []
#     ma = 0
#     for i, num in enumerate(nums):
#         counts[num] += 1
#         dist[num] = [i,i] if num not in dist else [dist[num][0], i]
#         if counts[num] > ma:
#             ma, check = counts[num], [num]
#         elif counts[num] == ma:
#             check.append(num)
#     mi = 2**32
#     for num in check:
#         if dist[num][1] - dist[num][0] < mi:
#             mi = dist[num][1] - dist[num][0]
#     return mi + 1
    
# print(findShortestSubArray([1,1]))


#   ====================  ====================

# def isStrobogrammatic(num: str) -> bool:
#     strobVals = {'0':'0', '1':'1', '8':'8', '6':'9', '9':'6'}
#     for i in range((len(num) // 2) + 1):
#         if num[i] not in strobVals or strobVals[num[i]] != num[-1-i]:
#             return False
#     return True

# print(isStrobogrammatic(num = "61619"))


#   ====================  ====================

# def nextPermutation(nums) -> None:

#     if len(nums) < 2:
#         return nums
#     for i in range(len(nums) - 2, -1, -1):
#         found = 0
#         for j in range(i + 1, len(nums)):
#             if nums[j] > nums[i]:
#                 found = j
#         if found:
#             break
#     if not found:
#         nums.sort()
#     else:
#         nums[i], nums[found] = nums[found], nums[i]
#         temp = sorted(nums[i+1:])
#         nums[i+1:] = temp
#     print(nums)

# print(nextPermutation([2,3,1]))

#     # Works but too slow and also violates constraints
# def nextPermutation(nums) -> None:
    
#     dummy = nums[:]
#     nums.sort()
#     perms = []

#     def permute(nums, perm = []):

#         if not nums:
#             perms.append(perm)
#             if perm > dummy:
#                 print('d', dummy, 'p', perm)
#                 return perm
#         for i in range(len(nums)):
#             # perm += [nums[i]]
#             ans = permute(nums[:i] + nums[i+1:], perm + [nums[i]])
#             if ans:
#                 return ans

#     print(permute(nums))

#     return perms

# print(nextPermutation([3,2,1]))


#   ====================  ====================

# from collections import Counter
# import heapq

# def topKFrequent(words, k: int):
    # words = Counter(words)
    # ans = []
    # for word in sorted(words.items(), key = lambda x: (-x[1], x[0])):
    #     ans.append(word[0])
    #     k -= 1
    #     if not k:
    #         break
    # return ans

#     words = Counter(words)
#     heap = []
#     ans = []
#     for word in words:
#         heapq.heappush(heap, [-words[word], word])
#     while k:
#         ans.append(heapq.heappop(heap)[1])
#         k -= 1
#     return ans

# print(topKFrequent(["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"], k = 4))


#   ====================  ====================

# import heapq

# def kClosest(points, K: int):
#     distances = []
#     ans = []
#     for x, y in points:
#         heapq.heappush(distances, [x**2 + y**2,[x, y]])
#     while K:
#         temp = heapq.heappop(distances)
#         ans.append(temp[1])
#         K -= 1
#     return ans

# def kClosest(points, K: int):
#     distances = sorted(points, key = lambda n: n[0]*n[0] + n[1]*n[1])
#     ans = []
#     for i in range(K):
#         ans.append(distances[i])
#     return ans

# print(kClosest(points = [[1,3],[-2,2]], K = 1))


#   ====================  ====================

# from collections import Counter

#     # Faster with Array
# def numPairsDivisibleBy60(time) -> int:
    # ans = 0
    # comps = [0]*60
    # for song in time:
    #     mod = song % 60
    #     adj = (60 - mod) if mod else 0
    #     ans += comps[mod]
    #     comps[adj] += 1 
    # return ans

#     # With hash table
# def numPairsDivisibleBy60(time) -> int:
    # ans = 0
    # comps = Counter()
    # for song in time:
    #     mod = song % 60
    #     adj = 60 - mod if mod else 0
    #     if mod in comps:
    #         ans += comps[mod]
    #     comps[adj] += 1
    # return ans

# print(numPairsDivisibleBy60([418,204,77,278,239,457,284,263,372,279,476,416,360,18]))


#   ====================  ====================

# def minMeetingRooms(intervals) -> int:
    
#     if not intervals:
#         return 0
    
#     intervals.sort()
#     rooms_reserved = [intervals[0]]
    
#     for i in range(1, len(intervals)):
#         end, start = rooms_reserved[-1][1], intervals[i][0]
#         if end > start:
#             for j in range(len(rooms_reserved) - 1):
#                 end = rooms_reserved[j][1]
#                 if end <= start:
#                     rooms_reserved[j] = intervals[i]
#                     break
#             else:
#                 rooms_reserved.append(intervals[i])
#         else:
#             rooms_reserved[-1] = intervals[i]
    
#     print(rooms_reserved)
#     return len(rooms_reserved)

# print(minMeetingRooms([]))


#   ====================  ====================

# def canAttendMeetings(intervals) -> bool:
#     intervals.sort()
#     for i in range(len(intervals) - 1):
#         end, start = intervals[i][1], intervals[i+1][0]
#         if end > start:
#             return False
#     return True

#     # Did it without sort but it's actually a lot slower and doesn't pass =( )
# def canAttendMeetings(intervals) -> bool:
#     se = set()
#     for i in range(len(intervals)):
#         s, e = intervals[i]
#         temp = set(range(s, e))
#         ln = len(se) + len(temp)
#         se = se.union(temp)
#         if ln != len(se):
#             return False
#     return True

# print(canAttendMeetings([[7,10],[9,11]]))


#   ================== REVIEW ====================
#   ==================== 560 ====================
#   ==================== REVIEW ==================

# def subarraySum(nums, k: int) -> int:
#     ans = 0
#     runningSum = 0
#     all_sums = {}
#     for i in range(len(nums)):
#         runningSum += nums[i]
#         diff = runningSum - k
#         if runningSum == k:
#             ans += 1
#         if diff in all_sums:
#             ans += all_sums[diff]
#         all_sums[runningSum] = 1 if runningSum not in all_sums else all_sums[runningSum] + 1
#     return ans

# def subarraySum(nums, k: int) -> int:
#     ans = 0
#     for i in range(len(nums)):
#         runningSum = nums[i]
#         if runningSum == k:
#             ans += 1
#         for j in range(i + 1, len(nums)):
#             runningSum += nums[j]
#             if runningSum == k:
#                 ans += 1
#     return ans

# print(subarraySum([0,0,0,0,0,0,0,0,0,0], 0))


#   ===================  ===================

# def minRemoveToMakeValid(s: str) -> str:
#     s = list(s)
#     parenSt, idxSt = [], []
#     for i, c in enumerate(s):
#         if c == '(':
#             parenSt.append('(')
#         elif c == ')' and not parenSt:
#             idxSt.append(i)
#         elif c == ')':
#             parenSt.pop()
#     for num in idxSt:
#         s[num] = ''
#     if parenSt:
#         for i in range(len(s) - 1, -1, -1):
#             if s[i] == '(':
#                 s[i] = ''
#                 parenSt.pop()
#                 if not parenSt:
#                     break
#     return ''.join(s)

# print(minRemoveToMakeValid(s = "lee()t(c)o)de)"))


#   ===================  ===================

# def repeatedSubstringPattern(s: str) -> bool:
#     for i in range((len(s) // 2) - 1, -1, -1):
#         if s[:i+1] * (len(s) // (i + 1)) == s:
#             return True
#     return False

# print(repeatedSubstringPattern("babbabbabbabbab"))


#   ===================  ===================

    #     # Okay
# def judgeCircle(moves: str) -> bool:
    
    # dic = {'R': [0,1], 'L': [0,-1], "U": [1,0], 'D': [-1,0]}
    # su = [0,0]
    # for c in moves:
    #     su[0] += dic[c][0]
    #     su[1] += dic[c][1]
    # return su == [0,0]

    #     # Better
# def judgeCircle(moves: str) -> bool:
    # dic = {x: 0 for x in 'UDLR'}
    # for c in moves:
    #     dic[c] += 1
    # return dic['R'] == dic['L'] and dic['U'] == dic['D']

# print(judgeCircle("UD"))


#   ===================  ===================

# def merge(intervals):
#     if len(intervals) < 2:
#         return intervals
#     intervals = sorted(intervals)
#     # print(intervals)
#     ans = []
#     low, i = 0, 0
#     while i < len(intervals) - 1:
#         if intervals[low][1] >= intervals[i+1][0]:
#             intervals[low][1] = max(intervals[low][1], intervals[i+1][1])
#             i += 1
#         else:
#             ans.append([intervals[low][0], max(intervals[low][1], intervals[i][1])])
#             low = i = i + 1
#         if i == len(intervals) - 1:
#             ans.append([intervals[low][0], max(intervals[low][1], intervals[i][1])])
#     return ans

# print(merge([[0,1],[1,1],[1,2]]))


#   =================== 114 ===================

#     # ITERATIVE
# def longestPalindrome(s: str) -> str:
#     def testString(s: str) -> str:
#         i = j = len(s) // 2
#         if len(s) % 2 == 0:
#             i -= 1
        
#         while True:
#             try:
#                 if s[i] == s[j]:
#                     i, j = i-1, j+1
#                 else:
#                     break
#             except:
#                 break
        
#         return s[i+1:j]

#     ans = ''
#     for i in range(len(s)):
#         s1 = testString(s[i:])
#         s2 = testString(s[:-i])
#         ma = s1 if len(s1) > len(s2) else s2
#         if len(ma) > len(ans):
#             ans = ma

#     return ans

#     # RECURSION + MEMOIZATION
# def longestPalindrome(s: str) -> str:

#     dp = {}

#     def testString(s: str) -> str:
#         if not s:
#             return ''
#         if s in dp:
#             return dp[s]

#         i = j = len(s) // 2
#         if len(s) % 2 == 0:
#             i -= 1
        
#         while True:
#             try:
#                 if s[i] == s[j]:
#                     i, j = i-1, j+1
#                 else:
#                     break
#             except:
#                 break

#         ans = s[i+1:j]
#         s1 = testString(s[1:])
#         s2 = testString(s[:-1])
#         temp = s1 if len(s1) > len(s2) else s2

#         if len(temp) > len(ans):
#             dp[s] = temp
#             return temp
#         dp[s] = ans
#         return ans

#     return testString(s)

# print(longestPalindrome("babaddtattarrattatddetartrateedredividerb"))


#   =================== 114 ===================

# def flatten(self, root: TreeNode) -> None:
#         if not root:
#             return root
        
#         stack = [root]
        
#         while stack:
#             node = stack.pop()
#             if node.right:
#                 stack.append(node.right)
#             if node.left:
#                 stack.append(node.left)
#                 node.right = node.left
#                 node.left = None
#             elif stack:
#                 node.right = stack.pop()
#                 stack.append(node.right)
        
#         return root


#   =========== REVIEW STACK SOLUTION ===========
#   =================== 1475 ===================
#   =========== REVIEW STACK SOLUTION ===========

# def finalPrices(prices):
#     ans = []
#     for i in range(len(prices) - 1):
#         for j in range(i + 1, len(prices)):
#             if prices[j] <= prices[i]:
#                 ans.append(prices[i] - prices[j])
#                 break
#         else:
#             ans.append(prices[i])
#     return ans + [prices[-1]]

# def finalPrices(prices):
#     stack = []
#     for i in range(len(prices)):
#         while stack and prices[stack[-1]] >= prices[i]:
#             prices[stack.pop()] -= prices[i]
#         stack.append(i)
#     return prices

# print(finalPrices(prices = [8,7,4,2,8,1,7,7,10,1]))


#   ====================  ====================

# from collections import Counter

# def sortString(s: str) -> str:
#     s = sorted(list(s))
#     counts, tracker = Counter(s), 0
#     ans = []
    
#     def iterate(key):
#         nonlocal tracker
#         if counts[key]:
#             ans.append(key)
#             counts[key] -= 1
#             if not counts[key]:
#                 tracker += 1

#     def forward():
#         for key in counts:
#             iterate(key)

#     def backward():
#         for key in reversed(counts.keys()):
#             iterate(key)

#     while len(counts) != tracker:
#         forward()
#         backward()

#     return ''.join(ans)

# print(sortString(s = "aaaabbbbcccc"))


#   ====================  ====================

# def flipAndInvertImage(A):
#     ans = []
#     for lis in A:
#         temp = []
#         for num in reversed(lis):
#             temp.append(0 if num == 1 else 1)
#         ans.append(temp)
#     return ans

# print(flipAndInvertImage([
#     [1,1,0],
#     [1,0,1],
#     [0,0,0]
# ]))


#   ====================  ====================

# def tictactoe(moves) -> str:
#     grid = [[0 for i in range(3)] for i in range(3)]
#     for i, move in enumerate(moves):
#         x, y = move
#         grid[x][y] = 'A' if i % 2 == 0 else 'B'
#     print(grid)
#     for i in range(3):
#         if grid[0][i] and grid[0][i] == grid[1][i] == grid[2][i]:
#             return grid[0][i]
#         if grid[i][0] and grid[i][0] == grid[i][1] == grid[i][2]:
#             return grid[i][0]
#         if i == 0 and grid[1][1] and grid[0][0] == grid[1][1] == grid[2][2]:
#             return grid [1][1]
#         if i == 2 and grid[1][1] and grid[2][0] == grid[1][1] == grid[0][2]:
#             return grid[1][1]
#     return 'Draw' if len(moves) == 9 else 'Pending'

# print(tictactoe(moves = [[0,0],[1,1]]))


#   ====================  ====================

# def countLetters(S: str) -> int:
#     runningSum = 0
#     dp = {}
#     for i in range(len(S) + 1):
#         runningSum += i
#         dp[i] = runningSum
        
#     cur, count = S[0], 1
#     ans = 0
#     for i in range(1, len(S)):
#         if S[i] == cur:
#             count += 1
#         else:
#             ans += dp[count]
#             cur, count = S[i], 1
#     return ans + dp[count]

# print(countLetters(S = "aaaaaaaaaa"))


#   ====================  ====================

# def arraysIntersection(arr1, arr2, arr3):
#     ans = set(arr1)
#     ans = ans.intersection(arr2)
#     ans = ans.intersection(arr3)
#     return sorted(list(ans))

# def arraysIntersection(arr1, arr2, arr3):
#     ans = arr1
#     for lis in [arr2, arr3]:
#         temp = []
#         for num in lis:
#             if num in ans:
#                 temp.append(num)
#         ans = temp
#     return ans

# print(arraysIntersection([6,16,23,37,45,54,58,60,66,87,95,102,135,136,145,146,159,161,170,171,175,178,200,208,209,211,215,217,218,227,229,238,239,276,289,295,298,313,318,324,331,333,340,344,355,357,372,373,374,376,379,390,394,395,399,413,418,419,425,431,432,436,449,458,481,484,487,489,494,501,511,515,518,524,526,528,529,534,542,544,547,552,559,564,565,571,581,589,590,595,607,618,620,641,652,663,664,669,672,680,686,694,702,713,715,729,735,746,755,769,773,774,778,780,791,793,802,804,808,810,812,816,822,827,831,841,842,850,851,861,865,877,883,891,904,907,910,912,913,915,917,934,945,958,960,971,974,976,997,999,1008,1010,1011,1015,1027,1037,1040,1045,1055,1056,1070,1090,1099,1114,1118,1122,1125,1132,1133,1141,1143,1146,1153,1159,1165,1168,1170,1172,1173,1179,1181,1184,1207,1214,1218,1219,1239,1247,1255,1267,1273,1282,1285,1295,1300,1304,1312,1326,1346,1358,1360,1362,1367,1375,1396,1397,1402,1410,1412,1416,1418,1420,1424,1425,1435,1443,1447,1464,1470,1479,1491,1502,1507,1509,1515,1520,1531,1537,1539,1556,1562,1563,1565,1577,1582,1583,1587,1589,1619,1642,1645,1648,1652,1662,1665,1677,1678,1695,1707,1711,1713,1725,1727,1731,1736,1744,1747,1751,1757,1771,1776,1783,1784,1787,1797,1802,1809,1812,1823,1827,1828,1829,1833,1836,1847,1854,1860,1867,1873,1874,1880,1887,1888,1897,1911,1913,1919,1923,1931,1948,1951,1954,1964,1965,1967,1969,1971,1973,1982,1988],
# [21,33,38,50,53,57,64,78,81,82,89,96,97,117,123,131,140,147,149,152,160,161,173,178,185,186,200,233,234,236,245,250,256,288,294,314,318,323,327,330,337,338,347,350,352,369,384,385,386,391,395,396,397,407,410,425,435,449,458,461,469,472,476,488,489,490,506,512,522,533,537,545,560,561,562,564,572,588,596,601,603,617,651,653,656,659,661,673,678,684,685,698,699,701,712,716,719,725,726,727,729,732,743,744,747,753,759,771,772,773,780,783,786,799,827,830,834,836,837,840,843,847,850,853,860,866,870,879,883,888,893,912,914,924,929,931,938,946,948,951,959,972,980,981,985,993,996,1010,1011,1014,1015,1022,1025,1029,1044,1048,1050,1053,1057,1066,1067,1070,1080,1083,1093,1095,1100,1102,1137,1151,1152,1155,1159,1170,1191,1192,1195,1211,1214,1222,1228,1229,1232,1247,1249,1256,1275,1276,1279,1280,1281,1292,1293,1306,1324,1326,1332,1348,1362,1363,1368,1386,1397,1401,1407,1408,1411,1417,1419,1421,1424,1430,1433,1443,1445,1457,1467,1471,1472,1484,1486,1488,1498,1504,1505,1521,1526,1540,1549,1550,1555,1558,1559,1563,1565,1578,1582,1584,1600,1601,1603,1612,1623,1626,1635,1640,1644,1652,1653,1654,1655,1658,1661,1669,1670,1703,1714,1726,1734,1739,1747,1749,1759,1760,1770,1796,1815,1821,1826,1838,1840,1841,1850,1853,1855,1857,1858,1859,1878,1882,1886,1888,1892,1896,1897,1899,1909,1911,1918,1920,1922,1937,1943,1953,1962,1963,1964,1980,1993,1995],
# [4,7,8,9,12,21,25,29,32,37,39,48,55,63,65,71,72,81,82,83,96,97,104,109,114,116,118,120,122,124,127,131,136,154,161,165,166,177,182,184,187,200,203,213,223,226,230,240,278,283,286,309,313,315,337,338,349,354,357,362,363,364,366,369,377,380,381,384,393,399,409,410,416,422,435,441,444,452,459,460,462,463,464,467,470,471,485,491,511,515,536,553,557,571,573,576,577,594,598,599,601,618,619,635,642,647,652,661,671,674,680,697,705,712,713,730,733,735,746,754,759,767,768,777,781,787,801,804,808,814,819,831,835,847,859,860,861,872,888,890,892,899,902,907,916,926,928,929,932,937,942,949,960,968,970,980,986,993,996,1005,1006,1007,1009,1014,1017,1026,1028,1031,1036,1041,1043,1047,1048,1054,1062,1066,1069,1072,1075,1079,1089,1090,1091,1094,1105,1111,1112,1113,1122,1139,1142,1143,1148,1157,1159,1160,1162,1163,1187,1190,1202,1219,1235,1244,1247,1249,1250,1261,1265,1279,1296,1297,1308,1309,1313,1315,1320,1323,1340,1344,1358,1370,1372,1375,1380,1415,1418,1419,1422,1432,1438,1450,1464,1466,1471,1473,1476,1479,1490,1503,1508,1511,1521,1535,1538,1541,1562,1571,1572,1576,1583,1602,1618,1620,1626,1628,1630,1647,1650,1662,1664,1665,1667,1669,1679,1686,1687,1705,1707,1742,1745,1750,1757,1784,1793,1813,1825,1826,1827,1846,1854,1863,1871,1872,1878,1886,1888,1898,1909,1913,1915,1916,1919,1932,1935,1939,1948,1970,1984,1996]))


#   ====================  ====================

# def restoreString(s: str, indices) -> str:
#     ans = [0] * len(s)
#     for i, index in enumerate(indices):
#         ans[index] = s[i]
#     return ''.join(ans)

# print(restoreString(s = "codeleet", indices = [4,5,6,7,0,2,1,3]))


#   ==================== 581 ====================

# def canPlaceFlowers(flowerbed, n: int) -> bool:
#     for i in range(len(flowerbed)):
#         if not n:
#             return True
#         if not flowerbed[i]:
#             if i == 0 or (i - 1 > 0 and not flowerbed[i - 1]):
#                 if i + 1 == len(flowerbed) or (i + 1 < len(flowerbed) and not flowerbed[i + 1]):
#                     flowerbed[i] = 1
#                     n -= 1
#     return n == 0

# print(canPlaceFlowers(flowerbed = [0,1,0,0,0], n = 2))


#   ================== REVIEW ====================
#   ==================== 581 ====================
#   ==================== REVIEW ==================

# def findUnsortedSubarray(nums) -> int:
#     i, j = 0, len(nums)
#     so = sorted(nums)
#     for i in range(len(nums)):
#         if nums[i] != so[i]:
#             break
#     else:
#         return 0
#     for j in range(len(nums)-1,-1,-1):
#         if nums[j] != so[j]:
#             break
#     return (j - i) + 1

            # NONE OF THESE WORKED
        # def findUnsortedSubarray(nums) -> int:
            # i, j = 0, len(nums) - 1

            # while i < j:
            #     if nums[i] > nums[j]:
            #         return (j - i) + 1
            #     if nums[i] > nums[j-1]:
            #         return (j - i)
            #     if nums[i+1] > nums[j]:
            #         return (j - i)
            #     if nums[i] > nums[i+1]:
            #         while nums[j] > nums[j-1]:
            #             j -= 1
            #         return (j - i) + 1
            #     if nums[j] < nums[j-1]:
            #         while nums[i] < nums[i+1]:
            #             i += 1
            #         return (j - i) + 1
            #     i, j = i+1, j-1
            # return 0


            # while i < len(nums)-1:
            #     if nums[i] > nums[j]:
            #         return (j - i) + 1
            #     if nums[i] > nums[i+1]:
            #         while i - 1 > 0 and nums[i-1] == nums[i]:
            #             i -= 1
            #         break
            #     i += 1
            # else:
            #     return 0
            # while j > 0:
            #     if nums[i] > nums[j]:
            #         return (j - i) + 1
            #     if nums[j] < nums[j-1]:
            #         while j + 1 < len(nums) and nums[j+1] == nums[j]:
            #             j += 1
            #         break
            #     j -= 1
            # return (j - i) + 1

            # for i in range(len(nums)):
            #     for j in range(len(nums)-1,i,-1):
            #         if nums[i] > nums[j]:
            #             return (j - i) + 1
            # return 0

# print(findUnsortedSubarray([2, 6, 4, 8, 10, 9, 15]))


#   ====================  ====================

# def pivotIndex(nums) -> int:
#     su = sum(nums)
#     runningSum = 0
#     for i in range(len(nums)):
#         su -= nums[i]
#         runningSum += nums[i-1] if i > 0 else 0
#         if su == runningSum:
#             return i
#     return -1

# print(pivotIndex(nums = [1,7,3,6,5,6]))


#   ====================  ====================

# def twoCitySchedCost(costs) -> int:
#     costs.sort(key = lambda x: abs(x[0] - x[1]), reverse=True)
#     ans = 0
#     a = b = len(costs) // 2
#     for i in range(len(costs)):
#         if a and b:
#             mi = min(costs[i])
#             ans, idx = ans + mi, costs[i].index(mi)
#             if idx == 0:
#                 a -= 1
#             else:
#                 b -= 1
#         elif a:
#             ans += costs[i][0]
#         else:
#             ans += costs[i][1]
#     return ans

# print(twoCitySchedCost([[10,20],[30,200],[400,50],[30,20]]))


#   ====================  ====================

#     # Better than below
# def nextGreaterElement(nums1, nums2):
#     dic = {nums2[i]: nums2[i+1:] for i in range(len(nums2))}
#     ans = []
#     for num in nums1:
#         for x in dic[num]:
#             if x > num:
#                 ans.append(x)
#                 break
#         else:
#             ans.append(-1)
#     return ans

#     # Slower than above
# def nextGreaterElement(nums1, nums2):
#     ans = []
#     for num in nums1:
#         for x in range(nums2.index(num)+1,len(nums2)):
#             if nums2[x] > num:
#                 ans.append(nums2[x])
#                 break
#         else:
#             ans.append(-1)
#     return ans

# print(nextGreaterElement(nums1 = [4,1,2], nums2 = [1,3,4,2]))


#   ====================  ====================

# def gcdOfStrings(str1: str, str2: str) -> str:
#     def checker(s1, s2):
#         for i in range(len(s1)-1,-1,-1):
#             slc = s1[:i+1]
#             if slc*(len(s2) // (i+1)) == s2 and slc*(len(s1) // (i+1)) == s1:
#                 return slc
#         return ""

#     if str1 in str2:
#         return checker(str1, str2)
#     elif str2 in str1:
#         return checker(str2, str1)
#     else:
#         return ""

# print(gcdOfStrings("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
# "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"))


#   ====================  ====================

#     # My first solutionm, which is actually the better one
# def maximumProduct(nums) -> int:
#     nums.sort()
#     j = len(nums) - 1
#     ans = 0
    
#     if nums[len(nums) - 1] <= 0:
#         return nums[-1]*nums[-2]*nums[-3]
#     elif nums[0]*nums[1] > nums[j-1]*nums[j-2]:
#         ans = nums[0]*nums[1]
#     else:
#         ans = nums[j-1]*nums[j-2]

#     return ans * nums[j]

#     # My second attempt at 0(n) -- but ends up being slower? Probably because of the numerous conditionals
# def maximumProduct(nums) -> int:
#     negA = negB = 0
#     a = b = c = -(2 ** 32)
#     for i in range(len(nums)):
#         temp = nums[i]
#         if temp < 0:
#             if abs(temp) > negA:
#                 negA, negB = abs(temp), negA
#             elif abs(temp) > negB:
#                 negB = abs(temp)
#         if temp > a:
#             a, b, c = temp, a, b
#         elif temp > b:
#             b, c = temp, b
#         elif temp > c:
#             c = temp
#     return max(negA*negB, b*c)*a

# print(maximumProduct([7,3,1,0,0,6]))


#   ====================  ====================

#     # First attempt with merge sort(this was bad - heap is better -- see next solution)
# class KthLargest:
#     def __init__(self, k: int, nums):
#         def merge(left, right):
#             s = []
#             i = j = 0
#             while i < len(left) and j < len(right):
#                 if left[i] < right[j]:
#                     s.append(left[i])
#                     i += 1
#                 else:
#                     s.append(right[j])
#                     j += 1
#             s += left[i:] if i < len(left) else right[j:]
#             return s
#         def mergeSort(nums):
#             if len(nums) <= 1:
#                 return nums
            
#             ln = len(nums) // 2
            
#             left = mergeSort(nums[:ln])
#             right = mergeSort(nums[ln:])
            
#             return merge(left, right)
#         self.nums = mergeSort(nums)
#         self.k = k
        
#     def add(self, val: int) -> int:
#         if len(self.nums) < self.k:
#             for i in range(len(self.nums)):
#                 if val <= self.nums[i]:
#                     self.nums.insert(i, val)
#                     break
#             else:
#                 self.nums.append(val)
#         elif val > self.nums[-self.k]:
#             for i in range(len(self.nums) - (self.k), len(self.nums)):
#                 if val <= self.nums[i]:
#                     self.nums.insert(i, val)
#                     break
#             else:
#                 self.nums.append(val)
#         return self.nums[-self.k]

#     # Much better
# import heapq

# class KthLargest:
#     def __init__(self, k: int, nums):
#         self.k = k
#         heapq.heapify(nums)
#         self.nums = nums
    
#     def add(self, val: int) -> int:
#         heapq.heappush(self.nums, val)
#         while len(self.nums) > self.k:
#             heapq.heappop(self.nums)
#         return self.nums[0]


#   ====================  ====================

# def anagramMappings(A, B):
#     bMap = {num: i for i, num in enumerate(B)}
#     return [bMap[num] for num in A]

# print(anagramMappings(A = [12, 28, 46, 32, 50],
#                         B = [50, 12, 32, 46, 28]))


#   ====================  ====================

# def calculateTime(keyboard: str, word: str) -> int:
#     dic = {c: i for i, c in enumerate(keyboard)}
#     curIdx = ans = 0
#     for c in word:
#         ans += abs(curIdx - dic[c])
#         curIdx = dic[c]
#     return ans

# print(calculateTime(keyboard = "abcdefghijklmnopqrstuvwxyz", word = "cba"))


#   ====================  ====================

# def xorOperation(n: int, start: int) -> int:
#     nums = [start + 2 * i for i in range(n)]
#     xor = nums[0]
#     for i in range(1, len(nums)):
#         xor ^= nums[i]
#     return xor

# print(xorOperation(n = 5, start = 0))


#   ====================  ====================

# def removeVowels(S: str) -> str:
#     # ans = [c for c in S if c not in 'aeiou']
#     # return ''.join(ans)
#     return ''.join([c for c in S if c not in 'aeiou'])

# print(removeVowels("leetcodeisacommunityforcoders"))


#   ====================  ====================

# def numIdenticalPairs(nums) -> int:
#     ans = [[i, j] for i in range(len(nums) - 1) for j in range(i+1, len(nums)) if nums[i] == nums[j]]
#     return len(ans)

# print(numIdenticalPairs(nums = [1,2,3]))


#   ====================  ====================

# def runningSum(nums):
#     sum = 0
#     for i in range(len(nums)):
#         nums[i] = nums[i] + sum
#         sum = nums[i]
#     return nums
# print(runningSum(nums = [1,2,3,4]))


#   ====================  ====================

# def minCost(costs) -> int:
#     path = {}
#     # print(costs)
#     for k in range(2):
#         i = 0; prev = len(costs)
#         path[k] = 0
#         while i < len(costs):
#             m = min(costs[i][:prev] + costs[i][prev+1:])
#             path[k] += m
#             prev = costs[i].index(m)
#             # print(m, path[k], prev)
#             costs[i][prev] = 2**16
#             i += 1
#         print(costs, path)
#     return min(path[0], path[1])

# print(minCost([[3,5,3],[6,17,6],[7,13,18],[9,10,18]]))


#   ====================  ====================

# def backspaceCompare(S: str, T: str) -> bool:

#     def modStr(orig, mod, i, count):
#         while i >= 0:
#             if orig[i] == '#':
#                 count, i = count+1, i-1
#             elif count == 0:
#                 mod.append(orig[i])
#                 i -= 1
#             else:
#                 count, i = count-1, i-1
#         return mod

#     modS = modStr(S, [], len(S)-1, 0)
#     modT = modStr(T, [], len(T)-1, 0)

#     return ''.join(modS) == ''.join(modT)

# print(backspaceCompare(S = "a#c", T = "b"))


#   ====================  ====================
# from collections import deque

# def sortedSquares(A):
#     ans = deque()
#     i, j = 0, len(A)-1
#     while i <= j:
#         if abs(A[i]) > abs(A[j]):
#             ans.appendleft(A[i]**2)
#             i += 1
#         else:
#             ans.appendleft(A[j]**2)
#             j -= 1
#     return ans

# # def sortedSquares(A):
# #     squares = [num*num for num in A]
# #     squares.sort()
# #     return squares

# print(sortedSquares([2, -1, 1]))


#   ====================  ====================

# def customSortString(S: str, T: str) -> str:
#     ans = []
#     for c in S:
#         if c in T:
#             ans = ans + [c]*T.count(c)
#     for c in T:
#         if c not in S:
#             ans += [c]
#     return ''.join(ans)

# print(customSortString(S = "cba", T = "abcd"))


#   ====================  ====================

# def floodFill(image, sr: int, sc: int, newColor: int):
#     st = [[sr, sc]]
#     orig = image[sr][sc]
#     if orig == newColor:
#         return image
#     image[sr][sc] = newColor
#     while st:
#         i, j = st.pop()
#         directions = [[i, j+1], [i, j-1], [i-1, j], [i+1, j]]
#         for i, j in directions:
#             if 0 <= i < len(image) and 0 <= j < len(image[0]):
#                 if image[i][j] == orig:
#                     image[i][j] = newColor
#                     st += [[i, j]]
#     return image

# print(floodFill(image = [[1,1,1],[1,1,0],[1,0,1]], sr = 1, sc = 1, newColor = 2))


#   ====================  ====================
# from collections import defaultdict

#     # This did NOT pass when I tried to create my own dictionary as opposed to using defaultdict
# # Manhattan(p1, p2) = |p1.x - p2.x| + |p1.y - p2.y|
# def assignBikes(workers, bikes):
#     ans = [-1]*len(workers)
#     choices = defaultdict(list)
#     for i, worker in enumerate(workers):
#         for j, bike in enumerate(bikes):
#             dist = (abs(worker[0] - bike[0]) + abs(worker[1] - bike[1]))
#             choices[dist].append([i, j])
#     for dist in sorted(choices):
#         for choice in choices[dist]:
#             w, b = choice
#             if ans[w] == -1 and bikes[b] != 0:
#                 ans[w], bikes[b] = b, 0
#     return ans

# print(assignBikes(workers = [[0,0],[1,1],[2,0]], bikes = [[1,0],[2,2],[2,1]]))


#   ================== REVIEW ====================
#   ==================== 406 ====================
#   ==================== REVIEW ==================

# import heapq as hp

#     # My solution which took me forever and runs super slow (5%) even though it's the same time complexity
#      # as the solution below this one
# def reconstructQueue(people):
#     hp.heapify(people)
#     rq = [0]*len(people)
#     while people:
#         i = count = 0
#         p = hp.heappop(people)
#         while i < len(rq):
#             if rq[i] == 0 or rq[i][0] >= p[0]:
#                 count += 1
#             if count > p[1] and rq[i] == 0:
#                 rq[i] = p
#                 break
#             i += 1
#     return rq

#     # After looking at solution (O(N^2))
# def reconstructQueue(people):
#     people.sort(key = lambda x: (-x[0], x[1]))
#     ans = []
#     for p in people:
#         ans.insert(p[1], p)
#     return ans

# print(reconstructQueue([[7,0], [4,4], [7,1], [5,0], [6,1], [5,2]]))


#   ====================  ====================

# def shortestDistance(words, word1: str, word2: str) -> int:
#     i1 = i2 = -1
#     mi = 2**8
#     for i, word in enumerate(words):
#         if word == word1:
#             i1 = i
#         elif word == word2:
#             i2 = i
#         if i1 > -1 and i2 > -1:
#             mi = min(mi, abs(i1-i2))
#     return mi

# print(shortestDistance(["a","c","b","a"], "a", "b"))


#   ====================  ====================

# def findPairs(nums, k: int) -> int:
#     nums.sort()
#     se, ans = set(), 0
#     i, j = 0, 1
#     while i < len(nums) and j < len(nums):
#         diff = abs(nums[i]-nums[j])
#         if nums[i] not in se and diff == k and i != j:
#             ans += 1
#             se.add(nums[i])
#         if i == j or diff < k:
#             j += 1
#         elif diff > k:
#             i += 1
#     return ans

# def findPairs(nums, k: int) -> int:
#     seen, ans = set(), set()
#     for i in range(len(nums)):
#         if nums[i] + k in seen:
#             ans.add(tuple(sorted((nums[i], nums[i] + k))))
#         if nums[i] - k in seen:
#             ans.add(tuple(sorted((nums[i], nums[i] - k))))
#         seen.add(nums[i])
#         print(ans)
#     return len(ans)

# print(findPairs([6,3,5,7,2,3,3,8,2,4], 2))


#   ==================== 986 ====================

#     # Refactored after looking at discussion -- about the same time complexity but much cleaner
# def intervalIntersection(A, B):
#     i, j, ans = 0, 0, []
#     while i < len(A) and j < len(B):
#         l1, l2 = A[i], B[j]
#         if l1[0] <= l2[1] and l2[0] <= l1[1]:
#             ans.append([max(l1[0], l2[0]), max(l1[1], l2[1])])
#         if i < len(A)-1 and l1[1] < l2[1] or j == len(B)-1:
#             i += 1
#         else:
#             j += 1
#     return ans

#     # My first solution
# def intervalIntersection(A, B):
#     i, j, ans = 0, 0, []
#     while i < len(A) and j < len(B):
#         ci, k, l1, l2 = [], 0, A[i], B[j]
#         aRange, bRange = range(l1[0], l1[1]+1), range(l2[0], l2[1]+1)
#         while k < 2:
#             if l1[k] in bRange:
#                 ci.append(l1[k])
#             elif l2[k] in aRange:
#                 ci.append(l2[k])
#             else:
#                 break
#             k += 1
#         else:
#             ans.append(ci)
#         if i < len(A)-1 and l1[1] < l2[1] or j == len(B)-1:
#             i += 1
#         else:
#             j += 1
#     return ans

# print(intervalIntersection([[0,5],[12,14],[15,18]], [[11,15],[18,19]]))


#   ==================== 1086 ====================
# from collections import deque

# def highFive(items):
#     items.sort()
#     di = {}
#     for item in items:
#         id, score = item
#         di[id] = deque([score]) if id not in di else di[id] + deque([score])
#         if len(di[id]) > 5:
#             di[id].popleft()
#     return [[key, sum(val) // 5] for key, val in di.items()]

# print(highFive([[1,91],[1,92],[2,93],[2,97],[1,60],[2,77],[1,65],[1,87],[1,100],[2,100],[2,76]]))


#   ==================== 366 ====================

#     I did this with BFS but it also works with a stack and DFS

# from collections import deque

# def __init__(self):
#     self.queue = deque()
#     self.ans = []

# def findLeaves(self, root: TreeNode) -> List[List[int]]:
    
#     def removeLeaves(root, path):
#         self.queue.append([root, None])
#         while self.queue:
#             node, parent = self.queue.popleft()
#             if not node:
#                 continue
#             if node.left == node.right:
#                 path.append(node.val)
#                 if node == parent.left:
#                     parent.left = None
#                 else:
#                     parent.right = None
#             else:
#                 self.queue.append([node.left, node]); self.queue.append([node.right, node])
#         self.ans.append(path)
    
#     if root:
#         while root.left != root.right:
#             removeLeaves(root, [])
#         return self.ans + [[root.val]]
#     return root


#   ====================  ====================

# def findAndReplacePattern(words, pattern: str):
#     ans = []
#     for word in words:
#         mp, seen = {}, set()
#         for i, c in enumerate(word):
#             if c not in mp:
#                 if pattern[i] in seen:
#                     break
#                 mp[c] = pattern[i]
#             if c in mp and mp[c] != pattern[i]:
#                 break
#             seen.add(pattern[i])
#         else:
#             ans.append(word)
#     return ans

# print(findAndReplacePattern(words = ["abc","deq","mee","aqq","dkd","ccc"], pattern = "abb"))


#   ==================== 543 ====================

# def minAddToMakeValid(S: str) -> int:
#     count, add = 0, 0
#     for p in S:
#         count = count -1 if p == ')' else count + 1
#         if count < 0:
#             add, count = add + 1, count + 1
#     if count > 0:
#         add += count
#     return add

# print(minAddToMakeValid("()))(("))


#   ================== REVIEW ====================
#   ==================== 543 ====================
#   ==================== REVIEW ==================

# def __init__(self):
#     self.ans = 0

# def diameterOfBinaryTree(root) -> int:
    
#     def findDepth(node):
#         if not node:
#             return 0
#         left, right = findDepth(node.left), findDepth(node.right)
#         self.ans = max(self.ans, left+right)
#         return max(left, right) + 1
    
#     findDepth(root)
#     return self.ans


# #   ==================== 811 ====================

# def subdomainVisits(cpdomains):
#     counts = {}
#     for domain in cpdomains:
#         domain = domain.split()
#         count, urls = int(domain[0]), [domain[1]]
#         while urls:
#             sub = urls.pop()
#             counts[sub] = count if sub not in counts else counts[sub] + count
#             nextSub = sub.find('.')
#             if nextSub > -1:
#                 urls.append(sub[nextSub+1:])
#     return [f'{val} {key}' for key, val in counts.items()]

# print(subdomainVisits(["900 google.mail.com", "50 yahoo.com", "1 intel.mail.com", "5 wiki.org"]))


# #   ==================== 953 ====================

# def isAlienSorted(words, order) -> bool:
#     alienDic = {c: i for i, c in enumerate(order, 1)}
#     for i in range(len(words)-1):
#         for j in range(len(words[i])):
#             try:
#                 if alienDic[words[i][j]] > alienDic[words[i+1][j]]:
#                     return False
#                 elif alienDic[words[i][j]] < alienDic[words[i+1][j]]:
#                     break
#             except:
#                 return False
#     return True

# print(isAlienSorted(words = ["apple","app"], order = "abcdefghijklmnopqrstuvwxyz"))


# #   ====================  ====================

# def reorderLogFiles(logs):
#     letterLogs, digitLogs = [], []
#     for log in logs:
#         deli = log.find(' ')
#         if log[deli+1].isalpha():
#             letterLogs.append(log)
#         else:
#             digitLogs.append(log)
#     letterLogs.sort()
#     return sorted(letterLogs, key = lambda x: x[x.find(' ')+1:]) + digitLogs

# print(reorderLogFiles(logs = ["a1 9 2 3 1","g1 act car","zo4 4 7","ab1 off key dog","a8 act zoo","a2 act car"]))


# #   ====================  ====================

#     # Passes (~25%)
# def threeSum(nums):
#     iSet, ans = set(), set()
#     for i in range(len(nums)-2):
#         if not nums[i] in iSet:
#             comp = {}
#             for j in range(i+1, len(nums)-1):
#                 key = 0 - nums[i] - nums[j]
#                 comp[key] = [i, j]
#                 if nums[j+1] in comp:
#                     x, y = comp.pop(nums[j+1])
#                     ans.add(tuple(sorted([nums[x], nums[y], nums[j+1]])))
#         iSet.add(nums[i])
#     return ans

#     # Similar to above but with sort instead
# def threeSum(nums):
#     nums.sort()
#     ans = set()
#     if not nums or nums[0] > 0 or nums[len(nums)-1] < 0:
#         return ans
#     for i in range(len(nums)-2):
#         if i == 0 or i > 0 and not nums[i] == nums[i-1]:
#             comp = {}
#             for j in range(i+1, len(nums)-1):
#                 key = 0 - nums[i] - nums[j]
#                 comp[key] = [i, j]
#                 if nums[j+1] in comp:
#                     x, y = comp.pop(nums[j+1])
#                     ans.add((nums[x], nums[y], nums[j+1]))
#     return ans

#     # Too slow
# def threeSum(nums):
#     nums.sort()
#     ans = set()

#     def findTriplets(nums, se):
#         print(nums)
#         if len(nums) < 3 or nums[0] > 0 or nums[len(nums)-1] < 0:
#             return
#         if not nums[0] in se or not nums[len(nums)-1] in se:
#             temp = 0 - nums[0] - nums[len(nums)-1]
#             if temp <= 0:
#                 for k in range(1, len(nums)-1):
#                     if nums[k] > 0:
#                         break
#                     if nums[k] == temp:
#                         ans.add((nums[0], nums[k], nums[len(nums)-1]))
#             elif temp > 0:
#                 for k in range(len(nums)-2, 0, -1):
#                     if nums[k] < 0:
#                         break
#                     if nums[k] == temp:
#                         ans.add((nums[0], nums[k], nums[len(nums)-1]))
#         findTriplets(nums[1:], se | {nums[0]})
#         findTriplets(nums[:-1], se | {nums[len(nums)-1]})

#     findTriplets(nums, set())
#     return ans

#     # Too slow
# def threeSum(nums):
#     ans = set()
#     for i in range(len(nums)-2):
#         for j in range(i+1, len(nums)-1):
#             for k in range(j+1, len(nums)):
#                 if nums[i]+nums[j]+nums[k] == 0:
#                     ans.add(tuple(sorted([nums[i],nums[j],nums[k]])))
#     return [list(x) for x in ans]

# print(threeSum([-1, 0, 1, 2, -1, -4]))


# #   ==================== 994 ====================

# from collections import deque

#     # My solution after refactoring
# def orangesRotting(grid) -> int:
#     row, col, freshOranges = len(grid), len(grid[0]), 0
#     queue, ans = deque(), 0
#     for i in range(row):
#         for j in range(col):
#             if grid[i][j] == 2:
#                 queue.append([i, j, 0])
#             elif grid[i][j] == 1:
#                 freshOranges += 1

#     def turnToRot(n, m, cnt):
#         nonlocal freshOranges
#         if 0 <= n < row and 0 <= m < col:
#             if grid[n][m] == 1:
#                 grid[n][m], cnt = 2, cnt+1
#                 freshOranges -= 1
#                 queue.append([n, m, cnt])

#     while queue:
#         x, y, ans = queue.popleft()
#         if grid[x][y] == 2:
#             turnToRot(x-1, y, ans); turnToRot(x+1, y, ans)
#             turnToRot(x, y-1, ans); turnToRot(x, y+1, ans)
#     if freshOranges > 0:
#         return -1
#     return ans

#     # My first solution
# def orangesRotting(self, grid: List[List[int]]) -> int:
#     queue = deque()
#     row, col, freshOranges, ans = len(grid), len(grid[0]), 0, 0
#     for i in range(row):
#         for j in range(col):
#             if grid[i][j] == 2:
#                 queue.append([i, j, 0])
#             elif grid[i][j] == 1:
#                 freshOranges += 1
#     while queue:
#         x, y, mn = queue.popleft()
#         ans = max(ans, mn)
#         if grid[x][y] == 2:
#             if x > 0:
#                 temp = mn
#                 if grid[x-1][y] == 1:
#                     grid[x-1][y], temp = 2, temp+1
#                     freshOranges -= 1
#                     queue.append([x-1, y, temp])
#             if x < (row - 1):
#                 temp = mn
#                 if grid[x+1][y] == 1:
#                     grid[x+1][y], temp = 2, temp+1
#                     freshOranges -= 1
#                     queue.append([x+1, y, temp])
#             if y > 0:
#                 temp = mn
#                 if grid[x][y-1] == 1:
#                     grid[x][y-1], temp = 2, temp+1
#                     freshOranges -= 1
#                     queue.append([x, y-1, temp])
#             if y < (col - 1):
#                 temp = mn
#                 if grid[x][y+1] == 1:
#                     grid[x][y+1], temp = 2, temp+1
#                     freshOranges -= 1
#                     queue.append([x, y+1, temp])
#     if freshOranges > 0:
#         return -1
#     return ans

# print(orangesRotting([
#     [0],[1]
# ]))


# #   ==================== 3 ====================

# def lengthOfLongestSubstring(s: str) -> int:
#     seen, se, lss, i = {}, set(), 0, 0
#     for j, c in enumerate(s):
#         if c in se:
#             se -= set(s[i:seen[c] + 1])
#             i = seen[c] + 1
#         seen[c], lss = j, max(j+1 - i, lss)
#         se.add(c)
#     return lss

# print(lengthOfLongestSubstring("abba"))


# #   ==================== 1306 ====================

# def canReach(arr, start: int,) -> bool:

#     def splitUp(idx, count):
#         if count >= len(arr) or idx >= len(arr) or idx < 0:
#             return False
#         if arr[idx] == 0:
#             return True
#         return splitUp(idx + arr[idx], count + 1) or splitUp(idx - arr[idx], count + 1)
#     return splitUp(start, 1)

# print(canReach(arr = [4,2,3,0,3,1,2], start = 0))


# #   ==================== 841 ====================

# def canVisitAllRooms(rooms) -> bool:
#     keys = set(rooms[0])
#     visited = {0}
#     while keys:
#         visit = keys.pop()
#         visited.add(visit)
#         for key in rooms[visit]:
#             if key not in visited:
#                 keys.add(key)
#     return len(visited) == len(rooms)

# print(canVisitAllRooms([[1,3],[3,0,1],[2],[0]]))


# #   ==================== 1161 ====================

# def maxLevelSum(self, root: TreeNode) -> int:
    
#     levelSums = {}
    
#     def addEmUp(node, level):
#         if not node:
#             return
        
#         levelSums[level] = node.val if level not in levelSums else levelSums[level] + node.val
    
#         addEmUp(node.left, level + 1)
#         addEmUp(node.right, level + 1)
        
#     addEmUp(root, 1)
#     maximal = max(levelSums.values())
#     for key in levelSums:
#         if levelSums[key] == maximal:
#             return key


# #   ====================  ====================

# def gardenNoAdj(N: int, paths):
#     gardens = [0] + [[i, (i % 4) if (i % 4) > 0 else 4] for i in range(1, N + 1)]
#     adj, ans = {}, [1] * N
#     for x, y in paths:
#         if x not in adj:
#             adj[x] = []
#         if y not in adj:
#             adj[y] = []
#         adj[x].append(y)
#         adj[y].append(x)
#     for g in adj:
#         flowers = {1,2,3,4}
#         for connection in adj[g]:
#             flowers -= {gardens[connection][1]}
#         newFlower = flowers.pop()
#         ans[g-1], gardens[g] = newFlower, [g, newFlower]
#     return ans

# print(gardenNoAdj(N = 5, paths = [[2,3],[3,4],[4,1],[1,3],[2,4],[5,1]]))


# #   ====================  ====================

# def filterRestaurants(restaurants, veganFriendly: int, maxPrice: int, maxDistance: int):
#     filtered = []
#     for r in restaurants:
#         if r[3] <= maxPrice:
#             if r[4] <= maxDistance:
#                 if veganFriendly == 0:
#                     filtered.append(r)
#                 elif r[2] == veganFriendly:
#                     filtered.append(r)
#     sortedRes = sorted(filtered, key = lambda x: (x[1], x[0]), reverse = True)
#     return [r[0] for r in sortedRes]

# print(filterRestaurants(restaurants = [[33433,15456,1,99741,58916],[61899,85406,1,27520,12303],[63945,3716,1,56724,79619]],
#     veganFriendly = 0, maxPrice = 91205, maxDistance = 58378))


# #   ==================== 1471 ====================

#     # My first solution
# def getStrongest(arr, k: int):
#     if len(arr) == 1:
#         return arr
    
#     arr.sort()
#     median = arr[(len(arr)-1) // 2]
#     strongestList = sorted(arr, key = lambda x: abs(x-median))
    
#     return strongestList[len(strongestList)-k:]

#     # Without sorted() -- though I don't see a good reason not to use sorted
# def getStrongest(arr, k: int):
#     if len(arr) == 1:
#         return arr
    
#     arr.sort()
#     median = arr[(len(arr)-1) // 2]
#     i, j = 0, len(arr)-1

#     while i + (len(arr) - 1 - j) < k:
#         if abs(arr[i] - median) > abs(arr[j] - median):
#             i += 1
#         else:
#             j -= 1
    
#     return arr[:i] + arr[j + 1:]

# print(getStrongest(arr = [1,2,3,4,5], k = 2))

# #   ==================== 969 ====================

# def pancakeSort(A):
#     idx, flips = len(A), []
#     while idx > 1:
#         mi = A.index(max(A[:idx]))
#         # print(mi, idx, A)
#         if mi == 0:
#             A[:idx] = [num for num in reversed(A[:idx])]
#             mi = idx-1
#             flips.append(idx)
#         elif mi < idx-1:
#             rev = A[:mi+1]
#             A[:mi+1] = [num for num in reversed(rev)]
#             flips.append(mi+1)
#         if mi == idx-1:
#             idx -= 1
#     # print(A)
#     return flips

# print(pancakeSort([1,2,3]))


# #   ==================== 215 ====================
# import random

# def findKthLargest(self, nums: List[int], k: int) -> int:

#     def partition(lis, left, piv):
#         low, right = left, len(lis)-1
#         lis[right], lis[piv] = lis[piv], lis[right]
#         while left < right:
#             if lis[left] < lis[piv]:
#                 lis[left], lis[low] = lis[low], lis[left]
#                 low += 1
#             left += 1
#         lis[low], lis[piv] = lis[piv], lis[low]
#         return low

#     def quickSelect(lis, k):
        
#         piv = random.randint(0, len(lis)-1)
#         part_idx = partition(lis, 0, piv)
        
#         if part_idx == k:
#             return lis[part_idx]
#         elif part_idx > k:
#             return quickSelect(lis[:part_idx], k)
#         elif part_idx < k:
#             return quickSelect(lis[part_idx+1:], k-part_idx-1)
                    
#     return quickSelect(nums, len(nums)-k)
    
#     def mergeSort(lis):
#         if len(lis) == 1:
#             return lis
        
#         left = lis[:len(lis) // 2]
#         right = lis[len(lis) // 2:]
        
#         def merge(left, right):
#             li, ri, m = 0, 0, []
#             while li < len(left) and ri < len(right):
#                 if left[li] < right[ri]:
#                     m.append(left[li])
#                     li += 1
#                 else:
#                     m.append(right[ri])
#                     ri += 1
#             m += right[ri:] if li == len(left) else left[li:]
#             return m
        
#         return merge(mergeSort(left), mergeSort(right))

#     new_lis = mergeSort(nums)
#     return new_lis[-k]

# def findKthLargest(nums, k):
#     for i in range(len(nums), len(nums)-k, -1):
#         tmp = 0
#         for j in range(i):
#             if nums[j] > nums[tmp]:
#                 tmp = j
#         nums[tmp], nums[i-1] = nums[i-1], nums[tmp]
#     return nums[len(nums)-k]

# print(findKthLargest([1,2,3,4,5], k = 2))


# #   ==================== 1347 ====================

    # My first attempt.. decent runtime
# def minSteps(s: str, t: str) -> int:
#     s_dic, count = {}, 0
#     for c in s:
#         s_dic[c] = 1 if c not in s_dic else s_dic[c] + 1
#     for c in t:
#         if c in s_dic:
#             s_dic[c] -= 1
#             if s_dic[c] == 0:
#                 del s_dic[c]
#         else:
#             count += 1
#     return count

#     # Really cool solution I saw in discussion -- replaces 1 instance of each leter in s with 1 instance of the \
#      # same letter in t (if it doesn't exist in t it does nothing). Whatever is left in t is how many letters \
#       # need to be replaced
# def minSteps(s: str, t: str) -> int:
#     for c in s:
#         t = t.replace(c, '', 1)

# print(minSteps(s = "anagram", t = "mangaar"))


# #   ==================== 328 ====================

# def oddEvenList(self, head: ListNode) -> ListNode:
    
#     if not head or not head.next:
#         return head
    
#     r1, r2, temp = head, head.next, head.next
    
#     while r1.next and r2.next:
#         r1.next = r1.next.next
#         r2.next = r2.next.next
#         r1, r2 = r1.next, r2.next
    
#     r1.next = temp
    
#     return head


# #   ====================  ====================

# def partitionLabels(S):
#     cur, rt, checked, ans = 0, 0, set(), []
#     for i, c in enumerate(S):
#         if c not in checked:
#             rep = S.rfind(c)
#             if rep != i and rep > rt:
#                 rt = rep
#         if i == rt:
#             ans += [S[cur:rt+1]]
#             cur = rt = rt+1
#         checked.add(c)

#     return [len(a) for a in ans]

# print(partitionLabels(S = "ababcbacadefegdehijhklij"))


# #   ====================  ====================

# def numTilePossibilities(tiles: str) -> int:
#     perms = set()

#     def findPerms(lis, perm):
#         if perm not in perms:
#             perms.add(perm)
#         for i in range(len(lis)):
#             findPerms(lis[:i] + lis[i+1:], perm + lis[i])
        
#     for i in range(len(tiles)):
#         perm = tiles[i]
#         lis = tiles[:i] + tiles[i+1:]
#         findPerms(lis, perm)

#     return len(perms)

# print(numTilePossibilities("AAB"))


#   ====================  ====================

# def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
#     vals = []
    
#     def findVals(node):
#         if not node:
#             return
        
#         vals.append(node.val)
        
#         findVals(node.left); findVals(node.right)
    
#     findVals(root1); findVals(root2)
    
#     return sorted(vals)

# def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
    
#     vals = []
    
#     def findVals(node):
#         if not node:
#             return
        
#         vals.append(node.val)
        
#         findVals(node.left); findVals(node.right)
    
#     findVals(root1); findVals(root2)
    
#     return sorted(vals)


#   ==================== 1329 ====================

# def diagonalSort(mat):
#     lis, ans, p = [], [], len(mat[0])
#     for row in mat:
#         lis += row
#     for i in range(-len(mat[0]), len(mat[0])):
#         if i < 0:
#             step = (p+1)*-1
#             cut = lis[i:(i - (p+i)*abs(step))-1:step]
#             cut = sorted(cut, reverse = True)
#             lis[i:(i - (p+i)*abs(step))-1:step] = cut
#         else:
#             step = p+1
#             cut = lis[i:(i + (p-i)*step):step]
#             cut = sorted(cut)
#             lis[i:(i + (p-i)*step):step] = cut
#     for i in range(len(mat)):
#         ans += [lis[i*p:(i*p)+p]]
#     return ans

    # After looking at discussion -- using dictionary
# def diagonalSort(mat):
#     r, c, dic = len(mat), len(mat[0]), {}
#     for i in range(r):
#         for j in range(c):
#             dic[i-j] = [mat[i][j]] if i-j not in dic else dic[i-j] + [mat[i][j]]
#     for key, val in dic.items():
#         dic[key] = sorted(val, reverse= True)
#     for i in range(r):
#         for j in range(c):
#             mat[i][j] = dic[i-j].pop()


# print(diagonalSort([
#     [11,25,66,1,69,7],
#     [23,55,17,45,15,52],
#     [75,31,36,44,58,8],
#     [22,27,33,25,68,4],
#     [84,28,14,11,5,50]
#     ]))


#   ================== REVIEW ====================
#   ==================== 1008 ====================
#   ==================== REVIEW ==================

#     # My solution (terrible time complexity)
# def bstFromPreorder(self, preorder: List[int]) -> TreeNode:
    
#     self.root = TreeNode(preorder[0])
    
#     def add_node(val):
#         cur = self.root
#         while True:
#             if val < cur.val:
#                 if not cur.left:
#                     cur.left = TreeNode(val)
#                     return
#                 cur = cur.left
#             elif val > cur.val:
#                 if not cur.right:
#                     cur.right = TreeNode(val)
#                     return
#                 cur = cur.right
    
#     for i in range(1, len(preorder)):
#         add_node(preorder[i])
    
#     return self.root

#     # After looking at discussion -- with stack
# def bstFromPreorder(self, preorder: List[int]) -> TreeNode:
#     self.root = TreeNode(preorder[0])
#     s = [self.root]
    
#     for i in range(1, len(preorder)):
#         if preorder[i] < s[-1].val:
#             s[-1].left = TreeNode(preorder[i])
#             s.append(s[-1].left)
#         else:
#             while s and preorder[i] > s[-1].val:
#                 last = s.pop()
#             last.right = TreeNode(preorder[i])
#             s.append(last.right)
    
#     return self.root

#     # # After looking at discussion -- with recursion
# def bstFromPreorder(self, preorder: List[int]) -> TreeNode:
#     if not preorder:
#         return None
    
#     root = TreeNode(preorder[0])
    
#     i = 1
#     while i < len(preorder) and preorder[i] < root.val:
#         i += 1
    
#     root.left = self.bstFromPreorder(preorder[1:i])
#     root.right = self.bstFromPreorder(preorder[i:])
    
#     return root


#   ================== REVIEW ====================
#   ==================== 48 ======================
#   =================== REVIEW ===================

#     # Learned after looking at discussion that the bitwise NOT operator is a really \
#       # good way to get the inverse of an index (ie. ~0 == -1, ~1 == -2 etc..)
# def rotate(matrix) -> None:

#     for i in range(len(matrix) // 2):
#         for j in range(i, len(matrix[0])-i-1):

#             temp = matrix[i][j]
#             matrix[i][j] = matrix[~j][i]
#             matrix[~j][i] = matrix[~i][~j]
#             matrix[~i][~j] = matrix[j][~i]
#             matrix[j][~i] = temp

#     return matrix

# print(rotate([
#     [1,2,3,4],
#     [5,6,7,8],
#     [9,10,11,12],
#     [13,14,15,16]
# ]))


#   ==================== 49 ======================

#     # I realized the lamba function that I provided in the sorted() method here is \
#       # unnessesary, but it made me feel smrt
# def groupAnagrams(strs):
#     dic = {}
#     for st in strs:
#         unsc = ''.join(sorted(st, key = lambda x: ord(x)))
#         dic[unsc] = [st] if unsc not in dic else dic[unsc] + [st]
#     return list(dic.values())

# print(groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))


#   ====================  ======================

    # ma = max(nums)
    # idx = nums.index(ma)
    # root = TreeNode(val = ma)
    # s = [(nums[:idx], root, 'l'), (nums[idx+1:], root, 'r')]
    
    # while s:
    #     sub, node, side = s.pop()
    #     if sub:
    #         ma = max(sub)
    #         idx = sub.index(ma)
    #         if side == 'l':
    #             node.left = TreeNode(val = ma)
    #             node = node.left
    #         else:
    #             node.right = TreeNode(val = ma)
    #             node = node.right
    #         s += [(sub[:idx], node, 'l'), (sub[idx+1:], node, 'r')]
    # return root


#   ====================  ======================

# import random

# a = ord('a')
# alph = [f'{i}' if i < 10 else chr(a+(i-10)) for i in range(36)]

# def encode(longUrl: str) -> str:
#     rand_str = ''.join(random.sample(alph, 6))
#     dic = {rand_str: longUrl}
#     shortUrl = f'http://tinyurl.com/{rand_str}'
#     return (shortUrl, dic)

# def decode(shortUrl: str, dic) -> str:
#     return dic[shortUrl[shortUrl.rfind("/")+1:]]

# enc = encode('askdjhsjbdfbhih')
# dec = decode(*enc)
# print(dec)


#   ====================  ======================

# def productExceptSelf(nums):

#     if len(nums) < 2:
#         return 0
#     skip = 1
#     prod = []
#     for num in nums:
#         prod.append(skip)
#         skip *= num
#     skip = 1
#     for i in range(len(nums)-1,-1,-1):
#         prod[i] *= skip
#         skip *= nums[i]
#     return prod

# print(productExceptSelf([1,2,3,4]))


#   ==================== 1395 ======================

# def numTeams(rating) -> int:
#     count = 0
#     for i in range(len(rating)-2):
#         for j in range(i+1, len(rating)):
#             if rating[j] > rating[i]:
#                 for k in range(j, len(rating)):
#                     if rating[k] > rating[j]:
#                         count += 1
#             if rating[j] < rating[i]:
#                 for k in range(j, len(rating)):
#                     if rating[k] < rating[j]:
#                         count += 1
#     return count

# print(numTeams(rating = [1,2,3,4]))


#   ==================== 1409 ======================

    # First attemp with recursion -- time limit exceeded
# def subsets(nums):
    
    # subs = [[num] for num in nums] + [[]]
    # if len(nums) < 2:
    #     return subs
    
    # subs += [nums]
    
    # def goDeeper(nums):
    #     if len(nums) == 1:
    #         return
    #     print("s",subs, "n",nums)
    #     subs.append(nums) if nums not in subs else None
    #     for i, num in enumerate(nums):
    #         goDeeper(nums[:i] + nums[i+1:])
    
    # for i, num in enumerate(nums):
    #     goDeeper(nums[:i] + nums[i+1:])

    # return subs

    # My second solution with stack (passed!)
# def subsets(nums):
#     if not nums:
#         return [nums]
#     s1, ans = [nums], []
#     while s1:
#         lis = s1.pop()
#         ans += [lis]
#         if len(lis) > 1:
#             for i in range(len(lis)):
#                 cut = lis[:i] + lis[i+1:]
#                 if cut not in ans:
#                     s1.append(cut)
#     return ans + [[]]

    # After looking at discussion.. genius
# def subsets(nums):
#     perms = [[]]
#     for num in nums:
#         perms += [perm+[num] for perm in perms]
#     return perms

# print(subsets(nums = [1,2,3,4]))


#   ==================== 1409 ======================

# from collections import deque

# def processQueries(queries, m):
#     deck, ans = deque(), []
#     for i in range(1, m+1):
#         deck.append(i)
#     for num in queries:
#         idx = deck.index(num)
#         ans += [idx]
#         deck.remove(num)
#         deck.appendleft(num)
#     return ans

# print(processQueries(queries = [3,1,2,1], m = 5))


#   ==================== 347 ======================

# def topKFrequent(nums, k: int):
#     if len(nums) < 3:
#         return set(nums)
#     se, reps = set(nums),[]
#     for i, num in enumerate(se):
#         reps.append([nums.count(num), num])
#     reps, ans = sorted(reps), []
#     for i in range(k):
#         ans += [reps[-1-i][1]]
#     return ans

# def topKFrequent(nums, k: int):
#     if len(nums) < 3:
#         return set(nums)
#     reps = {}
#     for num in nums:
#         reps[num] = 1 if num not in reps else reps[num] + 1
#     ans, i = [], 0
#     for key, val in reversed(sorted(reps.items(), key = lambda x: x[1])):
#         ans += [key]
#         i += 1
#         if i >= k:
#             break
#     return ans

# print(topKFrequent(nums = [1,1,1,2,2,3], k = 2))


#   ================ IM A GENIUS ==================
#   ================== REVIEW =====================
#   ==================== 1302 ======================
#   =================== REVIEW ===================

# import math

# def deepestLeavesSum(root) -> int:
#     if not root:
#         return 0
#     vals = {}
#     vals[0] = root.val
    
#     def goDiggin(node, lv):
#         if not node:
#             return
        
#         vals[lv] = node.val if lv not in vals else vals[lv] + node.val
        
#         goDiggin(node.left, lv+1)
#         goDiggin(node.right, lv+1)
    
#     goDiggin(root.left, 1)
#     goDiggin(root.right, 1)
    
#     gen = reversed(vals.values())
#     return next(gen)


#   ================ IM A GENIUS ==================
#   ================== REVIEW =====================
#   ==================== 22 ======================
#   =================== REVIEW ===================

# def generateParenthesis(n: int):
#     perms = []
#     def generate(path, count, ref):
#         if count == n:
#             if len(path) < n*2:
#                 path += ')'*((n*2)-len(path))
#             perms.append(path)
#             return
#         if ref > 0:
#             generate(path + ')', count, ref-1)
#         generate(path + '(', count+1, ref+1)

#     generate('(', 1, 1)
#     return perms

# print(generateParenthesis(4))


#   ==================== 1282 ======================

#     # My solution
# def groupThePeople(groupSizes):
#     dic, ans = {}, []
#     for i, p in enumerate(groupSizes):
#         dic[p] = [i] if p not in dic else dic[p] + [i]
#         if len(dic[p]) == p:
#             ans += [dic[p]]
#             del dic[p]
#     return ans

#     # From discussion -- I did not and probably could not reproduce this, but this is good to see for reference
# import collections
# def groupThePeople(groupSizes):
#     count = collections.defaultdict(list)
#     for i, size in enumerate(groupSizes):
#         count[size].append(i)
#     print(count)
#     return [l[i:i + s]for s, l in count.items() for i in range(0, len(l), s)]

# print(groupThePeople(groupSizes = [3,3,3,3,3,1,3]))


#   ================== REVIEW =====================
#   ==================== 46 ======================
#   =================== REVIEW ====================

# def permute(nums):
#     perms = []

#     def idfk(lis, n):
#         if not lis:
#             perms.append(n)
#         for i, num in enumerate(lis):
#             idfk(lis[:i] + lis[i+1:], n+[num])

#     for i, num in enumerate(nums):
#         idfk(nums[:i] + nums[i+1:], [num])

#     return perms

# print(permute([1,2,3,4]))


#   ==================== 1379 ======================

#     # Should work if values are repeated
# def getTargetCopy(self, original: TreeNode, cloned: TreeNode, target: TreeNode) -> TreeNode:
    
#     if target == original or original.left == original.right:
#         return cloned
    
#     def dive(o, c):
#         if not o:
#             return None
#         if o == target:
#             return c
        
#         return dive(o.left, c.left) or dive(o.right, c.right)
        
#     return dive(original.left, cloned.left) or dive(original.right, cloned.right)

#     # Would not work if values are repeated
#     def getTargetCopy(self, original: TreeNode, cloned: TreeNode, target: TreeNode) -> TreeNode:
        
#         if target == original or original.left == original.right:
#             return cloned
        
#         def dive(node):
#             if not node:
#                 return None
#             if node.val == target.val:
#                 return node
            
#             return dive(node.left) or dive(node.right)
            
#         return dive(cloned.left) or dive(cloned.right)


#   ==================== 1315 ======================

# def sumEvenGrandparent(self, root: TreeNode) -> int:
    
#     su = [0]
    
#     def ihopethisworks(gp, left, right):
    
#         if left == right:
#             return
        
#         if left:
#             if gp & 1 == 0:
#                 if left.left:
#                     su[0] += left.left.val
#                 if left.right:
#                     su[0] += left.right.val
#             ihopethisworks(left.val, left.left, left.right)
            
#         if right:
#             if gp & 1 == 0:
#                 if right.left:
#                     su[0] += right.left.val
#                 if right.right:
#                     su[0] += right.right.val
#             ihopethisworks(right.val, right.left, right.right)
    
#     ihopethisworks(root.val, root.left, root.right)
    
#     return su[0]

# def sumEvenGrandparent(self, root: TreeNode) -> int:
    
#     su = [0]
#     def ihopethisworks(root):
#         if not root or root.left == root.right:
#             return
#         if root.val & 1 == 0:
#             if root.left:
#                 if root.left.left:
#                     su[0] += root.left.left.val
#                 if root.left.right:
#                     su[0] += root.left.right.val
#             if root.right:
#                 if root.right.left:
#                     su[0] += root.right.left.val
#                     print(su)
#                 if root.right.right:
#                     su[0] += root.right.right.val
#                     print(su)
#         ihopethisworks(root.left)
#         ihopethisworks(root.right)

#     ihopethisworks(root)

#     return su[0]


#   ==================== 1304 ======================

    # First solution (my second (below) was much faster)
# def findContentChildren(g, s):
#     g.sort(), s.sort()
#     count = i = 0
#     for ch in g:
#         while i < len(s):
#             if ch <= s[i]:
#                 count, i = count+1, i+1
#                 break
#             i += 1
#     return count

    # Much faster
# def findContentChildren(g, s):
#     g.sort(), s.sort(reverse=True)
#     count = 0
#     for ch in g:
#         if not s:
#             return count
#         while ch > s[-1]:
#             s.pop()
#             if not s:
#                 return count
#         count += 1
#         s.pop()
#     return count

# print(findContentChildren([1,2], [1,2,3]))


#   ==================== 1304 ======================

#     # My first solution
# def sumZero(n: int):
#     if n & 1 == 1:
#         lis = [i for i in range(0 - (n // 2), (n // 2) + 1)]
#     else:
#         lis = [i for i in range(0 - (n // 2), 0)] + [i for i in range(1, (n // 2) + 1)]
#     return lis

#     # After looking at discussion -- much quicker
# def sumZero(n: int):
#     if n & 1 == 1:
#         return list(range(0 - (n // 2), (n // 2) + 1))
#     else:
#         return list(range(0-n+1, n, 2))

# print(sumZero(8))


#   ==================== 453 ======================

#     # My initial solution
# def minMoves(nums) -> int:
#     if len(nums) < 2:
#         return 0
#     nums.sort()
#     i, max, count = len(nums)-2, nums[-1], 0
#     while nums[0] < max:
#         count += (max - nums[0])
#         nums[0] += (max-nums[0]); nums[i] += count
#         if nums[i] > max:
#             max = nums[i]
#         i -= 1
#     return count

#     # After looking at discussion - a trick that makes it much easier
# def minMoves(nums) -> int:
#     mi, count = min(nums), 0
#     for num in nums:
#         count += (num-mi)
#     return count

# print(minMoves([1,2,3,4,5]))


#   ====================  ======================

# def replaceElements(arr):
#     if not arr:
#         return arr
#     idx = 0
#     m = max(arr[idx+1:])
#     for i in range(1, len(arr)):
#         if arr[i] == m:
#             arr[idx:i] = [arr[i]]*(i-idx)
#             idx = i
#             if i < len(arr)-1:
#                 m = max(arr[idx+1:])
#     arr[-1] = -1
#     return arr

# def replaceElements(arr):
#     if not arr:
#         return []
#     idx = 0
#     for i, num in enumerate(arr):
#         if idx < len(arr)-1:
#             m = max(arr[idx+1:])
#             if num == m:
#                 arr[idx:i] = [num]*(i-idx)
#                 idx = i
#     arr[-1] = -1
#     return arr

# print(replaceElements(arr = [17,18,5,4,6,1]))


#   ================== REVIEW =====================
#   ==================== 447 ======================
#   =================== REVIEW ====================

#     # My original solution (doesn't pass due to time limit exceeded -- see next solution)
# def numberOfBoomerangs(points) -> int:
#     diff = []
#     for i in range(len(points)):
#         temp = []
#         for j in range(len(points)):
#             if i != j:
#                 a = abs(points[i][0]-points[j][0])
#                 b = abs(points[i][1]-points[j][1])
#                 temp += [(a*a)+(b*b)]
#         diff += [temp]
#     count = 0
#     for l in diff:
#         for ln in set(l):
#             if l.count(ln) > 1:
#                 count += l.count(ln)*(l.count(ln)-1)
#     return count

#     # After looking at discussion
# def numberOfBoomerangs(points) -> int:
#     count = 0
#     for a in points:
#         lens = {}
#         for b in points:
#             if a != b:
#                 x = a[0]-b[0]
#                 y = a[1]-b[1]
#                 lens[x*x + y*y] = 1 if (x*x + y*y) not in lens else lens[x*x + y*y] + 1
#         for ln in lens.values():
#             count += (ln*(ln-1))
#     return count

# print(numberOfBoomerangs([[0,0],[1,0],[-1,0],[0,1],[0,-1]]))


#   ====================  ======================

# def uniqueMorseRepresentations(words) -> int:
#     morse = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
#     se = set()
#     for word in words:
#         st = ""
#         for c in word:
#             st += morse[ord(c)-ord('a')]
#         se.add(st)
#     return len(se)

# print(uniqueMorseRepresentations(words = ["gin", "zen", "gig", "msg"]))


#   ====================  ======================

# def shuffle(nums, n: int):
#     lis = []
#     for i, j in zip(nums[:n], nums[n:]):
#         lis += [i, j]
#     return lis

# print(shuffle(nums = [2,5,1,3,4,7], n = 3))


#   ==================== 443 ======================

    # My solution
# def compress(chars) -> int:
#     i, idx, count = 0, 0, 0
#     while i < len(chars):
#         count += 1
#         if count > 1:
#             if count == 2:
#                 idx = i
#             chars.pop(i)
#             i -= 1
#         if i == len(chars)-1 or chars[i] != chars[i+1]:
#             if count > 1:
#                 while count:
#                     chars.insert(idx, str(count % 10))
#                     count //= 10
#                     i += 1
#             count = 0
#         i += 1
#     return chars

    # From discussion -- study this
# def compress(chars):
#     left = i = 0
#     while i < len(chars):
#         char, length = chars[i], 1
#         while (i + 1) < len(chars) and char == chars[i + 1]:
#             length, i = length + 1, i + 1
#         chars[left] = char
#         print(chars[left], left, chars)
#         if length > 1:
#             len_str = str(length)
#             chars[left + 1:left + 1 + len(len_str)] = len_str
#             left += len(len_str)
#         left, i = left + 1, i + 1
#         print(chars, left)
#     return left

# print(compress(["a","b","b","b","b","b","b","b","b","b","b","b","b","c","c","c"]))


#   ==================== 441 ======================

# def arrangeCoins(n: int) -> int:
#     if n == 0:
#         return 0
#     i, count = 1, 0
#     while i <= n:
#         count += 1
#         i += i + 1
#     return count

# print(arrangeCoins(8))


#   ====================  ======================

# def freqAlphabets(s: str) -> str:
#     a = ord('a')
#     dic = {f'{i}' if i < 10 else f'{i}#': chr(a+i-1) for i in range(1, 27)}
#     i = 0
#     decrypt = ""
#     while i < len(s):
#         if i < len(s)-2 and s[i+2] == '#':
#             decrypt += dic[s[i]+s[i+1]+s[i+2]]
#             i += 3
#         else:
#             decrypt += dic[s[i]]
#             i += 1
#     return decrypt

# def freqAlphabets(s: str) -> str:
#     a = ord('a') - 1
#     for i in range(26, 0, -1):
#         s = s.replace(str(i)+'#' if i > 9 else str(i), chr(a+i))
#     return s

# print(freqAlphabets(s = "1326#"))


#   ==================== 437 ======================

#     # Adding on the way back
# def pathSum(self, root: TreeNode, s: int) -> int:
    
#     def newRoot(root):
    
#         if not root:
#             return 0

#         def find(node, tot, c):
#             if not node:
#                 return c
#             tot += node.val
#             if tot == s:
#                 c += 1
#             if node.left == node.right:
#                 return c

#             return find(node.left, tot, c) + find(node.right, tot, 0)
        
#         tott = newRoot(root.left) + newRoot(root.right)
        
#         return find(root, 0, 0) + tott
    
#     return newRoot(root)    
    
#         # Adding on the way down
# def pathSum(self, root: TreeNode, s: int) -> int:
    
#     c = [0]
    
#     def newRoot(node):
    
#         if not node:
#             return
        
#         newRoot(node.left); newRoot(node.right)

#         def find(node, tot):
#             if not node:
#                 return
#             tot += node.val
#             if tot == s:
#                 c[0] += 1
#             if node.left == node.right:
#                 return c

#             find(node.left, tot); find(node.right, tot)
        
#         find(node, 0)
        
#     newRoot(root)
    
#     return c[0]


#   ==================== 434 ======================

# def countSegments(s: str) -> int:
#     strp = s.strip()
#     if not strp:
#         return 0
#     count = 0
#     for i, c in enumerate(strp):
#         if c == ' ' and strp[i-1] != ' ':
#             count += 1
#     return count + 1

# print(countSegments(", , , ,        a, eaefa"))


#   ==================== 1351 ======================

# def countNegatives(grid) -> int:
    # count = 0
    # for i in range(len(grid)):
    #     for j in range(len(grid[i])-1, -1, -1):
    #         # count += 1 if grid[i][j] == 0 else break
    #         if grid[i][j] < 0:
    #             count += 1
    #         else:
    #             break
    # return count

# def countNegatives(grid) -> int:
#     i = len(grid)-1
#     j = 0
#     count = 0
#     while i > -1 and j < len(grid[0]):
#         if grid[i][j] < 0:
#             count += len(grid[i])-j
#             i -= 1
#         else:
#             j += 1
#     return count

# print(countNegatives(grid = [[4,3,2,-1],[3,2,1,-1],[1,1,-1,-2],[-1,-1,-2,-3]]))


#   ==================== 414 ======================

    # def thirdMax(self, nums: List[int]) -> int:
    #     m1 = m2 = m3 = -2**32
    #     se = set()
    #     for i in range(len(nums)):
    #         if nums[i] not in se:
    #             if nums[i] > m1:
    #                 m3, m2, m1 = m2, m1, nums[i]
    #             elif nums[i] > m2:
    #                 m3, m2 = m2, nums[i]
    #             elif nums[i] > m3:
    #                 m3 = nums[i]
    #         se.add(nums[i])
    #     if len(se) < 3:
    #         return max(se)
    #     return m3


#   ==================== 409 ======================

# def checkRecord(s: str) -> bool:
#     if "A" in s and "LLL" in s:
#         return False
#     return True

# print(checkRecord("PPALLL"))


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


#   ================== REVIEW ===================== x1
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