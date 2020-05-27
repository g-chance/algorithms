set_1 = set((1,2,3))
set_2 = set((3,4,5))
print(set_1 - set_2)
print(set_1)


#   ==================== Iterator Practice ====================

# my_gen = (x*x for x in range(5))
# print(my_gen)
# my_list = [x*x for x in range(5)]
# print("my_list",my_list)
# iter_ml = my_list.__iter__()
# print("iter_ml.__next__",iter_ml.__next__())
# print("iter_ml.__next__",iter_ml.__next__())
# print("*my_list",*my_list)
# butts, farts, *therest = my_list
# print("therest",therest)
# print(*my_gen)
# for num in my_gen:
#     print(num)
# print(my_gen)
# for num in my_gen:
#     print(num)
# print(*my_gen.__next__())


#   ==================== The Minion Game ====================

# def minion_game(s = "BAANANAS"):
#     kev_count = 0
#     stu_count = 0
#     for i in range(len(s)):
#         if s[i].lower() in "aeiou":
#             kev_count += len(s)-i
#         else:
#             stu_count += len(s)-i
#     if kev_count > stu_count:
#         print('Kevin', kev_count)
#     if kev_count < stu_count:
#         print('Stuart', stu_count)
#     else:
#         print('Draw')
# minion_game()

# def minion_game(s = "BAANANAS"):
#     stu_count = 0
#     kev_count = 0
#     my_set = set()
#     for i, char in enumerate(list(s)):
#         if char.lower() not in "aeiou":
#             for j in range(i+1,len(s)+1):
#                 if s[i:j] not in my_set:
#                     my_set.add(s[i:j])
#                     for k in range(i,len(s)):
#                         if s[i:j] == s[k:k+len(s[i:j])]:
#                             stu_count += 1
#     my_set = set()
#     for i, char in enumerate(list(s)):
#         if char.lower() in "aeiou":
#             for j in range(i+1,len(s)+1):
#                 if s[i:j] not in my_set:
#                     my_set.add(s[i:j])
#                     for k in range(i,len(s)):
#                         if s[i:j] == s[k:k+len(s[i:j])]:
#                             kev_count += 1
#     if kev_count > stu_count:
#         return f'Kevin {kev_count}'
#     return f"Stuart {stu_count}"
# print(minion_game())


#   ==================== sWAP cASE ====================

# s = 'HackerRank.com presents "Pythonist 2".'

# def swap_case(s):
#     s = s.swapcase()
#     return s

# def swap_case(s):
#     x = "".join([char.upper() if char.islower() else char.lower() for char in s])
#     print(x)
    # new_str = "".join[char.upper() for char in s if char.islower()]
    # return new_str

# def swap_case(s):
#     new_str = ""
#     print(s)
#     for char in s:
#         if char.isupper():
#             new_str += char.lower()
#         elif char.islower():
#             new_str += char.upper()
#         else:
#             new_str += char
#     return new_str

# result = swap_case(s)
# print(result)


#   ==================== NESTED LISTS ====================

    # Not the complete code -- completed code on hackerrank
# myList = []
# for i in range(5):
#     name = "name"
#     score = 0
#     myList.append([name,score])
# marksheet = list(set([marks for name, marks in myList]))
# print(myList)
# print(marksheet)


#   ==================== RUNNER-UP ====================

# n = 5
# arr = [2,3,6,6,5]
# max = arr[0]
# runner_up = -(2)**64
# for i in range(1,len(arr)):
#     if arr[i] > max:
#         max = arr[i]
# for j in range(len(arr)):
#     if arr[j] > runner_up and arr[j] < max:
#         runner_up = arr[j]
# print(runner_up)


#   ==================== LIST COMPREHENSIONS ====================

# x=1
# y=1
# z=1
# n=2
    # This is the same as the below example of list comprehension
# myList = []
# for i in range(x+1):
#     for j in range(y+1):
#         for k in range(z+1):
#             if i+j+k != n:
#                 myList.append([i,j,k])
# print(myList)

    # This is an example as list comprehension but does the same as the above triple loop
# myList = [[i,j,k] for i in range(x+1) for j in range(y+1) for k in range(z+1) if i+j+k !=n]
# print(myList)


#   ==================== PRINT FUNCTION ====================

# n = 3
# my_str = ""
# for i in range(1,n+1):
#     my_str += str(i)
# print(my_str)


#   ==================== WRITE A FUNCTION ====================

# y = 2400
# def isLeapYear(y):
#     if y % 4 == 0:
#         if y % 100 == 0:
#             if y % 400 == 0:
#                 return True
#             return False
#         else:
#             return True
#     return False
# print(isLeapYear(y))


#   ==================== LOOPS ====================

# n = 9
# for i in range(n):
#     print(i*i)


# Read two integers from STDIN and print three lines where:

# a=3
# b=2
# print(a+b)
# print(a-b)
# print(a*b)