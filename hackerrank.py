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