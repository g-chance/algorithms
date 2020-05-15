# given an unsorted array of integers, each integer is repeated twice except for one of them. find and return the integer that is not duplicated
# 1.try without constraints
# 2.try with memory constraint -> dont use additional structures (look up bitwise)

def findUnique(arr):
    result = arr[0]
    print(bin(result))
    for i in range(1,len(arr)):
        result^=arr[i]
        # print(bin(result))
        # print(result)
    return result

# print(findUnique([1,0,6,0,5,1,6]))
print("findUnique", findUnique([1,2,2,1,3,1,1,4,4,3,5,6,6,7,7]))
print(findUnique([1,0,0,2,1]))