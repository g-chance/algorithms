def removeDuplicate(arr):
    i=0
    while i<len(arr)-1:
        if arr[i] == arr[i+1]:
            for j in range(i,len(arr)-1):
                arr[j],arr[j+1] = arr[j+1],arr[j]
            print(arr)
            arr.pop()
            print(arr)
        else:
            i+=1
    return arr

y = removeDuplicate([1,2,2,3,3,4,4,4,4])
print(y)