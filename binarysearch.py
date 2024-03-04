import numpy as np
k=0
i=0
size=input("Enter the size of list\n")
size =int(size)
l=np.zeros(size)
print("Enter the elements")
while i < size :
    l[i] =int(input())
    i = i+1
for i in range (0,size-1):
    if l[i] > l[i+1] :
        k = l[i+1]
        l[i+1] = l[i]
        l[i] = k
s=0
s=int(input("Enter the element to be found\n"))
f=0
b=0
r = int(size)-1
mid = 0
while b <= r:
    mid = b + (r-b)//2
    if l[mid] < s :
        b =mid + 1
    elif l[mid] > s :
        r = mid - 1
    elif l[mid] == s :
        print("The number is at ",mid+1)
        break

