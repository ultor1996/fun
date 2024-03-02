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
    if l[i]  > l[i+1] :
        k= l[i+1]
        l[i+1]=l[i]
        l[i]=k
print(l)
s=0
s=int(input("Enter the element to be found"))
f=0
l =0
r = int(size)-1
while l <= r:
    mid = l + (r-l)//2
    if l[mid] < s :
        l =mid + 1
    elif l[mid] > s :
        r = mid - 1
    else :
        print("The number is at ",mid)
        break

