a=[3,56,32,1,45,2,33]
b=0
s=len(a)
for i in range (0,s):
    for j in range (0,s):
        if a[i]>a[j] :
            b=a[i]
            a[i]=a[j]
            a[j]=b
print(a)