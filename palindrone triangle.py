r=0
n=9
x=' '
for i in range (1,n):
    print((n-i-1)*x,end="")
    for j in range (1,i):
           print(j,end="")
           if(j==i-1):                   
                r=j-1
                while(r>=1):
                    print(r,end="")
                    r=r-1      
    print("\n")

