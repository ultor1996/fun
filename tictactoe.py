import numpy as np
t=[['*','*',"*"],['*','*',"*"],['*','*',"*"]]
for i in range(0,3):
    for j in range(0,3):
        print(t[i][j],end='\t')
    print('\n')
user=input("choose o or x\n")
pos=input("enter pos for e.g. 01,02,03,10,11,12,20,21,22\n")
pos=list(pos)
pos=list(map(int,pos))
t[pos[0]][pos[1]]=user
