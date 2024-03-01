import random
t=[['*','*',"*"],['*','*',"*"],['*','*',"*"]]
for i in range(0,3):
    for j in range(0,3):
        print(t[i][j],end='\t')
    print('\n')
win = 0
comp=''
ans= 'y'
upos=''
cpos=''
turn =1
while ans == 'y':
    user = input("choose o or x\n")
    if user == 'o':
        comp = 'x'
        break
    elif user == 'x':
         comp = 'o'
         break
    else :
       ans = input(" Enter y to continue or n to exit\n")
       if ans == 'n' :
          exit(0)
while win == 0 :
    upos=input("enter pos for e.g. 00,01,02,10,11,12,20,21,22\n")
    upos=list(upos)
    upos=list(map(int,upos))
    t[upos[0]][upos[1]] = user
    if turn == 1 :
        if upos[0] == 1 and upos[1]== 1 :
            cpos = random.choice(['00','02','10','12','20','22'])
            cpos = list(map(int,cpos))
            t[cpos[0]][cpos[1]] = comp
        else :
            t[1][1] = comp
        for i in range(0, 3):
            for j in range(0, 3):
                print(t[i][j], end='\t')
            print('\n')
    elif turn == 2 :
        if t[upos[0]][upos[1]] == user :




