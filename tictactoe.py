import random
import time
t=[['*','*',"*"],['*','*',"*"],['*','*',"*"]]
for i in range(0,3):
    for j in range(0,3):
        print(t[i][j],end='\t')
    print('\n')
true = 1
comp=''
ans= 'y'
upos=''
upos1=''
cpos=''
user=''
comp=''
start = random.choice(['user','comp'])
if start == 'user' :
    print("You go first\n")
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
    upos=input("enter pos for e.g. 00,01,02,10,11,12,20,21,22\n")
    upos=list(upos)
    upos=list(map(int,upos))
    t[upos[0]][upos[1]] = user
    if upos[0] == 1 and upos[1]== 1 :
        cpos = random.choice(['00','02','10','12','20','22'])
        cpos = list(map(int,cpos))
        t[cpos[0]][cpos[1]] = comp
        for i in range(0, 3):
            for j in range(0, 3):
                print(t[i][j], end='\t')
            print('\n')
        print('Its a draw')
        time.sleep(5)
        exit(0)
    else :
        t[1][1] = comp
        for i in range(0, 3):
            for j in range(0, 3):
                print(t[i][j], end='\t')
            print('\n')
        upos = input("enter pos for e.g. 00,01,02,10,12,20,21,22\n")
        upos = list(upos)
        upos = list(map(int, upos))
        t[upos[0]][upos[1]] = user
        if (t[0][0] == user and t[2][2] == user) :
            t[1][2]= comp
            for i in range(0, 3):
                for j in range(0, 3):
                    print(t[i][j], end='\t')
                print('\n')
            print("Its a draw\n")
            time.sleep(5)
            exit(0)
        elif (t[0][2] == user and t[2][0] == user) :
            t[0][1] = comp
            for i in range(0, 3):
                for j in range(0, 3):
                    print(t[i][j], end='\t')
                print('\n')
            print("Its a draw\n")
            time.sleep(5)
            exit(0)
        else :
            for i in range(0, 3):
                for j in range(0, 3):
                    print(t[i][j], end='\t')
                print('\n')
            print("Its a draw\n")
            time.sleep(5)
            exit(0)
elif start == 'comp' :
    print('Computer goes first')
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
    #cpos = random.choice(['00', '02', '20', '22','11'])
    #cpos = list(map(int, cpos))
    t[0][0] = comp
    for i in range(0, 3):
        for j in range(0, 3):
            print(t[i][j], end='\t')
        print('\n')
    while true :
        upos = input("enter pos for e.g. 00,01,02,10,11,12,20,21,22\n")
        upos = list(upos)
        upos = list(map(int, upos))
        t[upos[0]][upos[1]] = user
        if upos[0] == 1 and upos[1]== 1 :
            t[2][2] = comp
            for i in range(0, 3):
                for j in range(0, 3):
                    print(t[i][j], end='\t')
                print('\n')
            upos1 = input("enter pos for e.g. 00,01,02,10,11,12,20,21,22\n")
            upos1 = list(upos1)
            upos1 = list(map(int, upos1))
            t[upos1[0]][upos1[1]] = user
            if (t[1][1] == user and t[0][2] == user)  :
                t[2][0] = comp
                for i in range(0, 3):
                    for j in range(0, 3):
                        print(t[i][j], end='\t')
                    print('\n')
                print("Computer Won")
                time.sleep(5)
                exit(0)
            elif (t[1][1] == user and t[2][0] == user):
                t[0][2] = comp
                for i in range(0, 3):
                    for j in range(0, 3):
                        print(t[i][j], end='\t')
                    print('\n')
                print("Computer Won")
                time.sleep(5)
                exit(0)
            else :
                for i in range(0, 3):
                    for j in range(0, 3):
                        print(t[i][j], end='\t')
                    print('\n')
                print("Its a draw")
                time.sleep(5)
                exit(0)

        else :
            for i in range(0, 3):
                for j in range(0, 3):
                    print(t[i][j], end='\t')
                print('\n')
            print("Its a guaranteed loss for you\n")
            time.sleep(5)
            exit(0)



