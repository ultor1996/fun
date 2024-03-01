import random
again = 'y'
while again == 'y':
    user = input("What's your choice : r for rocks, p for papers and s for scissors.\n")
    while (user != 'r') and (user != 's') and (user != 'p'):
        user = input("What's your choice : r for rocks, p for papers and s for scissors.\n")
    comp = random.choice(['r', 'p', 's'])
    if user == comp:
        print('Its a tie')
    elif (user == 'r' and comp == 's') or (user == 's' and comp == 'p') or (user == 'p' and comp == 'r'):
        print('You won!!')
    else:
        print('Better luck next time')
    again = input('Enter y to try again.\n')