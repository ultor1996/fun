import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
#reading in the words
words=open('names.txt','r').read().splitlines()
print(len(words))
print(max(len(w) for w in words))
print(words[:8])
# building vocabulary of characters and mapping to/from integers
chars=sorted(list(set(''.join(words))))
stoi={s:i+1 for i,s in enumerate(chars)}
stoi['.']=0
print(stoi)
itos={i:s for s,i in stoi.items()}
vocab_size=len(itos)
print(itos)
print(vocab_size)
#building the dataset
block_size=3 # context lenght: how many characters do we take to predict the next one
def build_dataset(words):
    X,Y=[],[]

    for w in words:
        context=[0]* block_size
        for ch in w + '.':
            ix=stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:]+[ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
   # print(X,Y)
    print(X.shape,Y.shape)
    return X,Y
random.seed(42)
random.shuffle(words)
n1=int(0.8*len(words))
n2=int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])
# basic prep done now to important stuff: