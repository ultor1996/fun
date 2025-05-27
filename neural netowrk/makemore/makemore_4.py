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
def cmp(s,dt,t):
    ex= torch.all(dt == t.grad).item() # matches the gradients exactly to each decinmal points
    app = torch.allclose(dt, t.grad) # mathces within some tolerance
    maxdiff = (dt -t.grad).abs().max().item()
    print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')

#neural network architecture
n_embd = 10 # dimensionality if the chcrater embedding vectors
n_hidden= 64 # number of neurons in the hidden layers

g=torch.Generator().manual_seed(2147483647)
C= torch.randn((vocab_size,n_embd),            generator=g) # embedding character layers

#layer 1

W1=torch.randn((n_embd * block_size), n_hidden, generator=g) * (5/3)/((n_embd * block_size)**0.5) # first layer  the post factor of 1/(n_embd * block_size)**0.5) is needed to unsre that the varance of the porduct of the wx+b is = 1, evne though this will work fine without the gain factor for first few or fthe first layer but later
                                                                                                # in the deep layers this factor alone wont be enough to keep the output of the activateion layer uniformaly distributed in its range of -1,1 rather it will get concetrated around zero which hampers the abaility iof the neural network to learn. Now once the gain of 5/3 is added
                                                                                                # the output of each activation layer maintains the uniformaity through the ranmge of -1,1 and does concetrate around zero. As being near to zero affects the strenghts of the forard pass signals weak.
                                                                                                # the number 5/3 occurs when you navegare the tanh^2(x) with a pdf of nortmal distribuiotn over the whole range of -inf to inf and sqaure root it
b1 = torch.randn(n_hidden,                        generator=g) * 0.1 # using b1 just for fun, it's useless because of BN
# Layer 2
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.1 # the prefactor here makes sure that intial loss and probability doistribution of the network equal to the theoretically what is expected as each character of lphabet should have equal probability so it should be equal to  -log(1/27). So here just reducing the variance of the gaussian from which the values are picked makes sure that the initial probability falls in that range.
b2 = torch.randn(vocab_size,                      generator=g) * 0.1
# BatchNorm parameters
bngain = torch.randn((1, n_hidden))*0.1 + 1.0
bnbias = torch.randn((1, n_hidden))*0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True

#mini- batches

batch_size=32
n =  batch_size
ix = torch.randint(0,Xtr.shape[0],(batch_size,), generator= g) # creates a list of random 32 indices from the total number of indicies in Xtr
Xb, Yb =Xtr[ix], Ytr[ix]


#forward pass in a more discretized way
emb=C[Xb]# ebed the vectors into chracters
embcat=emb.view(emb.shape[0],-1)

# Linear Layer 1
hprebn = embcat @ W1 + b1 # hidden layer pre activation layer
# batch norm layer
bnmeani=1/n*hprebn.sum(0, keepdim=True)
bndiff=  hprebn - bnmeani
bndiff2=bndiff**2
bnvar=1/(n-1)*(bndiff2).sum(0,keepdim=True) # bessels correction to use n-1 as here we use clauclated sample mean not the true population mean
bnvar_inv=(bnvar+ 1e-5)**(-0.5)
bnraw= bndiff * bnvar_inv # this the trasnformation used to convert the  ouptu of the linear layer back into a normal distribution
hpreact = bngain*bnraw +bnbias # ading two adtional parameter set like the bingain  and bnbias to so that the mean and the variance of the tranformed normal distribution has a little possibiliyy of variation
#Non-lienarity
h = torch.tanh(hpreact) # hidden layer activation
#Linear layer 2
logits= h @ W2 + b2 # ouput layer
# cross entropy loss
logit_maxes= logits.max(1, keepdim=True).values
norm_logits= logits- logit_maxes #subtract max for numerical stability
counts = norm_logits.exp()
counts_sum = counts.sum(1,keepdim=True) # keepdim true is necesarry to prevent broadcastign errors. For eg here if krrpdim= True was not there we would have a 32, tensor which would give an erro when needed to be broadcasted in the division counts/counts/sum where counts is 32 X 27 sand counts_sum is 32, as for broadcasting the 32 of 32, will coinsdered the number of columsn and 1 will be added to number of rows which messes upo the dimensality of matrices for division
counts_sum_inv=counts_sum**-1
probs= counts* counts_sum_inv
logprobs=probs.log()
loss = -logprobs[range(n), Yb].mean()

#Pytorch backward pass
for p in parameters:
    p.grad=None
for t in [logprobs, probs, counts, counts_sum, counts_sum_inv, # afaik there is no cleaner way
          norm_logits, logit_maxes, logits, h, hpreact, bnraw,
         bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani,
         embcat, emb]:
  t.retain_grad()
loss.backward()
loss
# Excercise 1: backpropogate through the whole thing manually,
# backpropogataing throigh exactly all of the variables
# as tther are define in the forward pass above, one by one. PLEASE TAKE INTO CONSIDERATION THE SHapes OF MATRICES IT GIVE CLUES TO HOW TO STORE THE GRADIENTS. Use .shape and print them and see how they are.

dlogprobs=  torch.zeros_like(logprobs) # or torch.zeros_like(logprobs)
dlogprobs[range(n), Yb]=-1.0/n# loss= -sum(1,n)logprobs[xn,yn]/n => dlogprobs(xi,yi)= d(-sum(1,n)logprobs[xn,yn]/n)/d(logprobs(xi,yi))=-sum(1,n)del_in*del_in/n=-1.0/n
cmp('logprobs', dlogprobs, logprobs)

dprobs=torch.zeros((n_embd * block_size),vocab_size) # or torch.zeros_like(logprobs)
dprobs= dlogprobs * (1/ probs)# dprobs= dloss/dlogprobs * dlogprobs/dprobs = -1/n * (1/ probs)
cmp('probs', dprobs, probs)

dcounts_sum_inv= (dprobs * counts).sum(1, keepdim=True)#dcounts_sum_inv=dloss/dlogprobs * dlogprobs/dprobs * dprobs/dcounts_sum_inv= (-1/n * 1/prob * counts).sum(1, keepdim=True) as the dimension of the probs is 32 * 27 and of count_sum_inv is 32 *1 implies that the dierevative wrt each elemsnts in probs has to be summed across the 27 columns and stored into the repective row.
cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)

dcounts_sum= -dcounts_sum_inv * counts_sum**(-2)#dcounts_sum=dloss/dlogprobs * dlogprobs/dprobs * dprobs/dcounts_sum_inv * dcounts_sum_inv/dcounts_sum= -dcounts_sum_inv * count_sum**(-2)
cmp('counts_sum', dcounts_sum, counts_sum)

dcounts= dprobs * counts_sum_inv + torch.ones_like(counts)*dcounts_sum # dcounts= dloss/dlogprobs * dlogprobs/dprobs * dprobs/dcounts + dloss/dlogprobs * dlogprobs/dprobs * dprobs/dcounts_sum_inv * dcounts_sum_inv/dcounts_sum *dcounts_sum/dcounts
cmp('counts', dcounts, counts)

dnorm_logits= dcounts * norm_logits.exp() #dnorm_logits= dloss/dlogprobs * dlogprobs/dprobs * dprobs/dcounts * dcounts/dnorm_logits
cmp('norm_logits', dnorm_logits, norm_logits)

dlogit_maxes= (-dnorm_logits).sum(1, keepdim=True)#dnorm_logits= dloss/dlogprobs * dlogprobs/dprobs * dprobs/dcounts * dcounts/dnorm_logits * dnorm_logits/dlogit_maxes , .sum() same logic as above for counts_sum_inv
cmp('logit_maxes', dlogit_maxes, logit_maxes)

dlogits= dnorm_logits.clone()+ F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes#dnorm_logits= dloss/dlogprobs * dlogprobs/dprobs * dprobs/dcounts * dcounts/dnorm_logits * dnorm_logits/dlogits clone soi that ther is a copy created. The second part we use one_hot encoding vector becasue the logit.max consist of the maximun of values in a given row. logits.max(1).indices find the index of the max in eaxh row adn thats where yopu get the 1 in one_hoot encoding and the second argument gives the size for the one_hot encoding vector which is number of columns here.
cmp('logits', dlogits, logits)

dh= dlogits @ W2.T # t is the transpose here
cmp('h', dh, h)

dW2=h.T@dlogits
cmp('W2', dW2, W2)

db2= dlogits.sum(0) # here we didnt us the keepdim= True as the dimension as the decaltion of b2 or b1 i.e. (n_hidden,) allows for broadcasting and is efficient to use as the matrix is mumimally declared and it adds the bias on the fly repeatedly using the CPU memory which makes it very fast
cmp('b2', db2, b2)

dhpreact=dh * (1.0-h**2)
cmp('hpreact', dhpreact, hpreact)

dbngain = (dhpreact * bnraw).sum(0, keepdim=True) # the sum is across the columns or the dimsnion one has the size of bngain is 1 x n_hidden whihc will also be th size of the dbngain, so we sum the derivative contribution from all the elemsts acroos the respective rows
cmp('bngain', dbngain, bngain)

dbnbias = dhpreact.sum(0, keepdim=True)
cmp('bnbias', dbnbias, bnbias)

dbnraw= bngain * dhpreact
cmp('bnraw', dbnraw, bnraw)

dbnvar_inv= (dbnraw * bndiff).sum(0, keepdim= True)
cmp('bnvar_inv', dbnvar_inv, bnvar_inv)

dbnvar= dbnvar_inv * ((-0.5) * (bnvar +1e-5) ** (-1.5))
cmp('bnvar', dbnvar, bnvar)

dbndiff2= dbnvar * 1.0/(n-1) * torch.ones_like(bndiff2) # torch.one_like(bndiff2) because here the dimenionality of bnvar is 1 x n_hidden and dbndiff2 is n_hidden X n_hidden
cmp('bndiff2', dbndiff2, bndiff2)

dbndiff = dbnraw * bnvar_inv + 2.0*dbndiff2 * bndiff # bndiff is in two places one inn  bnraw and second in bndiff2
cmp('bndiff', dbndiff, bndiff)

dbnmeani = -dbndiff.sum(0, keepdim=True) # as meani is sum over the rows and a havs the dimsnions of 1 * n_hidden
cmp('bnmeani', dbnmeani, bnmeani)

dhprebn= dbndiff.clone() +1.0/n * (torch.ones_like(hprebn) * dbnmeani)
cmp('hprebn', dhprebn, hprebn)

dembcat = dhprebn @ W1.T
cmp('embcat', dembcat, embcat)

dW1 = embcat.T @ dhprebn
cmp('W1', dW1, W1)

db1= dhprebn.sum(0)
cmp('b1', db1, b1)

demb= dembcat.view(emb.shape) # becaus ethe shape of emb in batch size * blocksize * n_embd unlike the shape of embcat which is batchsize *(blocksize * n_embd) which is 2d unlike the first one whihc is 3d
cmp('emb', demb, emb)

dC = torch.zeros_like(C)
for k in range(Xb.shape[0]):
    for j in range(Xb.shape[1]): # here we moving across the tensor Xb
        ix= Xb[k,j] # here we find the integer corrsoponoing the charcter from each element of X
        dC[ix]+=demb[k,j] # so here we add the gradient in the demb corrseponding to a to a character into the row of the dC matrix correspodning to that character, adding because when a varable if part of multiplr fucntion then the gradient corresponding to that varbaible is equal to the sum of derovative for all the functions under consideration


# Exercise 2: backprop through cross_entropy but all in one go
# to complete this challenge look at the mathematical expression of the loss,
# take the derivative, simplify the expression, and just write it out

dlogits = F.softmax(logits,1) # teh derivation is in sheets where   I solved it
dlogits[range(n),Yb]-=1
dlogits/=n # because of the 1/n for aveaegr eloss as we backprogarte throigh aveagrge loss
cmp('logits',dlogits,logits)
