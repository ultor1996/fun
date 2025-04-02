import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
words=open('names.txt','r').read().splitlines() #raed the file and split it ijnto words and stor the words in python list word
#print(words[1:])#first we build bigram model
b={}

#for w in words:
 #  chs=['<S>']+list(w)+['<E>']
  # for ch1,ch2 in zip(chs,chs[1:]): #prints the pair of the first two character
   #    bigram=(ch1,ch2)
    #   b[bigram]=b.get(bigram,0)+1# this counts the number of occurnaces of the bigram, the value for
                                # the key bigram in the dictionary b is its occurence in the dictionary and since the keys are unique
                                # everytime it occurs it adds to its occurence by 1
       #print(ch1,ch2)
#print(b)# display dictionary the key and its value, which is here an occurnece
#print(sorted(b.items(), key=lambda kv: -kv[1]))# lambda is a keyword which is used to create anonymous funciotns i.e fucntions wihtout name
                                               # generally one has to tell the element of the list to print using lambda however here we are alrady in
                                               # a predifned function which is moving through its elemenst so we dont have to specifiy.
N=torch.zeros((27,27),dtype=torch.int32)# empty tensor to store the 2d array of the bigram pairs as row and columns and entry as the occurence
chars=sorted(list(set(''.join(words))))# sorted lists of 26 alphabets
stoi={s:i+1 for i,s in enumerate(chars)}# creates a dict which has a map such that each alphabet has a corresponind integer assocuated wiht it
stoi['.']=0
itos = {i:s for s,i in stoi.items()}# inverse mapping of s:i
for w in words:
    chs=['.']+list(w)+['.']
    for ch1,ch2 in zip(chs,chs[1:]): #prints the pair of the first two character
      ix1=stoi[ch1]
      ix2=stoi[ch2]
      N[ix1,ix2]+=1
plt.figure(figsize=(14,14))
plt.imshow(N,cmap='Blues')
for i in range(27): #just a nicer way to visulaize the matrix N
   for j in range(27):
       chstr = itos[i] + itos[j]
       plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
       plt.text(j, i, N[i,j].item(), ha="center", va="top", color='gray')
#plt.axis('off');
#plt.show()# uncomment to see the plot
#now we have the biagram frequncies in a matrix we calucalte the proabilities of occurnces which will be used further to predict the outcome
#p=N[0].float()#N[0] givses the whole first row
#p=p/p.sum()
#print(p)
#g=torch.Generator().manual_seed(2147483647)#A Generator in PyTorch is a pseudorandom number generator that helps create reproducible random numbers. Let's break down your code
#ix=torch.multinomial(p,num_samples=1, replacement=True, generator=g).item()# torcc.multinomial takesa probaility distribution and generates and integr sample which mimics the probability distribution
g = torch.Generator().manual_seed(2147483647)
P=(N+1).float()
P/=P.sum(1, keepdim=True)#sums the N matrix alonmg the row and stors it in a column matrix of dimen 1,27 # take care of braodcasting
for i in range(30):
    out=[]
    ix=0
    while True:
        p = P[ix]  # N[0] givses the whole first row
        #p = p / p.sum()
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()  # torcc.multinomial takesa probaility distribution and generates and integr sample which mimics the probability distribution
        out.append(itos[ix])
        if ix == 0:
            break
    #print(''.join(out))

# GOAL: maximize likelihood of the data w.r.t. model parameters (statistical modeling)
# equivalent to maximizing the log likelihood (because log is monotonic)
# equivalent to minimizing the negative log likelihood
# equivalent to minimizing the average negative log likelihood

# log(a*b*c) = log(a) + log(b) + log(c)
log_likelihood = 0.0
n = 0

for w in words:
#for w in ["andrejq"]:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    prob = P[ix1, ix2]
    logprob = torch.log(prob)
    log_likelihood += logprob
    n += 1
    #print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}')


#now the neural netowrk approach
# create the training set of bigrams (x,y)
#xs, ys = [], []

#for w in words[:1]:
 #   chs = ['.'] + list(w) + ['.']
  #  for ch1, ch2 in zip(chs, chs[1:]):
   #     ix1 = stoi[ch1]
    #    ix2 = stoi[ch2]
     #   print(ch1, ch2)
      #  xs.append(ix1)
       # ys.append(ix2)

#xs = torch.tensor(xs)
#ys = torch.tensor(ys)
# now in order to feed this to the neural network we can not use interger indices because when muliplied by weight they will lose their identitiy so we enocde each integer corresponidng to  the alphabet to an vector
# such that they they are 27 dimensipnal and the integer correesoning top the alphabet will be 1 rest will be zero for eg for e, i=5 and vector is=[0,0,0,0,1,0,0.....,0]
# randomly initialize 27 neurons' weights. each neuron receives 27 inputs this is to generate the counts matrix we obatined above. now since the prodcuts of the inputs ad neuron weights will be real number we have to have interperat it in a special way
#g = torch.Generator().manual_seed(2147483647)
#W = torch.randn((27, 27), generator=g)
#xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
#logits = xenc @ W # predict log-counts
#counts = logits.exp() # counts, equivalent to N
#probs = counts / counts.sum(1,keepdims=True) # probabilities for next character
 # btw: the last 2 lines here are together called a 'softmax'

#nlls = torch.zeros(5)
#for i in range(5):
  # i-th bigram:
 # x = xs[i].item() # input character index
  #y = ys[i].item() # label character index
 # print('--------')
 # print(f'bigram example {i+1}: {itos[x]}{itos[y]} (indexes {x},{y})')
 # print('input to the neural net:', x)
 # print('output probabilities from the neural net:', probs[i])
 # print('label (actual next character):', y)
 # p = probs[i, y]
 # print('probability assigned by the net to the the correct character:', p.item())
 # logp = torch.log(p)
 # print('log likelihood:', logp.item())
 # nll = -logp
 # print('negative log likelihood:', nll.item())
 # nlls[i] = nll

#print('=========')
#print('average negative log likelihood, i.e. loss =', nlls.mean().item())
# above was the result with a randomly chosen weights now we will try to optimize these weights using forward pass and back propagationas done in micrograd
# create the dataset
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples: ', num)

# initialize the 'network'
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)
# gradient descent
for k in range(150):
    # forward pass
    xenc = F.one_hot(xs, num_classes=27).float()  # input to the network: one-hot encoding
    logits = xenc @ W  # predict log-counts
    counts = logits.exp()  # counts, equivalent to N
    probs = counts / counts.sum(1, keepdims=True)  # probabilities for next character
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W ** 2).mean()# this second part is the regularisation
    print(loss.item())

    # backward pass
    W.grad = None  # set to zero the gradient
    loss.backward()

    # update
    W.data += -50 * W.grad
    # finally, sample from the 'neural net' model
g = torch.Generator().manual_seed(2147483647)

for i in range(5):

    out = []
    ix = 0
    while True:

        # ----------
        # BEFORE:
        # p = P[ix]
        # ----------
        # NOW:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W  # predict log-counts
        counts = logits.exp()  # counts, equivalent to N
        p = counts / counts.sum(1, keepdims=True)  # probabilities for next character
        # ----------

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))


#trigram model
stoi={s:i+1 for i,s in enumerate(chars)}# creates a dict which has a map such that each alphabet has a corresponind integer assocuated wiht it
stoi['..']=0
itos = {i:s for s,i in stoi.items()}# inverse mapping of s:i
xs, ys = [], []
for w in words:
    # Add two start/end tokens instead of one to handle trigrams properly
    chs = ['..'] + list(w) + ['..']
    # Create triplets instead of pairs
    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        ix3 = stoi[ch3]
        # For trigram model, input is a pair of characters (context), output is the next character
        xs.append((ix1, ix2))
        ys.append(ix3)

# Create a simplified index mapping for character pairs
# Instead of storing all combinations, use a formula: index = first_char_idx * vocab_size + second_char_idx
vocab_size = 27  # Size of the character vocabulary


# Simple helper functions to convert between pair indices and single indices
def get_single_index(pair):
    # Convert a character pair (ix1, ix2) to a single context index
    ix1, ix2 = pair
    return ix1 * vocab_size + ix2


def get_pair_from_index(idx):
    # Convert a single context index back to a character pair (ix1, ix2)
    ix1 = idx // vocab_size
    ix2 = idx % vocab_size
    return (ix1, ix2)


# Convert our character pairs to single indices using the formula
xs_indices = [get_single_index(x) for x in xs]
xs_indices = torch.tensor(xs_indices)
ys = torch.tensor(ys)
num = xs_indices.nelement()
print('number of examples: ', num)

# The total number of possible contexts is vocab_size^2 = 27*27 = 729
context_size = vocab_size * vocab_size

# Initialize the 'network' - now it's mapping from context (pair of chars) to next char
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((context_size, vocab_size), generator=g, requires_grad=True)

# Gradient descent
for k in range(150):
    # Forward pass
    xenc = F.one_hot(xs_indices, num_classes=context_size).float()  # one-hot encoding of context
    logits = xenc @ W  # predict log-counts
    counts = logits.exp()  # counts, equivalent to N
    probs = counts / counts.sum(1, keepdims=True)  # probabilities for next character
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W ** 2).mean()
    print(loss.item())

    # Backward pass
    W.grad = None  # set to zero the gradient
    loss.backward()

    # Update
    W.data += -50 * W.grad

# Generate samples from the trigram model
g = torch.Generator().manual_seed(2147483647)

for i in range(5):
    out = []
    # Start with '..' (double start token)
    ix1, ix2 = 0, 0  # Assuming '.' is at index 0
    while True:
        # Convert the current context to a single index using the formula
        context_idx = get_single_index((ix1, ix2))

        # Get next character probabilities
        xenc = F.one_hot(torch.tensor([context_idx]), num_classes=context_size).float()
        logits = xenc @ W  # predict log-counts
        counts = logits.exp()  # counts, equivalent to N
        p = counts / counts.sum(1, keepdims=True)  # probabilities for next character

        # Sample the next character
        ix3 = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()

        # Add to output
        out.append(itos[ix3])

        # Move window forward (current pair becomes previous pair, new char becomes current)
        ix1, ix2 = ix2, ix3

        # If we hit an end token, stop
        if ix3 == 0:
            break
    print(''.join(out))