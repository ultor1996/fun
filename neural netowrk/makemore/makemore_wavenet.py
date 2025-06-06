import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
#reading the words
words = open('names.txt','r').read().splitlines()
print(len(words))
print(max(len(w) for w in words))
print(words[:8])
#buidling the vocabulary
chars= sorted(list(set(''.join(words))))
stoi={s:i+1 for i,s in enumerate(chars)}
stoi['.']=0
itos={i:s for s,i in stoi.items()}
vocab_size=len(itos)
print(itos)
print(vocab_size)
#shuffle up the words
import random
random.seed(42)
random.shuffle(words)
#building the dataset
block_size=8 # context lenght: how many characters do we take to predict the next one
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


class Linear:
    def __init__(self,fan_in, fan_out, bias= True): # this called to create the opbject of a class and initilaizxe its parmaeters like making an object such as P=Linear(4,10,bias= True)
        self.weight= torch.randn((fan_in,fan_out)) / fan_in**0.5 # not using the gain factor here just the factor to make variuance one
        self.bias= torch.zeros(fan_out) if bias else None

    def __call__(self, x):# this is when the already declared or inilized elements is called to perform an associated operation for eg P(x) p is alredy declared above.
        self.out= x @ self.weight
        if self.bias is not None :
            self.out+=self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ( [] if self.bias is None else [self.bias])
#-----------------------------------------------------------------------------
class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1 ):
        self.eps=eps
        self.momentum= momentum
        #parameters ( trained with backprop)
        self.gamma= torch.ones(dim)
        self.beta =torch.zeros(dim)
        #buffers (trained with a running 'momentum update'
        self.running_mean= torch.zeros(dim)
        self.running_var=torch.ones(dim)

    def __call__(self, x):
        #forward pass
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3: # because we are usign wavenet now
                dim = (0, 1)
            xmean = x.mean(dim, keepdim=True)  # batch mean
            xvar = x.var(dim, keepdim=True)  # batch variance
        else:
            xmean=self.running_mean
            xvar= self.running_var
        xhat=(x-xmean)/torch.sqrt(xvar-self.eps) # normalize to unit varinace
        self.out = self.gamma* xhat + self.beta
        #update the uffers
        if self.training:
            with torch.no_grad():
                self.running_mean=(1- self.momentum)*self.running_mean + self.momentum*xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

#------------------------------------------------------------------------------------------
class Tanh:
    def __call__(self, x):
        self.out=torch.tanh(x)
        return self.out
    def parameters(self):
        return []
#-------------------------------------------------------------------------------------------------
class Embedding: # create embedding tensor
    def __init__(self,num_embeddings, embedding_dim):
        self.weight=torch.randn((num_embeddings,embedding_dim))

    def __call__(self, IX):
        self.out=self.weight[IX]
        return self.out
    def parameters(self):
        return [self.weight]

#-------------------------------------------------------------------------------------------------------
class FlattenConsecutive:
    # this class objects flattens the input tesnor which consists of data and embeddings into a another tensor which works such that from a full batch of data a small buhc is fed to the first linear layer after its forwards pass in the seocnd layer another batch of data is passed and so on and son forth  see wavenet picture online

    def __init__(self,n): # n is the number of elemnts that we want to be concatenated in the final dimension
        self.n=n
    def __call__(self, x):
        B, T, C= x.shape
        x=x.view(B,T//self.n, C*self.n) # here the first dimsion reamins the same, the last dimeion which was the size of the embedding vector is now the number of the characters/elemnst * embedding dimension and for the seocnd dimesnionis T/n as we broke the context groups of n
        if x.shape[1]==1:
            x=x.squeeze(1)
        self.out= x
        return self.out

    def parameters(self):
       return []
#--------------------------------------------------------------------------------------------------------

class Sequential: # this class creates an object which is a array of differnet kinds of layers of a neural netwoprk and return there forwards pass for a gioven data set
    def __init__(self,layers):
        self.layers=layers
    def __call__(self, x):
        for layer in self.layers:
            x=layer(x)
        self.out=x
        return self.out
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
#---------------------------------------------------------------------------------------------------
torch.manual_seed(42)

n_embd= 24 # the dimenionslaioty if the character embeddings vectors
n_hidden = 128 # the number of the neurons in the hidden layer of the MLP


model=Sequential([Embedding(vocab_size, n_embd),
                  FlattenConsecutive(2),Linear(n_embd*2, n_hidden, bias = False), BatchNorm1d(n_hidden), Tanh(),
                  FlattenConsecutive(2),Linear(n_hidden*2, n_hidden, bias = False), BatchNorm1d(n_hidden), Tanh(),
                  FlattenConsecutive(2),Linear(n_hidden*2, n_hidden, bias = False), BatchNorm1d(n_hidden), Tanh(),
                  Linear(n_hidden, vocab_size),])  #here istead of layers we created to class sqeuential look up for there description

#parameter_init
with torch.no_grad():
    model.layers[-1].weight*= 0.1 # last layer make less confident
parameters = model.parameters()
print(sum(p.nelement() for p in parameters)) # number of parametes in total

for p in parameters:
    p.requires_grad = True

max_steps = 200000
batch_size = 32
lossi=[]

# putting batchnorm in train mode
for layer in model.layers:
    if hasattr(layer, 'train'):
        layer.train()

for i in range(max_steps):
    # mini batch construct
    ix=torch.randint(0,Xtr.shape[0],(batch_size,))
    Xb, Yb= Xtr[ix], Ytr[ix]

    #forward passs
    # emb=C[x]
    # x=emb.view(emb.shape[0],-1) #conctactenatein
    #for layer in layers: #here 1st is the embeding, then concatenations and then  the lijnear and otehr layer sof te n eurla network
       # x= layer(x)
    logits=model(Xb)
    loss=F.cross_entropy(logits,Yb) # loss functions

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # updae : simple SGD
    lr = 0.1 if i < 150000 else 0.01 # step learning rate decay
    for p in parameters:
        p.data+=-lr * p.grad

    # track stats
    if i % 10000 ==0: #p[rint every once in a while
       print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())

plt.plot(torch.tensor(lossi).view(-1,1000).mean(1))# since there are a lot of steps we averaged the loss value over each thpusand steps and plot them as a result we get 200 data points
plt.show()
# put the bacthnorm in eval mode
for layer in model.layers:
    if hasattr(layer,'train'):
        layer.eval()

#evaluate the loss
@torch.no_grad() # this decorator disablkes gradient tracking inside the whole fucntion underneath it
def split_loss(split):
    x,y={
        'train':(Xtr,Ytr),
        'val':(Xdev,Ydev),
        'test':(Xte,Yte),
    }[split]
    #emb=C[x]
    #x=emb.view(emb.shape[0],-1) #conctactenatein
    logits=model(x)
    loss=F.cross_entropy(logits,y)
    print(split,loss.item())

split_loss('train')
split_loss('val')

#sample from the model

for _ in range (20):
    out=[]
    context =[0]*block_size
    while True:
        # forward pass the neural net
        #emb=C[torch.tensor([context])]
        #x= emb.view(emb.shape[0],-1) #concatenate
        logits = model(torch.tensor([context]))
        probs= F.softmax(logits,dim=1)
        #sample from distribution
        ix = torch.multinomial(probs,num_samples=1).item()
        #shift the context window and track the samples
        context = context[1:] + [ix]
        out.append(ix)
        # if we sample the special '.' token, beak
        if ix == 0:
            break

    print(''. join(itos[i] for i in out))


