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


class Linear:
    def __init__(self,fan_in, fan_out, bias= True):
        self.weight= torch.randn((fan_in,fan_out)) / fan_in**0.5 # not using the gain factor here just the factor to make variuance one
        self.bias= torch.zeros(fan_out) if bias else None

    def __call__(self, x):
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
            xmean=x.mean(0, keepdim=True)# batch mean
            xvar=x.var(0, keepdim=True) #batch variance
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

torch.manual_seed(42)

n_embd= 10 # the dimenionslaioty if the character embeddings vectors
n_hidden = 200 # the number of the neurons in the hidden layer of the MLP

C= torch.randn((vocab_size, n_embd))
layers=[Linear(n_embd*block_size, n_hidden, bias = False), BatchNorm1d(n_hidden), Tanh(), Linear(n_hidden, vocab_size),]

#parameter_init
with torch.no_grad():
    layers[-1].weight*= 0.1 # last layer make less confident

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters)) # number of parametes in total
for p in parameters:
    p.requires_grad = True

# optimization
max_steps = 200000
batch_size = 32
lossi=[]

for i in range(max_steps):
    # mini batch construct
    ix=torch.randint(0,Xtr.shape[0],(batch_size,))
    Xb, Yb= Xtr[ix], Ytr[ix]

    #forward passs
    emb = C[Xb]
    x= emb.view(emb.shape[0],-1) # concatenate the vectors

    for layer in layers:
        if hasattr(layer,'train'):
            layer.train()

    for layer in layers:
        x= layer(x)
    loss=F.cross_entropy(x,Yb) # loss functions

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

plt.plot(lossi)
plt.show()
# put the bacthnorm in eval mode
for layer in layers:
    if hasattr(layer,'train'):
        layer.eval()


