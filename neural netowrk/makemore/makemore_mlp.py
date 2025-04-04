import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures
words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
print(itos)
# build the dataset

block_size = 3  # context length: how many characters do we take to predict the next one?
X, Y = [], []
for w in words: # this loop creates the datatset with variable context whose size can be changed by chnaging the block size
                #in the intial step the context is built such a way that the leading charcter is the label and the data correponidn gto it is the context/ block size time the end token which here is .
    # print(w)
    context = [0] * block_size

    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        #print(''.join(itos[i] for i in context), '--->', itos[ix])# context prints the . as 0---->. in itos
        context = context[1:] + [ix]  # crop and append

X = torch.tensor(X)
Y = torch.tensor(Y)

C=torch.randn((27,2))
emb=C[X]    #important line it embeds the 2 dimseional X tensor into 2 dimseional space from C, here the
print(emb.shape) # the shape of this array is 228146 x 3 x 2 as we have 228146 contexts, each context consists of 3 elements and each element is in a 2d space

#now we built the hidden layer
W1=torch.randn((6,100))# input is six as it will take 3 cahracters simultabeously and each charcter is 2d row and 100 is the number of neurons
b1=torch.randm(100)# bias for each neuron
#now we need to concatenate the last two dimesnions of emb as the matrix multiplciaoitn between the W1 and emb wont happen as they hasve differnce dimesionality
#so we trasnform the the emb from 228146 x 3 x 2 to 228146 x 6 by concatanting the matrix
#torch.cat(torch.unbind(emb,1),1)# torch.unbind(emb,1) splits the emb along the second dimesions an d.cat cntatenates along the new second dimeison after the unbinding whihc is equal to previos 3 diemsions
# or one can use view to change the struicture of tensor provided the numbre o felements remains the smae
h=torch.tanh(emb.view(-1,6)@W1+b1)# herhe teh braodcasting is correct as b1 is 100 dim and is added to 32 X 100 matrice, so here the b1 will first become 1 X 100 row vector and then the ro will copied into 32 identical rows and added to it
#also instead of 32 we used -1 in the first argument as the as pytorch will infer the nymber as it makes sure the number of componenet are conserved into the trasforamtion
#also view is more efficinet as it changes the exsisting tensior and not make anothe r tnesor
print(h.shape)

 # now we create the final layer
 W2=torch.randn((100,27)) # 100 is the numebr of neuirons in the alst layer and 27 the number of nerurons in th output layer
 b2=torch.randn(27)
 logits=h@W2+b2
