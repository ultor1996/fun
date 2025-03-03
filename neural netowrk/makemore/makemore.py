import torch
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
plt.axis('off');
#plt.show()# uncomment to see the plot
#now we have the biagram frequncies in a matrix we calucalte the proabilities of occurnces which will be used further to predict the outcome
p=N[0].float()#N[0] givses the whole first row
p=p/p.sum()
print(p)
g=torch.Generator().manual_seed(2147483647)#A Generator in PyTorch is a pseudorandom number generator that helps create reproducible random numbers. Let's break down your code
ix=torch.multinomial(p,num_samples=1, replacement=True, generator=g).item()# torcc.multinomial takesa probaility distribution and generates and integr sample which mimics the probability distribution
print(itos[ix])


