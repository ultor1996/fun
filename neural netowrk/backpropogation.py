import math
import matplotlib.pyplot as plt
import numpy as np
from graphviz import Digraph
import random
import torch


#def f(x):
 #   return 3*x**2-4*x+5

#xs=np.arange(-5,5,0.25)
#ys=f(xs)
#plt.plot(xs,ys)
#plt.show()


class Value:
    def __init__(self, data, _children=(), _op='', label =''):
        self.data = data
        self.grad=0.0
        self._backward =  lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label=label

    def __repr__(self):
        return f"value(data={self.data})"  # f is fomratted print where you can print varibale isnide a given text  statement, also its an automatic print statemnet

    def __add__(self, other):
        other =  other if isinstance(other,Value) else Value(other)
        out= Value(self.data + other.data, (self, other),
                     '+')  # add function show how to add the class objects and their elements individually
        def _backward():
            self.grad+=1.0*out.grad #here we are defining the gradient to be calculated automatiacally via the code rather than puttung invia hands
            other.grad+=1.0*out.grad# so for a node say t, t.grad=do/dn * dn/dt_n * dt_n/dt_n1 * ....... *dt_1/dt = t1.grad( everything except the last term or one can say a kind of global derivative) * dt_1/dt
                                   #= t1.grad * 1 ( as t is in t1 via the '+' operand), here t1.grad=out.grad
        out._backward=_backward
        return out
    def __mul__(self, other):
       other = other if isinstance(other, Value) else Value(other)
       out = Value(self.data * other.data, (self, other), '*')

       def _backward():
           self.grad += other.data* out.grad  # here we are defining the gradient to be calculated automatiacally via the code rather than puttung invia hands
           other.grad += self.data * out.grad  # so for a node say t, t.grad=do/dn * dn/dt_n * dt_n/dt_n1 * ....... *dt_1/dt = t1.grad( everything except the last term or one can say a kind of global derivative) * dt_1/dt
           # = t1.grad * the other term (for e.g. if the dependency of t1 on t is as t1= a * t, then dt_1/dt=a), here t1.grad=out.grad

       out._backward = _backward
       return out
    def __pow__(self,other):
       assert isinstance(other,(int,float)) ,"only supporting int/float powers for now"
       out = Value(self.data**other,(self,),f'**{other}')

       def _backward():
           self.grad+=other*(self.data**(other-1))*out.grad
    def __rmul__(self, other): # because a*b might work howevre b*a may not wok in python. Thes funciton dont need the _bakward funciotn as they use the ones already develioped for add, mul, tanh, exp
        return self.__mul__(other)
    def __truediv__(self, other):
        return self.__mul__(other.__pow__(-1))
    def __neg__(self):
        return self.__mul__(-1)
    def __sub__(self,other):
        return self.__add__(other.__neg__())
    def tanh(self):
        x=self.data
        t=(math.exp(2*x)-1)/(math.exp(2*x)+1)
        out =  Value (t,(self,),'tanh')

        def _backward():
            self.grad += (1-t**2)* out.grad  # here we are defining the gradient to be calculated automatiacally via the code rather than puttung invia hands
            # so for a node say t, t.grad=do/dn * dn/dt_n * dt_n/dt_n1 * ....... *dt_1/dt = t1.grad( everything except the last term or one can say a kind of global derivative) * dt_1/dt
            # = t1.grad * (1-data**2) (dt_1/dt=1-t.data**2), here t1.grad=out.grad
        out._backward = _backward
        return out

    def exp(self):
        x=self.data
        out=Value(math.exp(x), (self,), 'exp')
        def _backward():
            self.grad+=math.exp(x)* out.grad
        out._backward= _backward
        return out
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v): # this makes a list of the nodes going from last nodes towards earleuir ones in topo
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad=1.0
        for node in reversed(topo): # then in this we go through the topo in opposite direction that is from 1st node to last
            node._backward()# to fill in the gradients going backwards to the first node


a = Value(2.0,label='a')
b = Value(-3.0,label='b')
c = Value(10.0,label='c')
e=a.__mul__(b); e.label='e'
d = e+c; d.label='d'
f = Value(-2.0, label='f')
L = d.__mul__(f); L.label = 'L'
#print(d._prev)

def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    def build(v): # build is called with d via trace in draw_dot
        if v not in nodes:
            nodes.add(v)
            #print("nodes",nodes)
            for child in v._prev:
                edges.add((child, v))
             #   print("edges",edges)
                build(child)

    build(root)
    return nodes, edges
def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})  # format is the format pf the output file, rankdir controls the laout of the graph here LR

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name=uid, label="{%s | data %.4f | grad %.4f }" % ( n.label, n.data, n.grad ), shape='record')
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name=uid + n._op, label=n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op) # n1 and n2 move in each subset of the set

    #dot.render('forward_pass', format='png', view=True)# forward pass plot
    #dot.render('node_input', format='png', view=True)# node contribution plot
    #dot.render('node_out_activation', format='png', view=True)# node contribution with activation funciton pplot
    dot.render('example_net', format='png', view=True)
    return dot
L.grad=1.0
f.grad=4.0
d.grad=-2.0 #derivatives of L wrte L,f,d etc
c.grad=-2.0 # by chain rule dL/dd*dd/dc
e.grad=-2.0 # same chain rule reasoning above
a.grad=6.0
b.grad=-4.0
#draw_dot(L) # this is forward pass shows how calcaultions are happening has we move forwarduncomment to see the forward pass flow diagram
# now we will do back propogation where 'L' should signifys the loss function which we minimise to train the neural netowrk or ML model effectively

def lol(): # function to check various derivative or gradients here
    h=0.00001
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a.__mul__(b);
    e.label = 'e'
    d = e + c;
    d.label = 'd'
    f = Value(-2.0, label='f')
    L = d.__mul__(f);
    L.label = 'L'
    L1=L.data

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a.__mul__(b);
    e.label = 'e'
    d = e + c;
    d.label = 'd'
    f = Value(-2.0, label='f')
    L = d.__mul__(f);
    L.label = 'L'
    L2=L.data+h

    print((L2-L1)/h)
lol()

# so basically how neural network works is that a neuron gets input from other neurons which are mulitplied via weights
#then addin all these contributions along with the bias of the receiveing neuron. Then one passes it thorugh a squashing function (here we will use tanh)
# This is the output of the current neuron or node
#img = mpimg.imread('C:/Users/shiva/OneDrive/Desktop/fun/neural netowrk/neuron.png')# see this image for visual of neurla netowrk node
#plt.imshow(img)
#plt.show()

plt.plot(np.arange(-5,5,0.2), np.tanh(np.arange(-5,5,0.2)));# activation function
plt.show();
# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1.__mul__(w1); x1w1.label = 'x1*w1'
x2w2 = x2.__mul__(w2); x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'# the collective input of the previous tow neurons to the current neuron/node
#draw_dot(n) # this shows the node contribution of the previous nodes/ neurons to the current one withput activation fcntion uncomment to plot
o=n.tanh(); o.label='o'
#o.grad=1.0# again doing the derivative things
#n.grad= 0.5#do/dn=1-tanh(n)**2=1-o**2=1-0.7071**2
#b.grad=0.5#do/db=do/dn * dn/db=do/dn * 1
#x1w1x2w2.grad=0.5#do/d(x1w1x2w2)=do/dn * dn/d(x1w1x2w2)= do/dn * 1
#x2w2.grad=0.5#do/d(x2w2)=do/dn * dn/d(x1w1+x2w2) * d(x1w1x2w2)/d(x2w2)=do/dn * 1 *1
#x1w1.grad=0.5#do/d(x1w1)=do/dn * dn/d(x1w1+x2w2) * d(x1w1x2w2)/d(x1w1)=do/dn * 1 *1
#w1.grad= 1.0#do/d(w1)=do/dn * dn/d(x1w1+x2w2) * d(x1w1+x2w2)/d(x1w1) *d(x1w1)/d(w1)=do/dn *1 *1 *x1
#w2.grad=0.0 #do/d(w1)=do/dn * dn/d(x1w1+x2w2) * d(x1w1+x2w2)/d(x2w2) *d(x2w2)/d(w2)=do/dn *1 *1 *x2
#x1.grad= -1.5#do/d(w1)=do/dn * dn/d(x1w1+x2w2) * d(x1w1+x2w2)/d(x1w1) *d(x1w1)/d(x1)=do/dn *1 *1 *w1
#x2.grad=0.5 #do/d(w1)=do/dn * dn/d(x1w1+x2w2) * d(x1w1+x2w2)/d(x2w2) *d(x2w2)/d(x2)=do/dn *1 *1 *w2
#draw_dot(o)#this shows the node contribution of the previous nodes/ neurons to the current one with activation fcntion uncomment to plot
# now we dont have to insert the derivative manually so we udodated the class and now we can  do it via funciotn even though the first .grad still needs to be initlaized
#o._backward()
#n._backward()
#b._backward()# nothing will happen as b is not connected backward to any other node
#x1w1x2w2._backward()
#x1w1._backward()
#x2w2._backward()
# now instead of calling backward again and again we can build toplogical sort done in value class, to do it automatically
o.backward()
#draw_dot(o)
#start making  neural nets, here we will start upon developing neura; network layer
class Neuron:
    def __init__(self,nin):# nin is the number of inputs
        self.w=[Value(random.uniform(-1,1)) for _ in range(nin)]# generate weights for differnet inputs in a list form. Number of weights is equal to the number of inputs i.e. the weight and the input vector has the same dimensionality
        self.b=Value(random.uniform(-1,1))# geenrate the node bias

    def __call__(self, x):#calculates the conitribution and out to one node
        # w * x + b
          act=Value(0.0)
          for wi, xi in zip(self.w,x): # zip(self.w,x) pairs up corresponidng elements of w with x in a tuple
              act+=wi.__mul__(xi)
          out=act.tanh() # pass the sum thorugh the activation function, by translation act is alos object of class value
          return out


class Layer:

    def __init__(self, nin, nout):# nin here is again number of inputs whereas nout is the number of neurons in the layer
        self.neurons = [Neuron(nin) for _ in range(nout)]# here we are creating a list with the obejcts of class Neuron, so nin is the dimensionality if the neuron and nput is how many o fthem you nat in a layer

    def __call__(self, x):
        outs = [n.__call__(x) for n in self.neurons] #  the function __call__ in the class neuron for the calculation of out
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:

    def __init__(self, nin, nouts): # nin is the number of inputs,here we make an objects out of layer class, nouts is now  list instead of numbr becaue we want list which number of neurons in each layer
        sz = [nin] + nouts #concatenating the two lists so the frist elemtn will be the number of iputs of the dmensionality of the input layer and nouts list of number of neurons in each layer
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]# objects of class layer need 2 input one is the number of inputs and second is the dimeiosnality of the layer

    def __call__(self, x):
        for layer in self.layers:
            x = layer.__call__(x)# the function __calls__ of layer class is used here
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
#x=[2.0,3.0,-1.0]
n= MLP(3,[4,4,1])
#draw_dot(n.__call__(x))
# example trainin
xs = [  # thes tructure of inout is such because the first list is a 3 input vector for the first nde of the first layer
        # and the first layer has 4 nodes so 3 X 4 matrice
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
#for x in xs:
 #   print(x)
ys = [1.0, -1.0, -1.0, 1.0] # desired targets
ypred=[n.__call__(x) for x in xs]# preiction of the nural network for now
print(ypred)
# now we caluclate loss and minimize it
loss = Value(0.0)