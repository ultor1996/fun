import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import graphviz

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
        self._prev = set(_children)
        self._op = _op
        self.label=label

    def __repr__(self):
        return f"value(data={self.data})"  # f is fomratted print where you can print varibale isnide a given text  statement, also its an automatic print statemnet

    def __add__(self, other):
        return Value(self.data + other.data, (self, other),
                     '+')  # add function show how to add the class objects and their elements individually

    def __mult__(self, other):
        return Value(self.data * other.data, (self, other), '*')


a = Value(2.0,label='a')
b = Value(-3.0,label='b')
c = Value(10.0,label='c')
e=a.__mult__(b); e.label='e'
d = e+c; d.label='d'
f = Value(-2.0, label='f')
L = d.__mult__(f); L.label = 'L'
#print(d._prev)
from graphviz import Digraph
def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    def build(v): # build is called with d via trace in draw_dot
        if v not in nodes:
            nodes.add(v)
            print("nodes",nodes)
            for child in v._prev:
                edges.add((child, v))
                print("edges",edges)
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

    dot.render('forward_pass', format='png', view=True)
    return dot
L.grad=1.0
f.grad=4.0
d.grad=-2.0 #derivatives of L wrte L,f,d etc
c.grad=-2.0 # by chain rule dL/dd*dd/dc
e.grad=-2.0 # same chain rule reasoning above
a.grad=6.0
b.grad=-4.0
draw_dot(L) # this is forward pass shows how calcaultions are happening has we move forward
# now we will do back propogation where 'L' should signifys the loss function which we minimise to train the neural netowrk or ML model effectively

def lol(): # function to check various derivative or gradients here
    h=0.00001
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a.__mult__(b);
    e.label = 'e'
    d = e + c;
    d.label = 'd'
    f = Value(-2.0, label='f')
    L = d.__mult__(f);
    L.label = 'L'
    L1=L.data

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a.__mult__(b);
    e.label = 'e'
    d = e + c;
    d.label = 'd'
    f = Value(-2.0, label='f')
    L = d.__mult__(f);
    L.label = 'L'
    L2=L.data+h

    print((L2-L1)/h)
lol()

# so basically how neural network works is that a neuron gets input from other neurons which are mulitplied via weights
#then addin all these contributions along wthe bias of the receiveing neuron one passes it thorugh a squashing function
# output of the current
img = mpimg.imread('C:/Users/shiva/OneDrive/Desktop/fun/neural netowrk/neuron.png')# see this image for visual of neurla netowrk node
plt.imshow(img)
#plt.show()

plt.plot(np.arange(-5,5,0.2), np.tanh(np.arange(-5,5,0.2)));
plt.show();