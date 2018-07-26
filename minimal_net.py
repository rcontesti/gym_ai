# A bit of setup
import numpy as np
np.random.seed(0)
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D))
y = np.zeros(N*K, dtype='uint8')
for j in xrange(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j


(n,m)=X.shape
n,m
#Y:=index of each right class. it is not onehotcoded
h1=100
h2=3#classes

model0={
0:{'activation':'X',       'a':np.zeros((n,m))},
1:{'activation':'relu',    'a':np.zeros((h1,m)), 'da': np.zeros((h1,m)), 'Z': np.zeros((h1,n)), 'dZ':np.zeros((h1,n)), 'W':np.random.randn(h1,n), 'dW':np.zeros((h1,n)) },
2:{'activation':'softmax', 'a':np.ones((h2,m)),   'da': np.zeros((h2,m)),   'Z': np.zeros((h2,m)), 'dZ':np.zeros((h2,m)), 'W': np.random.randn(h2,h1), 'dW':np.zeros((h2,h1)) }
}

def forward(model, X):
    for l in model.keys():
        if model[l]['activation']=='X':
            model[0]['a']=X
        elif model[l]['activation']=='relu':
            model[l]['Z']=np.dot(model[l]['W'],model[l-1]['a'])
            model[l]['a']=model[l]['Z']
            model[l]['a'][model[l]['a']<0]=0
        elif model[l]['activation']=='softmax':
            model[l]['Z']=np.dot(model[l]['W'],model[l-1]['a'])
            model[l]['a']=np.exp(model[l]['Z'])/np.sum(np.exp(model[l]['Z']), axis=1, keepdims=True)
        else:
            pass
    return model

def loss(model,Y):
    '''
    Softmax loss
    '''
    L=model.keys()[-1]
    ymax=np.argmax(Y)
    return -np.log(model[L]['a'][ymax])

def backward(model, Y, learning_rate, update):# a:= partial_der; In all cases dVar:= (aL/aVar)
    '''
    BackProp Alg: Input dA[l], return dZ[l], dW[l], dA[l-1]
    '''
    for l in model.keys()[::-1]:
        if model[l]['activation']=='relu':
            model[l]['dZ']=model[l]['da']
            model[l]['dZ'][model[l]['a']<0]=0#(aL/aA2)(aA2/aZ2)(aZ2/aA2) (aA2/aZ1)
            model[l]['dW']=np.dot(model[l]['dZ'],model[l-1]['a'].T)#(aL/aA2)(aA2/aZ2)(aZ2/aA2) (aA2/aZ1) (aZ1/aW1)

            if update==1: model[l]['dW']-=learning_rate*model[l]['dW']

        elif model[l]['activation']=='softmax':
            model[l]['dZ']=Y-model[l]['a'] # (aL/aA2)(aA2/aZ2)
            model[l]['dW']=np.dot(model[l]['dZ'],model[l-1]['a'].T) #(aL/aA2)(aA2/aZ2)(aZ2/aW2)
            model[l-1]['da']=np.dot(model[l]['W'].T,model[l]['dZ']) #(aL/aA2)(aA2/aZ2)(aZ2/aA2)

            if update==1: model[l]['dW']-=learning_rate*model[l]['dW']
        else:
            pass
    return model

model=model0
model=forward(model,X)
model=backward(model,y, learning_rate=1e-0, update=1)

#-------------------------------------------------------------------------------


model[2]['a']


y.shape
model[2]['a'].shape
model[2]['a'][:,y]

for i in range(0,100):
    model=forward(model,X)
    model=backward(model,Y, learning, update=1)
    print(loss)
