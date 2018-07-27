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




X=X.T # I need last variable to be the number of examples
(h0,m)=X.shape
h0,m

#Y:= One HOT y:= index of the right class
Y=np.zeros((K,m))
Y.shape



Y[range(N*K),y]+=1


h1=100
h2=3#classes
learning_rate=1e-0
update=1
reg=1e-3

model0={
0:{'activation':'X',       'a':np.zeros((h0,m))},
1:{'activation':'relu',    'a':np.zeros((h1,m)), 'da': np.zeros((h1,m)), 'Z': np.zeros((h1,m)), 'dZ':np.zeros((h1,m)), 'W':0.01*np.random.randn(h1,h0), 'dW':np.zeros((h1,h0)) },
2:{'activation':'softmax', 'a':np.ones((h2,m)),   'da': np.zeros((h2,m)),   'Z': np.zeros((h2,m)), 'dZ':np.zeros((h2,m)), 'W': 0.01*np.random.randn(h2,h1), 'dW':np.zeros((h2,h1)) }
}


def forward(model, X):
    for l in model.keys():
        #print(l)
        if model[l]['activation']=='X':
            model[0]['a']=X
        elif model[l]['activation']=='relu':
            model[l]['Z']=np.dot(model[l]['W'],model[l-1]['a'])
            model[l]['a']=model[l]['Z']
            model[l]['a'][model[l]['a']<0]=0
            #print(np.sum(model[l]['a']))
        elif model[l]['activation']=='softmax':
            model[l]['Z']=np.dot(model[l]['W'],model[l-1]['a'])
            model[l]['a']=np.exp(model[l]['Z'])/np.sum(np.exp(model[l]['Z']), axis=1, keepdims=True)
            #print(np.sum(model[l]['a']))
        else:
            pass
    return model
Y.shape[1]
def loss(model,Y, reg):
    '''
    Softmax loss
    '''
    L=model.keys()[-1]
    ymax=np.argmax(Y)
    correct_log_probs= -np.log(model[L]['a'][ymax])
    data_loss=np.sum(correct_log_probs)/Y.shape[1]
    reg_loss=(1.0/2.0)*reg*(np.sum(model[1]['W']**2)+np.sum(model[2]['W']**2))
    loss=data_loss+reg_loss
    return loss

def backward(model, Y, learning_rate, update, reg):# a:= partial_der; In all cases dVar:= (aL/aVar)
    '''
    BackProp Alg: Input dA[l], return dZ[l], dW[l], dA[l-1]
    '''
    for l in model.keys()[::-1]:
        #print(1)
        if model[l]['activation']=='relu':

            model[l]['dZ']=model[l]['da']
            model[l]['dZ'][model[l]['a']<0]=0#(aL/aA2)(aA2/aZ2)(aZ2/aA2) (aA2/aZ1)
            model[l]['dW']=np.dot(model[l]['dZ'],model[l-1]['a'].T)#(aL/aA2)(aA2/aZ2)(aZ2/aA2) (aA2/aZ1) (aZ1/aW1)
            model[l]['dW']+=reg*model[l]['W']

            if update==1: model[l]['W']-=learning_rate*model[l]['dW']
            #print(np.sum(model[l]['da']))
        elif model[l]['activation']=='softmax':
            #print(2)
            model[l]['dZ']=model[l]['a']-Y # (aL/aA2)(aA2/aZ2)
            model[l]['dW']=np.dot(model[l]['dZ'],model[l-1]['a'].T) #(aL/aA2)(aA2/aZ2)(aZ2/aW2)
            model[l]['dW']+=reg*model[l]['W']
            model[l-1]['da']=np.dot(model[l]['W'].T,model[l]['dZ']) #(aL/aA2)(aA2/aZ2)(aZ2/aA2)

            if update==1: model[l]['W']-=learning_rate*model[l]['dW']
            #print(np.sum(model[l]['da']))
        else:
            pass
    return model

#-------------------------------------------------------------------------------
#http://cs231n.github.io/neural-networks-case-study/
#TODO: Loop is ok but, loss increases with steps and eventually blows up due to division in forward by infinite due to exp(2000)
#Also my first loss is aroun 577 while example is around 1
model=model0.copy()
for i in range(0,10000):
    model=forward(model,X)
    model=backward(model,Y, learning_rate, update, reg)
    print(loss(model,Y, reg))




model0[2]['W']
