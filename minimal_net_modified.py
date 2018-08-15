#setup---------------------------------------------------------------------------
import numpy as np
np.random.seed(0)
#Data---------------------------------------------------------------------------
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


#SHAPES: X=[[data m],...[data m]](n,m), y=[label1,...labelM](,m), Y=OH(y)(K,M)
X=X.T # I need last variable to be the number of examples
(h0,m)=X.shape
h0,m

#Y:= One HOT y:= index of the right class
Y=np.zeros((K,m))
Y.shape[0]
range(Y.shape[0])
Y[y,range(Y.shape[1])]+=1

#Parameters---------------------------------------------------------------------

h1=100
h2=3#classes
learning_rate=1e-0
update=1
reg=1e-3



#Model--------------------------------------------------------------------------

def z(f,c): return np.zeros((f,c))
def o(f,c): return np.ones((f,c))
def r(f,c): return 0.01*np.random.randn(f,c)

model0={
0:{'activation':'X',       'a':z(h0,m)},
1:{'activation':'relu',    'a':z(h1,m), 'da': z(h1,m), 'Z': z(h1,m), 'dZ':z(h1,m), 'W':r(h1,h0), 'dW':z(h1,h0),'b':z(h1,1), 'db':z(h1,1)},
2:{'activation':'softmax', 'a':z(h2,m), 'da': z(h2,m), 'Z': z(h2,m), 'dZ':z(h2,m), 'W':r(h2,h1), 'dW':z(h2,h1),'b':z(h2,1), 'db':z(h1,1)}
}


def forward(model, X):
    for l in model.keys():
        #print(l)
        if model[l]['activation']=='X':
            model[l]['a']=X
        elif model[l]['activation']=='relu':
            model[l]['Z']=np.dot(model[l]['W'],model[l-1]['a'])+model[l]['b']
            model[l]['a']=model[l]['Z']
            model[l]['a'][model[l]['a']<0]=0
            #print(np.sum(model[l]['a']))
        elif model[l]['activation']=='softmax':
            model[l]['Z']=np.dot(model[l]['W'],model[l-1]['a'])+model[l]['b']
            model[l]['a']=np.exp(model[l]['Z'])/np.sum(np.exp(model[l]['Z']), axis=0, keepdims=True)
            #print(np.sum(model[l]['a']))
        else:
            pass
    return model
def loss(model,Y, reg):
    '''
    Softmax loss
    '''
    L=model.keys()[-1] # get last layer L=2
    ymax=np.argmax(Y,axis=0) # es lo mismo que tener y
    correct_log_probs= -np.log(model[L]['a'][ymax,range(Y.shape[1])])
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
            model[l]['dZ'][model[l]['a']<=0]=0#(aL/aA2)(aA2/aZ2)(aZ2/aA2) (aA2/aZ1)
            model[l]['dW']=np.dot(model[l]['dZ'],model[l-1]['a'].T)#(aL/aA2)(aA2/aZ2)(aZ2/aA2) (aA2/aZ1) (aZ1/aW1)
            model[l]['dW']+=reg*model[l]['W']
            model[l]['db']=np.sum(model[l]['dZ'], axis=1, keepdims=True)
            if update==1:
                model[l]['W']-=learning_rate*model[l]['dW']
                model[l]['b']-=learning_rate*model[l]['db']
            #print(np.sum(model[l]['da']))
        elif model[l]['activation']=='softmax':
            #print(2)
            model[l]['dZ']=(model[l]['a']-Y)/Y.shape[1] # (aL/aA2)(aA2/aZ2)
            model[l]['dW']=np.dot(model[l]['dZ'],model[l-1]['a'].T) #(aL/aA2)(aA2/aZ2)(aZ2/aW2)
            model[l]['dW']+=reg*model[l]['W']
            model[l]['db']=np.sum(model[l]['dZ'], axis=1, keepdims=True)
            model[l-1]['da']=np.dot(model[l]['W'].T,model[l]['dZ']) #(aL/aA2)(aA2/aZ2)(aZ2/aA2)

            if update==1:
                model[l]['W']-=learning_rate*model[l]['dW']
                model[l]['b']-=learning_rate*model[l]['db']
            #print(np.sum(model[l]['da']))
        else:
            pass
    return model

#LOOP------------------------------------------------------------------------------
model=model0.copy()
for i in range(0,10000):
    model=forward(model,X)
    print(np.argmax(model[2]['a'][:,5], axis=0))
    model=backward(model,Y, learning_rate, update, reg)
    if i % 1000 == 0:
      #print(np.sum(np.abs(model[2]['dW'])))
      print "iteration %d: loss %f" % (i, loss(model,Y, reg))

#-------------------------------------------------------------------------------
predicted_class = np.argmax(model[2]['a'], axis=0)
print 'training accuracy: %.2f' % (np.mean(predicted_class == y))
