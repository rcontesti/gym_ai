
import numpy as np
import pickle
#pickle.dump( x, open( "x.p", "wb" ) )
X= pickle.load( open( "x.p", "rb" ) )
X=X.reshape(X.shape[0],1)
Y= np.zeros([3,1])
Y[0]=1

model0={
0:{'activation':'X',       'a':np.zeros((6400,1))},
1:{'activation':'relu',    'a':np.zeros((200,1)), 'da': np.zeros((200,1)), 'Z': np.zeros((200,1)), 'dZ':np.zeros((200,1)), 'W':np.random.randn(200,6400), 'dW':np.zeros((200,6400)) },
2:{'activation':'softmax', 'a':np.ones((3,1)),   'da': np.zeros((3,1)),   'Z': np.zeros((3,1)), 'dZ':np.zeros((3,1)), 'W': np.random.randn(3,200), 'dW':np.zeros((3,200)) }
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
            model[l]['a']=np.exp(model[l]['Z'])/np.sum(np.exp(model[l]['Z']))
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

#-------------------------------------------------------------------------------
