
import numpy as np
import pickle
#pickle.dump( x, open( "x.p", "wb" ) )
X= pickle.load( open( "x.p", "rb" ) )
Y= np.zeros([3,1])
Y[0]=1

model={
0:{'activation':'X',       'a':np.zeros((6400,1))},
1:{'activation':'relu',    'a':np.zeros((200,1)), 'da': np.zeros((200,1)), 'Z': np.zeros((200,1)), 'dZ':np.zeros((200,1)), 'W':np.random.randn(200,6400), 'dW':np.zeros((200,6400)) },
2:{'activation':'softmax', 'a':np.ones((3,1)),   'da': np.zeros((3,1)),   'Z': np.zeros((3,1)), 'dZ':np.zeros((3,1)), 'W': np.random.randn(3,200), 'dW':np.zeros((3,200)) }
}

def forward(model, X):
    for l in model.keys()[1:]:
        if model[l]['activation']=='X':
            model[0]['a']=X
            print(1)
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
def backward(model, Y, learning_rate, update):
    for l in model.keys()[::-1]:
        if model[l]['activation']=='relu':
            model[l]['da']=np.ones(model[l]['a'].shape)
            model[l]['da'][model[l]['a']<=0]=0
            model[l]['dZ']=np.dot(model[l]['W'].T, model[l-1]['dZ'])
            model[l]['dZ']=np.multiply(model[l]['dZ'],model[l]['da'] )
            model[l]['dW']=np.dot(model[l]['dZ'],model[l-1]['a'].T)
            if update==1: model[l]['dW']-=learning_rate*model[l]['dW']
        elif model[l]['activation']=='softmax':
            model[l]['da']=1
            model[l]['dZ']=Y-model[l]['a']
            model[l]['dW']=np.dot(model[l]['dZ'],model[l-1]['a'].T)
            if update==1: model[l]['dW']-=learning_rate*model[l]['dW']
        else:
            pass
    return model

#-------------------------------------------------------------------------------





#TODO: fix backprop algorithm
for i in range(0,1000):
    model=forward(model,X)
    model=backward(model, Y, 0.05, update=1)
    l=loss(model,Y)
    print(i)
    print(model[2]['dZ'])
    print(model[2]['dW'][0][0])
    print(model[2]['W'][0][0])
