#--LIBRARIES
import numpy as np
import cPickle as pickle
import gym


#--FUNCTIONS
def get_x(cur_prep, prev_prep):
    if prev_prep is None: prev_prep=np.zeros(cur_prep.shape)
    return cur_prep-prev_prep
def prepro(obs):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  obs = obs[35:195] # crop
  obs = obs[::2,::2,0] # downsample by factor of 2 and just keep the first[0] of the 3rd dim
  obs[obs == 144] = 0 # erase background (background type 1)
  obs[obs == 109] = 0 # erase background (background type 2)
  obs[obs != 0] = 1 # everything else (paddles, ball) just set to 1
  return obs.astype(np.float).ravel() # convert to float and stack in 1-D array with ravel
def forward_policy(x, parameters):
    "Probability of Taking action a given obs o"
    a={}
    z={}
    i=1
    a[0]=x
    z[i]=np.dot(parameters['W'+str(i)],a[i-1])           # (21)
    a[i]=np.maximum(z[i],0)                               # (22) ReLU Activation
    i+=1
    z[i]=np.dot(parameters['W'+str(i)],a[i-1])
    p=np.exp(z[i])/np.sum(np.exp(z[i]))
    return p,z,a
def get_action(p,eps,action_dic):
    '''
    With probability 1-epsilon pick index of maximun value
    else explore sample index from the remaining of all indexes after removing
    the index of the maximun value
    '''

    imax=np.argmax(p)
    idxs=action_dic.keys()
    other_choices=idxs[0:imax]+idxs[imax+1:]

    if   np.random.uniform()>eps:
        action_index=imax
    else:
        action_index=np.random.choice(other_choices)

    action=action_dic[action_index]

    return action, action_index
def initialize_parameters(n):
    parameters={}
    for l in range(1,len(n)):
        parameters['W'+str(l)]=np.zeros([n[l],n[l-1]])
    return parameters
def get_advantage(rewards):
    return np.ones(len(rewards))*rewards[-1]/len(rewards)


#--PARAMETERS
'''
 layer[0]               layer[1]:fc                 layer[2]:'relu'               layer[3]:'relu'        layer[4]:'softmax'
  [X]---(.)  ------->[fc]:np.dot(W1,layer[0])-----> relu[layer[1]]:------(.)----np.dot(W3,layer[2])-------softmax(layer[3])
(6400,1) |           (200,6400)(6400,1)             (200,1)              |          (3,200)(200,1)             (3,1)
         |                (200,1)                                        |              (3,1)
  [W1]---|                                                        [W3]---|
(200,6400)                                                       (3,200)
'''

layers={
0: {'cath': 'X',      'shape':  (6400,1) },
1: {'cath': 'fc',     'shape':  (200,1), },
2: {'cath': 'relu',   'shape':  (200,1) },
3: {'cath': 'fc',     'shape':  (3,1)  },
4: {'cath': 'softmax','shape':  (3,1)}
}

def initialize_parameters(layers):
    parameters={}
    for l in layers.keys():
        if layers[l]['cath']=='fc':
            parameters['W'+str(l)]=np.zeros([layers[l]['shape'][0],layers[l-1]['shape'][0]])
        else:
            pass
    return parameters

parameters=initialize_parameters(layers)

def functions(functions,layers):
    if...


for l in range(len(layers)):
    if l['cath']=='X':
       forward.append[layer]_0=X
    elif l['cath']=='fc':
        forward.append(parameters['W']
}

backward={}



formulas:
{'fc':

}



n=[6400,200,3] # layer units n[0]=m
eps=.10 # exploration parameter

#-- START
action_dic={0:3,1:0,2:2} # 3==UP, 2==DOWN , 0==NOTHING ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'] env.unwrapped.get_action_meanings()
env = gym.make("Pong-v0")
obs = env.reset()

prev_prep=None
cur_prep=prepro(obs)
x=get_x(cur_prep, prev_prep)
prev_prep=cur_prep
x.shape

parameters=initialize_parameters(n)
p,z,a=forward_policy(x,parameters)
action, action_index=get_action(p,eps,action_dic)

# step the environment and get new measurements
P=[]
reward=0
rewards=[reward]
actions=[]
while reward==0:
    P.append[p]
    action=get_action(p,eps,action_dic)
    observation, reward, done, info = env.step(action)
    rewards.append(reward)
print(len(rewards))

#DONE: learning Understand Update: December 9, 2016 - alternative view. in http://karpathy.github.io/2016/05/31/rl/
#TODO:Necesito tener los layers mas ordenados en un diccionario y hacer el backprop tambien con un diccionario que elige el metodo.
#TODO: Tengo un problema con multiplicar y transpones arrays del tipo (n,). Necesito llevarlos a (n,1) por lo menos. Como soluciona esto NG?
# check:  np.array([1,2,3])[:, np.newaxis]
# Transpose: For a 1-D array, this has no effect.
# check: https://www.coursera.org/lecture/neural-networks-deep-learning/a-note-on-python-numpy-vectors-87MUx
# https://sebastianraschka.com/pdf/books/dlb/appendix_f_numpy-intro.pdf
#TODO: Build a layers dictionary. With That and X obtain fowrd. Same with backprop. Backprop f(foward_prop o por lo menos de los elementos que genera la foward)

grad=p
#BackProp
#Y es solamente 1 en el elemento que es la accion correcta o el elemento que elijo
#Cero en todo los demas
grad[action_index]=grad[action_index]-1
(a[1].transpose()).shape

def res(x): return x.reshape(x.shape[0],1)


dW2=(np.matmul(res(grad), res(a[1]).T))
dW2.shape


if reward!=0
action[action_index]
advantage=get_advantage(rewards)
np.multiply(advantage, np.array(P))


#############
import numpy as np
import pickle
#pickle.dump( x, open( "x.p", "wb" ) )
X= pickle.load( open( "x.p", "rb" ) )



#--PARAMETERS
'''
[x] (6400,1) layer 0
 |
 |
 W(h1,)X
 layer[0]               layer[1]:fc                 layer[2]:'relu'               layer[3]:'relu'        layer[4]:'softmax'
  [X]---(.)  ------->[fc]:np.dot(W1,layer[0])-----> relu[layer[1]]:------(.)----np.dot(W3,layer[2])-------softmax(layer[3])
(6400,1) |           (200,6400)(6400,1)             (200,1)              |          (3,200)(200,1)             (3,1)
         |                (200,1)                                        |              (3,1)
  [W1]---|                                                        [W3]---|
(200,6400)                                                       (3,200)
'''

action_dic={0:3,1:0,2:2} # 3==UP, 2==DOWN , 0==NOTHING
model={
0:{'operation':'X',       'shape': (6400,1),'value': None, 'parameters':None, 'derivatives':None},
1:{'operation':'fc',      'shape': (200,1), 'value': None, 'parameters':None, 'derivatives':None},
2:{'operation':'relu',    'shape': (200,1), 'value': None, 'parameters':None, 'derivatives':None},
3:{'operation':'fc',      'shape': (3,1),   'value': None, 'parameters':None, 'derivatives':None},
4:{'operation':'softmax', 'shape': (3,1),   'value': None, 'parameters':None, 'derivatives':None},
5:{'operation':'action',  'shape': (1,1),   'value': None, 'parameters':{'action_dic':action_dic, 'exploration_prob':0}, 'derivatives':None},
}

np.zeros((6400,1))
model={
0:{'activation':'X',       'a':np.zeros((6400,1))},
1:{'activation':'relu',    'a':np.zeros((200,1)), 'da': np.zeros((200,1)), 'Z': np.zeros((200,1)), 'dZ':np.zeros((200,1)), 'W':np.zeros((200,6400)), 'dW':np.zeros((200,6400)) },
2:{'activation':'softmax', 'a':np.zeros((3,1)),   'da': np.zeros((3,1)),   'Z': np.zeros((3,1)), 'dZ':np.zeros((3,1)), 'W':np.zeros((3,200)), 'dW':np.zeros((3,200)) }
}

def forward(model, X):
    for l in model.keys()[1:]:
        if 'activation'=='X':
            model[0]['a']=X
        elif 'activation'=='relu':
            model[l]['Z']=np.dot(model[l]['W'],model[l-1]['a'])
            model[l]['a']=model['l']['Z']
            model[l]['a'][model['l']['a']<0]=0
        elif 'activation'=='softmax':
            model[l]['Z']=np.dot(model[l]['W'],model[l-1]['a'])
            model[l]['a']=exp(model[l]['Z'])/np.sum(exp(model[l]['Z']))
        else:
            pass
    return model

def backward(model, Y, learning_rate, update):
    for l in model.keys()[::-1]:
        elif 'activation'=='relu':
            model[l]['da']=np.ones(model[l]['a'].shape)
            model[l]['da'][model[l]['a']<=0]=0
            model[l]['dZ']=np.dot(model[l]['W'].T, model[l-1]['dZ'])
            model[l]['dZ']=np.multiply(model[l]['dZ'],model[l]['da'] )
            model[l]['dW']=np.dot(model[l]['dZ'],model[l-1]['a'].T)
            model[l]['dW']-=learning_rate*model[l]['dW']
        elif 'activation'=='softmax':
            model[l]['da']=1
            model[l]['dZ']=Y-model[l]['a']
            model[l]['dW']=np.dot(model[l]['dZ'],model[l-1]['a'].T)
            model[l]['dW']-=learning_rate*model[l]['dW']
        else:
            pass
    return model







#foward
def forward(model, X, learning_rate):
    for l in model.keys():
        #X
        if model[l]['operation']=='X':
            model[l]['value']=X
        else:
            pass
        #FULLY CONNECTED
        if model[l]['operation']=='fc':
            #INITIALIZE OR UPDATE PARAMETERS
            if model[l]['parameters'] is None:
                model[l]['parameters']={'W':np.zeros([model[l]['shape'][0], model[l-1]['shape'][0]])}
                model[l]['derivatives']={'W':np.zeros([model[l]['shape'][0], model[l-1]['shape'][0]])}
            else:
                model[l]['parameters']['W']+=learning_rate*model[l]['derivatives']['W']
            #PROPAGATE
            model[l]['value']=np.dot(model[l]['parameters']['W'],model[l-1]['value'])
        #ReLU
        elif model[l]['operation']=='relu':
             model[l]['value']=model[l-1]['value']
             model[l]['value'][model[l]['value']<0]=0

        #SOFTMAX
        elif model[l]['operation']=='softmax':
             model[l]['value']=np.exp(model[l-1]['value'])/np.sum(np.exp(model[l-1]['value']))
        elif model[l]['operation']=='action':

            eps=model[l]['parameters']['exploration_prob']
            imax=np.argmax(model[l-1]['value'])
            idxs=model[l]['parameters']['action_dic'].keys()
            other_choices=idxs[0:imax]+idxs[imax+1:]
            #ACTION_INDEX
            if   np.random.uniform()>eps:
                model[l]['value']=imax
            else:
                model[l]['value']=np.random.choice(other_choices)

        else:
            pass
    return model

model=forward(model,X,0)
model[4]['value']

model.keys()[::-1]
print(a)
model.keys()


def backward(model,y):
    for l in model.keys()[::-1]:
        if model[l]['operation']=="softmax":
            model[l]['derivatives']=model[l]['values']-y
        elif model[l]['operation']=="relu":
            model[l]['derivatives']=np.ones(model[l]['values'].shape)
            model[l]['derivatives'][model[l]['values']<0]=0
        elif model[l]['operation']=="fc":
            model[l]['derivatives']['W']=np.matmul(model[l+1]['derivatives']['h'],model[l]['values'].T]








#backward
#TODOfor l in range(0, len(layeres)):
