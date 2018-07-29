import numpy as np
import cPickle as pickle
import gym
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation


#Parameters---------------------------------------------------------------------
h0=6400
h1=100
h2=3#classes
learning_rate=1e-0
update=1
reg=1e-3
exploration_eps=0.05
n_iterations=100
#functions----------------------------------------------------------------------
def prepro(obs):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  """ to get image back obs.reshape(80,80)
      to get the right of the image obs[]
  """
  obs = obs[35:195] # crop
  obs = obs[::2,::2,0] # downsample by factor of 2 and just keep the first[0] of the 3rd dim
  obs[obs == 144] = 0 # erase background (background type 1)
  obs[obs == 109] = 0 # erase background (background type 2)
  obs[obs != 0] = 1 # everything else (paddles, ball) just set to 1
  return obs.astype(np.float).ravel() # convert to float and stack in 1-D array with ravel
def get_x(obs,prev_obs):
    return obs-pre_obs if prev_obs is not None else np.zeros(obs.shape)
def get_action(model, exploration_eps):
    '''
    With probability 1-epsilon pick index of maximun value
    else explore sample index from the remaining of all indexes after removing
    the index of the maximun value
    env.unwrapped.get_action_meanings()
    ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
    '''
    action_dic={0:3,1:0,2:2} # 3==UP , 0==NOTHING , 2==DOWN


    imax=np.argmax(model[2]['a'])
    idxs=action_dic.keys()
    other_choices=idxs[0:imax]+idxs[imax+1:]


    if np.random.uniform()>exploration_eps:
        action_index=imax
    else:
        action_index=np.random.choice(other_choices)

    action=action_dic[action_index]

    return action, action_index
#Model--------------------------------------------------------------------------
'''
I move forward in every observation because that is how I get an action: m_f=1
I move backward only after I get a reward, this number is not fix: m_b=?
I update parameters only after e episodes so m_u= E_{e}(m_b0)
'''
#TODO: Build flexible architecture both for size of X as well forward and backward
def z(f,c): return np.zeros((f,c))
def o(f,c): return np.ones((f,c))
def r(f,c): return 0.01*np.random.randn(f,c)

model0={
0:{'activation':'X'},
1:{'activation':'relu',    'W':r(h1,h0), 'dW':z(h1,h0),'b':z(h1,1), 'db':z(h1,1)},
2:{'activation':'softmax', 'W':r(h2,h1), 'dW':z(h2,h1),'b':z(h2,1), 'db':z(h1,1)}
}

model={1:{'h':100,'activation': 'relu'},2:{'h':3, 'activation':'softmax'}}
params={1:{'W':None,'b':None},2:{'W':None,'b':None}}

model[]

def forward(X,model,params)
    (h0,m)=X.shape

    for l in model.keys():
        if model[l]['activation']=='relu':
            model[l]['Z']=np.dot(params[l]['W'],model[l-1]['a'])+model[l]['b']
            model[l]['a']=model[l]['Z']
            model[l]['a'][model[l]['a']<0]=0
        elif model[l]['activation']=='softmax':
                model[l]['Z']=np.dot(params[l]['W'],model[l-1]['a'])+model[l]['b']
                model[l]['a']=np.exp(model[l]['Z'])/np.sum(np.exp(model[l]['Z']), axis=0, keepdims=True)
                #print(np.sum(model[l]['a']))
        else:
            pass
        return model
def stack_forwards():
    pass
def backward():
    pass



#LOOP---------------------------------------------------------------------------
#Initialize Game and Model

env = gym.make("Pong-v0")
observation = env.reset()
env.render()
n_iterations=100
obs=prepro(observation)
X=get_x(obs,prev_obs=None)
X.shape
env.close()

for i in range(n_iterations):
    # start new episode
    done=False
    while done==False: #(while episode last collect data)
        #action=2
        action=env.action_space.sample()
        observation, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.01)
        #obs.append(observation)
    if done:
        print("episode finished")
        env.close()
        break

#-------------------------------------------------------------------------------
