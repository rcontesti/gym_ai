import numpy as np
import cPickle as pickle
import os.path
import os
import gym
import time

#TODO: Run 1000000
#TODO: Try Conv+ Aurelian method

#functions----------------------------------------------------------------------
'''
I move forward in every observation because that is how I get an action: m_f=1
I move backward only after I get a reward, this number is not fix: m_b=?
I update parameters only after e episodes so m_u= E_{e}(m_b0)
'''
def z(f,c): return np.zeros((f,c))
def o(f,c): return np.ones((f,c))
def r(f,c): return np.random.randn(f,c)/np.sqrt(c) #Xavier initialization
def prepro_orig(obs):
  """ Original """
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  """ to get image back obs.reshape(80,80)
      to get the right of the image obs[]
  """
  obs = obs[35:195] # crop
  obs = obs[::2,::2,0] # downsample by factor of 2 and just keep the first[0] of the 3rd dim
  obs[obs == 144] = 0 # erase background (background type 1)
  obs[obs == 109] = 0 # erase background (background type 2)
  obs[obs != 0] = 1 # everything else (paddles, ball) just set to 1
  obs=obs.astype(np.float).ravel()# convert to float and stack in 1-D array with ravel
  return  obs.reshape(obs.shape[0],1)
def prepro(obs):
    """ Reduced """
    """ prepro 210x160x3 uint8 frame into 6400 (53x23) 2D array: (53x23,1)"""
    obs_my_paddle= obs[35:194,-20:-15]
    obs_left_side= obs[35:194,16:80]
    obs=np.concatenate((obs_left_side, obs_my_paddle), axis=1)
    obs = obs[::3,::3,0]# downsample by a factor of 3
    obs[obs == 144] = 0
    obs[obs == 109] = 0 # erase background (background type 2)
    obs[obs != 0] = 1
    obs=obs.astype(np.float).ravel()# convert to float and stack in 1-D array with ravel
    return  obs.reshape(obs.shape[0],1)
def get_x(obs,prev_obs):
    return obs-prev_obs if prev_obs is not None else np.zeros(obs.shape)
def initialize_action_dic():
    '''
    env.unwrapped.get_action_meanings()
    ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
    # 3==UP , 0==NOTHING , 2==DOWN
    '''
    return {0:3,1:0,2:2}
def get_action(flayers, action_dic,exploration_eps):
    '''

    Once the model starts performing better the max probability of the softmax
    is going to increase. At the very begging that would be one third

    Another posibility of the have the exploration parameter fixed
    if np.random.uniform()>exploration_eps
    With probability 1-epsilon pick index of maximun value
    else explore sample index from the remaining of all indexes after removing
    the index of the maximun value

    '''
    L=flayers.keys()[-1]
    imax=np.argmax(flayers[L]['a'])
    idxs=action_dic.keys()
    other_choices=idxs[0:imax]+idxs[imax+1:]

    if flayers[L]['a'][imax]>np.random.uniform():
        action_index=imax
    else:
        action_index=np.random.choice(other_choices)

    action=action_dic[action_index]

    return action, action_index
def get_advantage(rewards, gamma):
    rewards=np.cumsum(rewards)
    discounted_r = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(xrange(0, rewards.size)):
      if rewards[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
      running_add = running_add * gamma + rewards[t]
      discounted_r[t] = running_add

    #Normalize
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)
    return discounted_r
def get_y(action_dic, action_indexes):
    "get one hot representation of the action_index"
    y=z(len(action_dic),len(action_indexes))
    y[action_indexes,range(y.shape[1])]+=1
    return y
#Model--------------------------------------------------------------------------
def initialize_flayers():
    return {0: {'z': None, 'a': None, 'dz': None, 'da': None, 'activation': 'X'},
            1: {'z': None, 'a': None, 'dz': None, 'da': None, 'activation': 'relu'},
            2: {'z': None, 'a': None, 'dz': None, 'da': None, 'activation': 'softmax'}}
def initialize_blayers():
    return {0: {'z': [], 'a': [], 'dz': [], 'da': [], 'activation': 'X'},
            1: {'z': [], 'a': [], 'dz': [], 'da': [], 'activation': 'relu'},
            2: {'z': [], 'a': [], 'dz': [], 'da': [], 'activation': 'softmax'}}
def initialize_model():
        return {1:{'W':r(h1,h0),'b':z(h1,1)},
                2:{'W':r(h2,h1),'b':z(h2,1)}}
def initialize_grads():
    return {1:{'W':z(h1,h0),'b':z(h1,1)},
            2:{'W':z(h2,h1),'b':z(h2,1)}}
def initialize_grads_buffer():
    return {1:{'W':z(h1,h0),'b':z(h1,1)},
            2:{'W':z(h2,h1),'b':z(h2,1)}}
def initialize_rmsprop_cache():
    return {1:{'W':z(h1,h0),'b':z(h1,1)},
            2:{'W':z(h2,h1),'b':z(h2,1)}}
def forward(X,layers,model):
    for l in layers.keys():
        if   layers[l]['activation']=='X':
             layers[l]['a']=X
        elif layers[l]['activation']=='relu':
             layers[l]['z']=np.dot(model[l]['W'],layers[l-1]['a'])+model[l]['b']
             layers[l]['a']=layers[l]['z']
             layers[l]['a'][layers[l]['a']<0]=0
        elif layers[l]['activation']=='softmax':
             layers[l]['z']=np.dot(model[l]['W'],layers[l-1]['a'])+model[l]['b']
             layers[l]['a']=np.exp(layers[l]['z'])/np.sum(np.exp(layers[l]['z']), axis=0, keepdims=True)
        else:
            print('Error')
    return layers
def append_flayers(flayers, blayers):
    for l in flayers.keys():
        for i  in ['z','a','dz','da']:
            blayers[l][i].append(flayers[l][i])
    return blayers
def backward(blayers,y, advantage, model, grads):
    # First transform lists appended to numpy arrays
    for l in blayers.keys():
        for i  in ['z','a','dz','da']:
            if type(blayers[l][i])==list:
                blayers[l][i]=np.hstack(blayers[l][i])
    # Second Backpropagate incorporating the negative of the advantage:
    for l in reversed(blayers.keys()):
        if blayers[l]['activation']=='softmax':
           blayers[l]['dz']=(blayers[l]['a']-y)/y.shape[1]
           blayers[l]['dz']=np.multiply(blayers[l]['dz'], -advantage) #advantage adjustment
           grads[l]['W'] =np.dot(blayers[l]['dz'],blayers[l-1]['a'].T)
           blayers[l-1]['da']=np.dot(model[l]['W'].T,blayers[l]['dz'])
        elif blayers[l]['activation']=='relu':
           blayers[l]['dz']=blayers[l]['da']
           blayers[l]['dz'][ blayers[l]['a']<=0]=0
           grads[l]['W'] =np.dot(blayers[l]['dz'],blayers[l-1]['a'].T)
        else:
            pass
        return blayers, grads
def rmsprop_update(model, grads, grads_buffer, episode_number, batch_size):
    for l in model.keys(): grads_buffer[l]['W']+=grads[l]['W']
    if episode_number % batch_size==0:
        for l in model.keys():
            rmsprop_cache[l]['W']=decay_rate*rmsprop_cache[l]['W']+(1-decay_rate)*np.power(grads_buffer[l]['W'],2)
            model[l]['W']-=learning_rate*grads_buffer[l]['W']/(np.sqrt(rmsprop_cache[l]['W'])+1e-5)
            grads_buffer=initialize_grads_buffer()#reset_grads
    return model, grads_buffer

#Parameters---------------------------------------------------------------------
h0=1219
h1=100
h2=3#classes
gamma=.9
learning_rate=1e-0
batch_size=10
decay_rate=.9
update=1
reg=1e-3
exploration_eps=0.01
n_episodes=100001
render=1
load_last=1
#LOOP---------------------------------------------------------------------------
#Initialize Game and Model
#initialize env
env = gym.make("Pong-v0")
env.reward_range
observation = env.reset()
action_dic=initialize_action_dic()


#load_model
if load_last==1:
    models=[f for f in os.listdir(os.getcwd()) if (f.endswith('.p') and  f.startswith('model'))]
    models_n=[int(m.split('.')[0].split('_')[-1]) for m in models if (m.split('.')[0].split('_')[-1]).isdigit()]
    cum_episodes=max(models_n)
    last_model='model_'+str(cum_episodes)+'.p'
    model = pickle.load(open(last_model, 'rb'))
else:
    model=initialize_model()


#Initialize backprop dics

grads=initialize_grads()
grads_buffer=initialize_grads_buffer()
rmsprop_cache=initialize_rmsprop_cache()

#loop
episode_number=0
for i in range(n_episodes):
    # start new episode
    observation = env.reset()
    flayers=initialize_flayers()
    blayers=initialize_blayers()
    counter=0
    done=False
    actions=[]
    rewards=[]
    action_indexes=[]
    max_prob=[]# I save the certainty of the model
    prev_obs=None
    while done==False: #(while episode last collect data)
        counter+=1
        obs=prepro(observation)
        X=get_x(obs,prev_obs)
        prev_obs=obs
        flayers=forward(X,flayers,model)
        blayers=append_flayers(flayers, blayers)
        #Get action and Action Index
        action,action_index=get_action(flayers, action_dic,exploration_eps)
        #action=env.action_space.sample()
        #Take Action and receive state and reward
        observation, reward, done, info = env.step(action)
        action_indexes.append(action_index)
        rewards.append(reward)
        max_prob.append(flayers[2]['a'][action_index])
        if render==1:
            env.render()
            time.sleep(0.005)

    if done:
        if episode_number % 10 == 0:print("episode finished:{}, max_prob:{}, total_training_episodes : {}".format(episode_number,max_prob[-1], cum_episodes+episode_number))

        advantage=get_advantage(rewards, gamma)
        y=get_y(action_dic,action_indexes)
        blayers, grads=backward(blayers,y, advantage, model, grads)
        #params=update_params(params)
        model,grads_buffer=rmsprop_update(model, grads, grads_buffer, episode_number, batch_size)
        if episode_number % 1000 == 0: pickle.dump(model, open('model_'+str(cum_episodes+episode_number)+'.p', 'wb'))
        episode_number+=1
        env.close()

#-------------------------------------------------------------------------------
