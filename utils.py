import numpy as np
import cPickle as pickle
import gym

import IPython
from PIL import Image, ImageFilter
env = gym.make("Pong-v0")
observation = env.reset()

'''
print(observation)
type(observation)
observation.shape
action=88
'''

'''
print(env.action_space.sample())
print(env.observation_space)
print(env.observation_space.high)
observation, reward, done, info = env.step(action)
'''

action = 2  # modify this!
o = env.reset()
for i in xrange(100): # repeat one action for five times
    o = env.step(action)[0]
    '''
    IPython.display.display(Image.fromarray(o[:,140:142]  # extract your bat
    ).resize((300, 300)))  # bigger image, easy for visualization
    '''
    IPython.display.display(Image.fromarray(o))
