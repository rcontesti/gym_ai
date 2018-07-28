import numpy as np
import cPickle as pickle
import gym
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation



#functions
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


#LOOP---------------------------------------------------------------------------
#Initialize Game and Model

env = gym.make("Pong-v0")

env.render()

for i in range(n_iterations):
    # start new episode
    done=False
    while done=False: #(while episode last collect data)
        action=2
        observation, reward, done, info = env.step(action)
        obs.append(observation)
    if done:
env.help()







action_dic={0:3,1:0,2:2} # 3==UP, 2==DOWN , 0==NOTHING ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'] env.unwrapped.get_action_meanings()

observation = env.reset()
done=False
counter=0
obs=[]
ims=[]
while done==False & counter<20:
    counter+=1
    action=2
    observation, reward, done, info = env.step(action)
    ob=prepro(observation).reshape(80,80)
    obs.append(ob)
    im = plt.imshow(ob, animated=True)
    ims.append([im])


#plt.imshow(obs[100].reshape(80,80))
im
fig = plt.figure()
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,repeat_delay=1000)
ani.save('dynamic_images.mp4')
plt.show()
