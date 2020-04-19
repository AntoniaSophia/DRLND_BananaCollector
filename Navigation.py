#!/usr/bin/env python
# coding: utf-8

# # Navigation
# 
# ---
# 
# In this notebook, you will learn how to use the Unity ML-Agents environment for the first project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).
# 
# ### 1. Start the Environment
# 
# We begin by importing some necessary packages.  If the code cell below returns an error, please revisit the project instructions to double-check that you have installed [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and [NumPy](http://www.numpy.org/).

# In[1]:


from unityagents import UnityEnvironment
from dqn_agent import Agent
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch


# Next, we will start the environment!  **_Before running the code cell below_**, change the `file_name` parameter to match the location of the Unity environment that you downloaded.
# 
# - **Mac**: `"path/to/Banana.app"`
# - **Windows** (x86): `"path/to/Banana_Windows_x86/Banana.exe"`
# - **Windows** (x86_64): `"path/to/Banana_Windows_x86_64/Banana.exe"`
# - **Linux** (x86): `"path/to/Banana_Linux/Banana.x86"`
# - **Linux** (x86_64): `"path/to/Banana_Linux/Banana.x86_64"`
# - **Linux** (x86, headless): `"path/to/Banana_Linux_NoVis/Banana.x86"`
# - **Linux** (x86_64, headless): `"path/to/Banana_Linux_NoVis/Banana.x86_64"`
# 
# For instance, if you are using a Mac, then you downloaded `Banana.app`.  If this file is in the same folder as the notebook, then the line below should appear as follows:
# ```
# env = UnityEnvironment(file_name="Banana.app")
# ```

# In[3]:


env = UnityEnvironment(file_name="../Unity_BananaCollector/Banana.exe")


# Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# ### 2. Examine the State and Action Spaces
# 
# The simulation contains a single agent that navigates a large environment.  At each time step, it has four actions at its disposal:
# - `0` - walk forward 
# - `1` - walk backward
# - `2` - turn left
# - `3` - turn right
# 
# The state space has `37` dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  A reward of `+1` is provided for collecting a yellow banana, and a reward of `-1` is provided for collecting a blue banana. 
# 
# Run the code cell below to print some information about the environment.

# In[5]:


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)




def train_banana_collector():

    agent = Agent(state_size=state_size, action_size=action_size, seed=999)
    
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score

    eps_start=1.0
    eps_end=0.01
    eps_decay=0.999
    eps = eps_start                    # initialize epsilon

    n_episodes=10000

    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores

    for i_episode in range(1, n_episodes+1):
        score = 0
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state

        while True:
            #action = np.random.randint(action_size)        # select an action
            action = agent.act(state, eps)
            #action = action.astype(int)
            #print("Action = " , action)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            #print("Reward = " , reward)
            done = env_info.local_done[0]                  # see if episode has finished
            # 3. inform agent on environment feedback
            agent.step(state, action, reward, next_state, done)
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            if done:                                       # exit loop if episode finished
                #print("Episode " , i_episode , " has ended with score: " , score)
                break
        
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f} , epsilon: {}'.format(i_episode, np.mean(scores_window),eps), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=15 and len(scores_window)==100:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnn_local.state_dict(), 'checkpoint.pth')
            break
   
    return scores


def plot_scores(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()





def test_banana_collector(agent,runs):

    scores = []

    for i in range(runs):
        env_info = env.reset(train_mode=False)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        
        score = 0 


        try:
            while (1):
                action = agent.act(state,0)   
                env_info = env.step(action)[brain_name] 
                state = env_info.vector_observations[0] 
                reward = env_info.rewards[0]
                done = env_info.local_done[0]  
                score += reward
                if done:
                    print("Score in episodes {}: {}".format(i, score))
                    break
        except Exception as e:
            print("exception:",e)
            continue
    
        scores.append(score)

    return scores

def load_banana_collector(filename):
    agent = Agent(state_size=state_size, action_size=action_size, seed=630)
    agent.qnn_local.load_state_dict(torch.load(filename))
    return agent



#train_banana_collector()

agent = load_banana_collector('checkpoint.pth')
n_episode_run = 10
scores = test_banana_collector(agent , n_episode_run)
print("Mean score over {} episodes: {}".format(n_episode_run, np.mean(scores)))


# When finished, you can close the environment.

env.close()
