from unityagents import UnityEnvironment
from dqn_agent import Agent
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch


#
# Train an agent and return the reached maximum reward
# 
def train_banana_collector(env, brain_name, maxEpisodes, threshold, \
                           eps_start, eps_end, eps_decay, seed, filename, prioritized_memory):

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    brain = env.brains[brain_name]

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

    env_info = env.reset(train_mode=True)[brain_name]
    agent = Agent(state_size=state_size, action_size=action_size, seed=seed , prioritized_memory=prioritized_memory)
    
    state = env_info.vector_observations[0]            # get the current state


    # initialize the score
    score = 0                                          
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores


    # initialize epsilon
    eps = eps_start                    

    # now execute up to maximum r"maxEpisodes" episodes
    for i_episode in range(1, maxEpisodes):
        score = 0

        # 1.Step: reset the environment - set the train_mode to True !!
        env_info = env.reset(train_mode=True)[brain_name] 
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
        if np.mean(scores_window)>=threshold and len(scores_window)==100:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnn_local.state_dict(), filename)
            break
   
    return scores

# plot the scores of all episodes
def plot_scores(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()



#
# Run a number of episodes with a given agent and calculate the average reward
# 
def test_banana_collector(env, brain_name, agent, runs):
    # set overall sum of scores to 0
    scores = []

    # now execute up to maximum runs episodes
    for i in range(runs):

        # 1.Step: reset the environment - set the train_mode to False !!
        env_info = env.reset(train_mode=False)[brain_name] 

        # 2. Step: get the current state 
        state = env_info.vector_observations[0]           
        
        # 3.Step: set the score of the current episode to 0
        score = 0 

        # 4.Step: while episode has not ended (done = True) repeat
        try:
            while (1):#
                # 5.Step: Calculate the next action from agent with epsilon 0 
                #         epsilon = 0 because we are not in training mode !
                action = agent.act(state,0)                 

                # 6.Step: Tell the environment about this action and get result
                env_info = env.step(action)[brain_name] 

                # 7.Step: now let's get the state observation from observation
                state = env_info.vector_observations[0] 

                # 8.Step: now let's get the reward observation from observation
                reward = env_info.rewards[0]

                # 9.Step: now let's get the done observation from observation
                done = env_info.local_done[0]

                # 10.Step: Add the reward of the last action-state result  
                score += reward

                # 11.Step: in case of end of episode print the result and break loop 
                if done:
                    print("Score in episodes {}: {}".format(i, score))
                    break
        except Exception as e:
            print("exception:",e)
            continue
    
        # 12.Step: Finally append the score of last epsisode to the overall scores
        scores.append(score)

    return scores

#
# Reload a trained DQN from file 'filename'
# 
def load_banana_collector(env, brain_name , filename):

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    brain = env.brains[brain_name]

    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)

    agent = Agent(state_size=state_size, action_size=action_size, seed=630, prioritized_memory=False)
    agent.qnn_local.load_state_dict(torch.load(filename))
    return agent



# ### 2. Examine the State and Action Spaces
# 
# The simulation contains a single agent that navigates a large environment.  
# At each time step, it has four actions at its disposal:
# - `0` - walk forward 
# - `1` - walk backward
# - `2` - turn left
# - `3` - turn right
# 
# The state space has `37` dimensions and contains the agent's velocity, 
# along with ray-based perception of objects around agent's forward direction.  
# A reward of `+1` is provided for collecting a yellow banana, and a reward 
# of `-1` is provided for collecting a blue banana. 
env = UnityEnvironment(file_name="../Unity_BananaCollector/Banana.exe")

# Environments contain **_brains_** which are responsible for deciding 
# the actions of their associated agents. Here we check for the first brain 
# available, and set it as the default brain we will be controlling from Python.

# get the default brain
brain_name = env.brain_names[0]


# Set the minimum score the agent has to reach in order to solve this task
threshold = 16.0

# Set the maximum number of episodes which the agent 
maxEpisodes = 3000

# Set this variable to "True" in case you want to retrain your agent
train = False

# Set the hyperparameters for training
eps_start=1.0
eps_end=0.02
eps_decay=0.995
seed = 999

# Set the filename for storage of the trained model
filename = "checkpoint.pth"

# Set the following parameter to True in case you would like to use the prioritized memory
prioritized_memory = True

if train:
    train_banana_collector(env, brain_name, maxEpisodes, threshold, \
                           eps_start, eps_end, eps_decay, seed, filename, prioritized_memory)


# Finally test the agent 

agent = load_banana_collector(env, brain_name, filename) # First load the trained agent
n_episode_run = 10                      # Execute 10 runs
scores = test_banana_collector(env, brain_name, agent , n_episode_run)  # run!
print("Mean score over {} episodes: {}".format(n_episode_run, np.mean(scores)))


# When finished, you can close the environment.
env.close()
