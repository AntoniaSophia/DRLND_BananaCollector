[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

## Description of my solution

In this chapter I describe my solution to this project. My solutions consists of the following files:
   - `Navigation.ipynb` - this is the Jupyter notebook
   - `dqn_agent.py` - this is a vanilla DQN agent and contains the following parameters
     - state_size           
     - action_size
     - seed
     - prioritized_memory
   - `model.py` - the neural network of the DQN
   - `DequeMemory.py`  - an implementation of the replay memory based on deque data structure
   - `FifoMemory.py`  - an alternative implementation of the replay memory based on a simplest first-in-first-out (FIFO) eviction scheme
   - `Navigation.py` - this basically has the same content as the Jupyter notebook - it was my 


## The neural network
The DQN performs the mapping between input states and predicted optimal actions. The state space is 1 dimensional and has 37 states, thus using a fully connected layer is suffienct and no convolutional layers are required.
Hence the input layer receives 37 states and the output layer has size 4:
- **`0`** - move forward
- **`1`** - move backward
- **`2`** - turn left
- **`3`** - turn right

I simply used the model which was also used in the lesson 2 (DQN) - basically it is identical.

I played around with the dimensions of the hidden layers, but it seems that using `size(fc1)=Size(fc2) = 64` is the optimal setting.
All other variants - either using less or increasing - finish the task only with more episodes.


## Used parameters



# Set this variable to "True" in case you want to retrain your agent


### DQN Agent

  - `eps_start=1.0` - starting with maximum exploration (when using epsilon-greedy policy) is always a good advise....
  - `eps_end=0.02` - this lower limit of epsilon I found during try&error approach
  - `eps_decay=0.995` - I took this parameter from the paper XYZ
  - `seed = 999` - the seed for the randomizer (I kept it constant in order to be able to compare results)

  - `BUFFER_SIZE = int(1e5)`
  - `BATCH_SIZE = 32` - 

  - `GAMMA = 0.99` # discount rate
  - `TAU = 1e-3` # soft updating of target params
  - `LR = 5e-4` # learning rate
  - `UPDATE_EVERY = 4` # how often to update the network (remember the stick-carrot problem !!)


### Other parameters
  - `maxEpisodes = 3000` - Set the maximum number of episodes which the agent 
  - `threshold = 16.0` - the value to be reached by the agent in order to be successful. Actually 13.0 was required, but I tried to push the limit....
  - `filename = "checkpoint.pth"` - the filename for storage of the trained model
  - `train = True` - this parameter should be set to True in case you would like to train, otherwise set this parameter to False
  - `memory_type = 0` - this parameters 


## Discussion

Finally I was able to solve this task for the threshold = 16.0
  - within 702 episodes for FifoMemory 
  - within ??? episodes for DequeMemory

Attached are the reward score plots 

a

b

And 


Setting BATCH_SIZE = 32 worked best as I found out during several attempts simply by try & error.

FifoMemory works much better than DequeMemory

During the consecutive test of 100 episodes it sometimes happens that the score in this episode is much less than 10.
Reason seems to be that the agent is "toggling" between left and right action. At least this is what I have observed watching the agent.
Definitely this could be improved
toggling

Improvements:
- Dueling DQN
- Prioritized Memory
