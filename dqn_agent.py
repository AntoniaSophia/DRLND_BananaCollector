import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim


from PrioritizedMemory import PrioritizedReplayMemory

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 32

GAMMA = 0.99 # discount rate
TAU = 1e-3 # soft updating of target params
LR = 5e-4 # learning rate
UPDATE_EVERY = 4 # how often to update the network (otherwise we'll have the endless and blind stick-carrot problem)

# CPU OR GPU 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """ Smart agent to interact and learn from the environment
    
    Function:
        1. act: return actions given current policy (network configuration)
        2. step: inform agent that the step has been taken in the environment
        3. learn: every xxx steps, the step function will learn based on previous experiences
        4. soft_update: update target neural network
    
    """
    
    def __init__(self, state_size, action_size, seed, prioritized_memory):
        
        # keep params
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # Create local and target network
        self.qnn_local = QNetwork(state_size, action_size, seed, fc1_units=74, fc2_units=74).to(device)
        self.qnn_target = QNetwork(state_size, action_size, seed, fc1_units=74, fc2_units=74).to(device)
        
        # Init optimizer
        self.optimizer = optim.Adam(self.qnn_local.parameters(), lr=LR)
        
        # Replay Memory
        if prioritized_memory == True:
            print("Using PrioritizedReplayMemory...")
            self.memory = PrioritizedReplayMemory(BUFFER_SIZE)
        else:
            print("Using simple ReplayBuffer...")
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        self.t_step = 0
        
    def act(self, state, eps=0.):
        """ Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        self.qnn_local.eval() # put network into evaluation mode
        with torch.no_grad():
            action_values = self.qnn_local(state)
            
        self.qnn_local.train() # put network into train model

        # Epsilon-greedy action selection
        if random.random() > eps:
            return int(np.argmax(action_values.cpu().data.numpy()))
        else:
            return int(random.choice(np.arange(self.action_size)))
        
    def step(self, state, action, reward, next_state, done):
        """ inform agent that a step has been taken """
        
        # save experience (step) in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            
            # learn if we have enough experience samples in memory
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(BATCH_SIZE)
                self.learn(experiences, GAMMA)
    
    def learn(self, experiences, gamma):
        """ Update value params using given batch of exp tuples
        
            experiences: tuple containing (s, a, r, s', done) tuples (Tuple[torch.Tensor])
            gamme: discount factor (float)
        """
        
        # unpack experiences into workable arrays
        states, actions, rewards, next_states, dones = experiences

        #print("States = ", states.shape , " type = " , type(states))
        #print("actions = ", actions.shape , " type = " , type(actions))
        #print("rewards = ", rewards.shape , " type = " , type(rewards))
        #print("next_states = ", next_states.shape , " type = " , type(next_states))
        #print("dones = ", dones.shape , " type = " , type(dones))
        
        # get best predicted Q-values for next_states from targe model [What does the target model tell us (delayed trained model)]
        Q_targets_next = self.qnn_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones)) # compute Q targets for current states
        
        # get expected Q-values from local model [What does the local model tell us?]
        Q_expected = self.qnn_local(states).gather(1, actions)
        
        # compute loss [HOW MUCH OFF?]
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # minimize loss [TRAIN]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        ### update target network [UPDATE HAPPENS HERE]
        self.soft_update(self.qnn_local, self.qnn_target, TAU)
        
    def soft_update(self, local_model, target_model, tau):
        """ Soft update model parameters
        
            θ_target = τ*θ_local + (1 - τ)*θ_target
            
            local_model: weights will be copied from (pytorch model)
            target_model: weights will be copied to (pytorch model)
            tau: interpolation parameter (float)
        """
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
        
class ReplayBuffer:
    """ Fixed-size buffer to store experience tuples """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """ Initialize a ReplayBuffer object
        Params
        ======
            action_size: dimension of each action
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
            seed: random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self , batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
