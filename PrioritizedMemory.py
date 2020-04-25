import random
import torch
import numpy as np
from collections import namedtuple, deque 

# Taken from https://github.com/hengyuan-hu/rainbow/blob/master/core.py

# CPU OR GPU 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Sample(object):
    def __init__(self, state, action, reward, next_state, done):
        self._state = state
        self._next_state = next_state
        self.action = action
        self.reward = reward
        self.done = done

    @property
    def state(self):
        return self._state

    @property
    def next_state(self):
        return self._next_state

    def __repr__(self):
        info = ('S(mean): %3.4f, A: %s, R: %s, NS(mean): %3.4f, Done: %s'
                % (self.state.mean(), self.action, self.reward,
                   self.next_state.mean(), self.done))
        return info


class PrioritizedReplayMemory(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.samples = []

        self.max_best_samples = 1000
        self.best_reward = -1000
        self.best_samples = deque(maxlen=self.max_best_samples)
        self.best_sample_rate = 0.3
        
        self.oldest_idx = 0

        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        self.time_period = 60
        self.history_reward = 0
        self.temporal_history = deque(maxlen=self.time_period)

        self.sample_counter = 0

    def __len__(self):
        return len(self.samples)

    def _evict(self):
        """Simplest FIFO eviction scheme."""
        to_evict = self.oldest_idx
        self.oldest_idx = (self.oldest_idx + 1) % self.max_size
        return to_evict

    def add(self, state, action, reward, next_state, end):
        assert len(self.samples) <= self.max_size

        # 0.Step: Increase the sample counter
        self.sample_counter += 1

        if (self.sample_counter % 100000 == 0):
            self.temporal_history.clear()
            self.history_reward = 0
            self.best_samples.clear()            

        # 1.Step: add to complete memory
        new_sample = Sample(state, action, reward, next_state, end)
        if len(self.samples) == self.max_size:
            avail_slot = self._evict()
            self.samples[avail_slot] = new_sample
        else:
            self.samples.append(new_sample)

        # 2.Step: add to "best samples" memory, consider history of 
        #         length self.temporal_history
        
        # 3.Step: in case this temporal history experiences a negative reward
        #          then drop all and reset the history 
        if reward < 0:
            self.temporal_history.clear()
            self.history_reward = 0

        # 4.Step: ok, we don't have any negative reward..... 
        #          have we reached history length? 
        if (len(self.temporal_history) < self.time_period):
            # 5.Step: in case history length not reached, append this sample
            #         and calculate the cumulative reward
            self.history_reward += reward
            self.temporal_history.append(new_sample)
        else:
            # 6.Step: ok, now we are close to the maximum history reward, at least 90%
            #         in that case consider this history
            if self.history_reward >= self.best_reward * 0.9:
                # 7.Step: in case of a new best reward - nail it and clear all past best samples
                if self.history_reward > self.best_reward:
                    print("New history reward: " , self.history_reward)
                    self.best_samples.clear()  
                    self.best_reward = self.history_reward

                # 8.Step: Ensure we have enough space in the best sample buffer - otherwise remove existing elements
                if (len(self.best_samples) > self.max_best_samples - self.time_period):
                    for i in range(self.time_period): self.best_samples.popleft()
    
                # 9.Step: finally add the history 
                self.best_samples += self.temporal_history

            # 10.Step: in any case: delete the history once the maximum length was reached
            self.temporal_history.clear()
            self.history_reward = 0

    def sample(self, batch_size):
        """Simpliest uniform sampling (w/o replacement) to produce a batch.
        """
        assert batch_size < len(self.samples), 'no enough samples to sample from'

        # In case enough "best samples" are available also use them at sampling
        # of the memory data after 100 episodes
        if (len(self.best_samples) > self.max_best_samples * 0.9 and self.sample_counter > 300 * 200):
            complete_sample_memory = int(batch_size * self.best_sample_rate)
            best_sample_memory = batch_size - complete_sample_memory
            #print("complete_sample_memory = " , complete_sample_memory)
            #print("best_sample_memory = " , best_sample_memory)            
            experiences1 = random.sample(self.samples, complete_sample_memory)
            experiences2 = random.sample(self.best_samples, best_sample_memory)
            experiences = experiences1 + experiences2
        else: 
            # Else just use complete memory samples
            experiences = random.sample(self.samples, batch_size)

        # convert experience tuples to arrays
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def clear(self):
        self.samples = []
        self.oldest_idx = 0
