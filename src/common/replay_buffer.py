
import collections
import numpy as np
import random

import torch

class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.set_device()

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append((np.ones(len(done)) - done).tolist())

        states = torch.tensor(np.array(s_lst), dtype=torch.float)
        actions = torch.tensor(np.array(a_lst), dtype=torch.float)
        rewards = torch.tensor(np.array(r_lst), dtype=torch.float)
        next_states = torch.tensor(np.array(s_prime_lst), dtype=torch.float)
        done_masks = torch.tensor(np.array(done_mask_lst), dtype=torch.float)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        done_masks = done_masks.to(self.device)

        return states, actions, rewards, next_states, done_masks

    def set_device(self, i=0):
        if torch.cuda.is_available():  
            dev = f'cuda:{i}' 
            print(f"Using Device: {torch.cuda.get_device_name(i)}")
        else:  
            dev = 'cpu'  
        print(f"device name: {dev}")
        device = torch.device(dev)  
        self.device = device


    def sample_chunk(self, batch_size, chunk_size):
        start_idx = np.random.randint(0, len(self.buffer) - chunk_size, batch_size)
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []

        for idx in start_idx:
            for chunk_step in range(idx, idx + chunk_size):
                s, a, r, s_prime, done = self.buffer[chunk_step]
                s_lst.append(s)
                a_lst.append(a)
                r_lst.append(r)
                s_prime_lst.append(s_prime)
                done_lst.append(done)

        n_agents, obs_size = len(s_lst[0]), len(s_lst[0][0])
        return torch.tensor(np.array(s_lst), dtype=torch.float).view(batch_size, chunk_size, n_agents, obs_size), \
               torch.tensor(np.array(a_lst), dtype=torch.float).view(batch_size, chunk_size, n_agents), \
               torch.tensor(np.array(r_lst), dtype=torch.float).view(batch_size, chunk_size, n_agents), \
               torch.tensor(np.array(s_prime_lst), dtype=torch.float).view(batch_size, chunk_size, n_agents, obs_size), \
               torch.tensor(np.array(done_lst), dtype=torch.float).view(batch_size, chunk_size, 1)

    def size(self):
        return len(self.buffer)
