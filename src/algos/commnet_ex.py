
import torch
import os

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import copy
import random

import numpy as np

class CommNet(nn.Module):
    '''
    Implements CommNet for a single building
    Of the CityLearn challenge
    LSTM version with skip connection for the final layer

    TODO: Try basic version without LSTM / alter skip connections etc
            But might be a better idea to explore more advanced architectures instead
    '''

    def __init__(
                self, 
                agent_number,       # Number of buildings present
                input_size,         # Observation accessible to each building (assuming homogenous)
                hidden_size = 10,   # Hidden vector accessible at each communication step
                comm_size = 4,      # Number of communication channels
                comm_steps = 2      # Number of communication steps
                ):
                
        super(CommNet, self).__init__()

        self.device = 'cpu'
        self.input_size = input_size
        self.comm_size = comm_size
        self.agent_number = agent_number
        self.comm_steps = comm_steps

        # Calculate first hidden layer 
        print(f"input size: {input_size}")
        self._in_mlp = nn.Sequential(
            nn.Linear(input_size,input_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size,input_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size,hidden_size)
        )

        # Communication 
        self._lstm = nn.LSTMCell(
            input_size = comm_size,
            hidden_size = hidden_size
        )

        self._comm_mlp = nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size,comm_size)
        )

        # Output
        # Calculate based on inputs and final memory
        self._out_mlp = nn.Sequential(
            nn.Linear(input_size+hidden_size, input_size+hidden_size),
            nn.LeakyReLU(),
            nn.Linear(input_size+hidden_size, input_size+hidden_size),
            nn.LeakyReLU(),
            nn.Linear(input_size+hidden_size, 1),
            nn.Tanh()
        )


    def forward(self,x : torch.Tensor, batch = False):

        out = None
        if not batch:

            # (Building, Observations)
            
            # Initial hidden states
            print(f"x is: {x.shape}")
            hidden_states = self._in_mlp(x)
            cell_states = torch.zeros(hidden_states.shape,device=self.device)

            # Communication
            for t in range(self.comm_steps):
                # Calculate communication vectors
                comm = self._comm_mlp(hidden_states)
                total_comm = torch.sum(comm,0)
                comm = (total_comm - comm) / (self.agent_number-1)
                # Apply LSTM   
                hidden_states, cell_states = self._lstm(comm,(hidden_states,cell_states))
            
            out = self._out_mlp(torch.cat((x,hidden_states),dim=1))
        else:
            # (Batch, Building, Observation)
            out = torch.stack([self.forward(a) for a in x])

        return out

    def to(self,device):
        super().to(device)
        self.device = device

class SingleCritic(nn.Module):

    def __init__(self,
                input_size, 
                action_size = 1,
                hidden_layer_size = 32):
        super(SingleCritic, self).__init__()

        self.input_size = input_size
        self.action_size = action_size

        self._in_mlp = nn.Sequential(
            nn.Linear(input_size + action_size, hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, 1),
        )

    def forward (self, state, action):
        x = torch.cat((torch.flatten(state,start_dim=1),torch.flatten(action,start_dim=1)),dim=1)
        return self._in_mlp(x)

from sklearn.preprocessing import MinMaxScaler

class MinMaxNormalizer:

    def __init__(self, obs_space, act_space):
        observation_space = obs_space
        low, high = observation_space[0].low, observation_space[0].high
        
        self.scalar = MinMaxScaler()
        self.scalar.fit([low,high])

    def transform(self, x):
        return self.scalar.transform(x)


# Experience replay needs a memory - this is it!
# Double stack implementation of a queue - https://stackoverflow.com/questions/69192/how-to-implement-a-queue-using-two-stacks
class Queue: 
    a = []
    b = []
    
    def enqueue(self, x):
        self.a.append(x)
    
    def dequeue(self):
        if len(self.b) == 0:
            while len(self.a) > 0:
                self.b.append(self.a.pop())
        if len(self.b):
            return self.b.pop()

    def __len__(self):
        return len(self.a) + len(self.b)

    def __getitem__(self, i):
        if i >= self.__len__():
            raise IndexError
        if i < len(self.b):
            return self.b[-i-1]
        else:
            return self.a[i-len(self.b)]

class DDPG:
    MEMORY_SIZE = 10000
    BATCH_SIZE = 128
    GAMMA = 0.95
    LR = 3e-4
    TAU = 0.001

    memory = Queue()

    def to(self,device):
        self.device = device
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)


    def __init__(self, observation_space, action_space, recurrent=False):

            
        agent_num = len(observation_space)
        action_n = action_space[0].n
        obs_len = observation_space[0].shape[0]

        # Initalize actor networks
        self.actor = CommNet(
            agent_number=agent_num,
            input_size=obs_len
        )

        self.actor_target = copy.deepcopy(self.actor)

        # Initialize critic networks
        self.critic = SingleCritic(
            input_size=obs_len*agent_num,
            action_size=action_n
        )

        self.critic_target = copy.deepcopy(self.critic)

        self.normalizer = MinMaxNormalizer(observation_space, action_space)

        self.c_criterion = nn.MSELoss()
        self.a_optimize = optim.Adam(self.actor.parameters(),lr=self.LR)
        self.c_optimize = optim.Adam(self.critic.parameters(),lr=self.LR)

        self.to("cpu")
        
    def compute_action(self, obs, exploration=True, exploration_factor = 0.3):
        #obs = self.normalizer.transform(obs)
        action = self.actor(torch.tensor(obs,device=self.device).float()).detach().cpu().numpy()
        # Adding some exploration noise
        if exploration:
            action = action + np.random.normal(scale=exploration_factor,size = action.shape)
            action = np.clip(action,a_min=-1.0,a_max=1.0)
        return action

    def add_memory(self, s, a, r, ns):
        s = self.normalizer.transform(s)
        ns = self.normalizer.transform(ns)
        self.memory.enqueue([s,a,r,ns])
        if len(self.memory) > self.MEMORY_SIZE:
            self.memory.dequeue()

    def clear_memory(self):
        self.memory.a = []
        self.memory.b = []

    # Conduct an update step to the policy
    def update(self):
        torch.set_grad_enabled(True)

        N = self.BATCH_SIZE
        if len(self.memory) < 1: # Watch before learn
            return 
        # Get a minibatch of experiences
        # mb = random.sample(self.memory, min(len(self.memory),N)) # This is slow with a large memory size
        mb = []
        for _ in range(min(len(self.memory),N)):
            mb.append(self.memory[random.randint(0,len(self.memory)-1)])

        s = torch.tensor(np.array([x[0] for x in mb]),device=self.device).float()
        a = torch.tensor(np.array([x[1] for x in mb]),device=self.device).float()
        r = torch.tensor(np.array([x[2] for x in mb]),device=self.device).float()
        ns = torch.tensor(np.array([x[3] for x in mb]),device=self.device).float()

        # Critic update
        self.c_optimize.zero_grad()
        nsa = self.actor_target.forward(ns,batch=True)
        y_t = torch.add(torch.unsqueeze(r,1), self.GAMMA * self.critic_target(ns,nsa))
        y_c = self.critic(s,a) 
        c_loss = self.c_criterion(y_c,y_t)
        critic_loss = c_loss.item()
        c_loss.backward()
        self.c_optimize.step()
# Actor update
        self.a_optimize.zero_grad()
        a_loss = -self.critic(s,self.actor.forward(s,batch=True)).mean() # Maximize gradient direction increasing objective function
        a_loss.backward()
        self.a_optimize.step()

        # Target networks
        for ct_p, c_p in zip(self.critic_target.parameters(), self.critic.parameters()):
            ct_p.data = ct_p.data * (1.0-self.TAU) + c_p.data * self.TAU

        for at_p, a_p in zip(self.actor_target.parameters(), self.actor.parameters()):
            at_p.data = at_p.data * (1.0-self.TAU) + a_p.data * self.TAU

        torch.set_grad_enabled(False)

        return critic_loss
