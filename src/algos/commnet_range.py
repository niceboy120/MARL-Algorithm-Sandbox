

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.replay_buffer import ReplayBuffer 
import numpy as np

from common.learner import MALearner

from torchview import draw_graph
import math

default_options = {
    'lr': 0.001,
    'k': 1, # number of communication steps
    'batch_size': 32,
    'gamma': 0.99,
    'buffer_limit': 50000,
    'log_interval': 20,
    'max_episodes': 30000,
    'max_epsilon': 0.9,
    'min_epsilon': 0.1,
    'test_episodes': 5,
    'warm_up_steps': 2000,
    'update_iter': 10,
    'update_target_interval': 20,
    'monitor': False
}

class COMMNET_RANGE(MALearner):
    def __init__(self, observation_space, action_space, env_name, neighborhood, options=default_options):
        super().__init__(observation_space, action_space, options)

        # instantiate the prediction (q) and target (q_target) networks
        self.q = QNet(observation_space, action_space, env_name, neighborhood)
        self.q_target = QNet(observation_space, action_space, env_name, neighborhood)
        self.q_target.load_state_dict(self.q.state_dict())

        print("\nInstantiating Prediction Network:")
        self.display_info(self.q)

        # tell the optimizer to update the network parameters at the lr
        self.optimizer = optim.Adam(self.q.parameters(), lr=self.lr)

    def init_hidden(self):
        pass

    def sample_action(self, obs, epsilon):
        return self.q.sample_action(obs, epsilon)

    def per_step_update(self, state, action, reward, next_state, done):
        self.memory.put((state, action, (np.array(reward)).tolist(), next_state, [int(all(done))]))

    def per_episode_update(self, episode_i):
        if self.memory.size() > self.warm_up_steps:
            self.train(self.gamma, self.batch_size, self.update_iter)

        # at every update interval update the target q_network
        if episode_i % self.update_target_interval:
            self.q_target.load_state_dict(self.q.state_dict())


    def train(self, gamma, batch_size, update_iter=10):
        for _ in range(update_iter):
            # gather a sample of the transition data from memory
            s, a, r, s_prime, done_mask = self.memory.sample(batch_size)
            
            # estimate the value of the current states and actions
            q_out = self.q(s)
            q_a = q_out.gather(2, a.unsqueeze(-1).long()).squeeze(-1)
            #shape = tuple(s.shape)
            ##print(shape)
            #draw_graph(self.q, input_size=shape, expand_nested=True, filename='./diagrams/commnet', save_graph=True)

            # estimate the target value as the discounted q value of the optimal actions in the next states
            max_q_prime = self.q_target(s_prime).max(dim=2)[0]
            target = r + gamma * max_q_prime * done_mask

            # calculate loss from the original state_action and next state
            loss = F.smooth_l1_loss(q_a, target.detach())

            # perform backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

class CommLayer(nn.Module):
    """ Custom Linear layer for CommNet implementation """
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out

        # seperate weights for hidden layer and communication vector
        # self.b = torch.Tensor(size_out)
        self.T = nn.Parameter(torch.ones(size_in, size_out))

        # initialize weights
        lim = 0.01  # initialize weights and bias
        # nn.init.kaiming_uniform_(self.T, a=math.sqrt(5)) # weight init
        # nn.init.uniform_(self.b, -lim, +lim)
        nn.init.uniform_(self.T, 1-lim, 1+lim)

    def forward(self, x):
        #h, c = x[:, :, :self.size_in], x[:, self.size_in:]
        n = x.shape[1]
        # print(x.size())
        # print(self.T.t())
        h = torch.permute(x, (0, 2, 1))
        J = (torch.ones(n) - torch.eye(n)) * (1/(n-1)) + torch.eye(n)

        T_norm = torch.mul(self.T.t(), J)
        h_next = torch.matmul(h, T_norm)
        return torch.permute(h_next, (0, 2, 1))


class QNet(nn.Module):
    """A Deep Q Network 
    """
    def __init__(self, observation_space, action_space, env_name, neighborhood):
        """
        Args:
            observation_space ([gym.Env.observation_space]): The observation space for each agent
            action_space ([gym.Env.action_space]): The action space for each agent
        """
        super(QNet, self).__init__()
        self.num_agents = len(observation_space)
        #self.agent_pos = [(0,0) for _ in range(self.num_agents)]

        self.set_device()

        self.hx_size = 64
        self.k = 1

        self.env_name = env_name
        self.neighborhood = neighborhood

        n_obs = observation_space[0].shape[0]
        self.encode_net = nn.Sequential(
            nn.Linear(n_obs, 128),
            nn.ReLU(),
            nn.Linear(128, self.hx_size),
            nn.ReLU(),
        ).to(self.device)
        # self.encode_nets.append(encode_net)

        self.comm_net = nn.Sequential(
            #nn.Linear(2*self.hx_size, self.hx_size),
            #CommLayer(self.hx_size, self.hx_size),
            CommLayer(self.num_agents, self.num_agents),
            nn.Tanh(),
        ).to(self.device)
        # self.comm_nets.append(comm_net)

        self.decode_net = nn.Sequential(
            #nn.Linear(self.hx_size, action_space[agent_i].n)
            nn.Linear(self.hx_size, action_space[0].n)
        ).to(self.device)
        # self.decode_nets.append(decode_net)

    def set_device(self, i=0):
        if torch.cuda.is_available():  
            dev = f'cuda:{i}' 
            print(f"Using Device: {torch.cuda.get_device_name(i)}")
        else:  
            dev = 'cpu'  
        print(f"device name: {dev}")
        device = torch.device(dev)  
        self.device = device

        self.to(device)

    def obs_to_pos(self, obs):
        env_name = self.env_name
        if env_name == "ma_gym:Switch4-v0":
            X, Y = 3, 7 # grid size for switch
            scales = torch.tensor([float(X-1), float(Y-1)])
            # obs is num_agents copies of 2d position of each agent scaled to the grid
            pos = torch.mul(obs, scales)
        elif env_name == "ma_gym:TrafficJunction4-v0": 
            X, Y = 14, 14 # grid-size for traffic
            scales = torch.tensor([float(X-1), float(Y-1)])
            # obs is num_agents copies of 3^2 observable area around each agent
            ob_size = 3**2
            pos = torch.zeros(obs.shape[0], self.num_agents, 2)
            for i in range(self.num_agents):
                ob = obs[:, i, :]
                # center cell contains info about the current agent
                center = ob[:, 4*ob_size:5*ob_size]
                #ag_id = center[:, 0:4]
                ag_pos = center[:, 4:6]
                #ag_rt = center[:, 6:]
                # rescale the agent position
                p = torch.mul(ag_pos, scales)
                pos[:, i] = p
        return pos

    def get_nei_mask(self, obs):
        agent_pos = self.obs_to_pos(obs)
        mask = torch.zeros(obs.shape[0], self.num_agents)
        # distances between each pair of agents (p=1.0 for manhatten distance)
        dists = torch.cdist(agent_pos, agent_pos, p=1.0)
        # mask represeting other agents which are within the neighborhood of a given agent
        mask = torch.where(dists > 0.0, dists, float('inf'))
        mask = torch.where(mask < self.neighborhood, 1.0, 0.0)
        return mask

    def get_communications(self, mask, hidden):
        batch_size = hidden.shape[0]
        hx_size = hidden.shape[2]

        comms = torch.empty(batch_size, self.num_agents, hx_size)
        for i in range(self.num_agents):
            # mask for which agents will be listened to by agent 'i'
            mask_i = mask[:, i, :]
            # coefficent tensor for averaging across whole batch
            J = mask_i.sum(dim=1)
            J = torch.where(J > 0.0, J, 1.0)

            # apply mask to get sum of hidden layers from only appropriate agents
            total_comm = torch.einsum('ik,ikj->ij', mask[:, i, :],  hidden)

            # multiply by coefficient tensor to get result representing average communication
            comms[:, i, :] = torch.einsum('ij,i->ij',total_comm, (1/J))

        return comms


    def forward(self, obs):
        # print(f"obs has size: {obs.size()}")
        q_values = [torch.empty(obs.shape[0], )] * self.num_agents
        torch.autograd.set_detect_anomaly(True)

        hidden = torch.empty(obs.shape[0], self.num_agents, self.hx_size).to(self.device)
        
        # get neighbor hook mask tensor
        mask = self.get_nei_mask(obs) # b x n x n

        # get observation encodings for all the agents
        hidden = self.encode_net(obs)
        #hidden = x.squeeze()

        # perform communication between agents
        for t in range(self.k):
            hidden = self.comm_net(hidden)

        # extract action preferences from the resultant hidden layers
        q_values = self.decode_net(hidden)
        return q_values

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        mask = (torch.rand((out.shape[0],)) <= epsilon)
        action = torch.empty((out.shape[0], out.shape[1],))
        action[mask] = torch.randint(0, out.shape[2], action[mask].shape).float()
        action[~mask] = out[~mask].argmax(dim=2).float()
        return action
