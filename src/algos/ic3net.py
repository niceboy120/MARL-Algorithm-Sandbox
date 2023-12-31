

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.replay_buffer import ReplayBuffer

import random

import numpy as np

from common.learner import MALearner

from torchview import draw_graph

default_options = {
    'lr': 0.0005,
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

class IC3NET(MALearner):
    def __init__(self, observation_space, action_space, options=default_options):
        super().__init__(observation_space, action_space, options)

        # instantiate the prediction (q) and target (q_target) networks
        self.q = QNet(observation_space, action_space)
        self.q_target = QNet(observation_space, action_space)
        self.q_target.load_state_dict(self.q.state_dict())

        print("\nInstantiating Prediction Network:")
        self.display_info(self.q)

        # tell the optimizer to update the network parameters at the lr
        self.optimizer = optim.Adam(self.q.parameters(), lr=self.lr)

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

    def get_gates(self):
        return self.q_target.gates


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
            #print(f"ABOUT TO CALL forward for q_target!!!")
            #print(f"state is: {s_prime}")
            max_q_prime = self.q_target(s_prime).max(dim=2)[0]
            target = r + gamma * max_q_prime * done_mask

            # calculate loss from the original state_action and next state
            loss = F.smooth_l1_loss(q_a, target.detach())

            # perform backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()




class QNet(nn.Module):
    """A Deep Q Network 
    """
    def __init__(self, observation_space, action_space):
        """
        Args:
            observation_space ([gym.Env.observation_space]): The observation space for each agent
            action_space ([gym.Env.action_space]): The action space for each agent
        """
        super(QNet, self).__init__()
        self.num_agents = len(observation_space)
        self.set_device()

        self.batch_size = 1
        self.hx_size = 64
        self.k = 4

        self.encode_nets = nn.ModuleList()
        self.gate_nets = nn.ModuleList()


        #self.lstm_nets = nn.ModuleList()
        self.shared_lstm = nn.LSTM(self.hx_size, self.hx_size).to(self.device)

        self.comm_nets = nn.ModuleList()
        self.decode_nets = nn.ModuleList()

        self.init_hidden(1)


        for agent_i in range(self.num_agents):
            n_obs = observation_space[agent_i].shape[0]

            encode_net = nn.Sequential(
                nn.Linear(n_obs, 128),
                nn.ReLU(),
                nn.Linear(128, self.hx_size),
                nn.ReLU(),
            ).to(self.device)
            self.encode_nets.append(encode_net)

            # for determining whether to communicate or not
            gate_net = nn.Sequential(
                nn.Linear(self.hx_size, 2),
                nn.ReLU(),
                nn.Softmax(),
            ).to(self.device)
            self.gate_nets.append(gate_net)

            #lstm_net = nn.LSTM(self.hx_size, self.hx_size).to(self.device)
            #self.lstm_nets.append(lstm_net)


            decode_net = nn.Sequential(
                nn.Linear(self.hx_size, action_space[agent_i].n)
            ).to(self.device)
            self.decode_nets.append(decode_net)

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


    def get_communications(self, mask, hidden):
        batch_size = hidden.shape[0]
        hx_size = hidden.shape[2]

        comms = torch.empty(batch_size, self.num_agents, hx_size).to(self.device)
        for i in range(self.num_agents):
            # mask for which agents will be listened to by agent 'i'
            mask_i = mask[:, i, :]
            # coefficent tensor for averaging across whole batch
            J = mask_i.sum(dim=1)
            J = torch.where(J > 0.0, J, 1.0)

            # apply mask to get sum of hidden layers from only appropriate agents
            #print(f"mask_i size: {mask_i.size()}")
            #print(f"hidden size: {hidden.size()}")
            total_comm = torch.einsum('ik,ikj->ij', mask_i,  hidden)

            # multiply by coefficient tensor to get result representing average communication
            comms[:, i, :] = torch.einsum('ij,i->ij',total_comm, (1/J))

        return comms

    def init_hidden(self, batch_size):
        self.lstm_s = torch.zeros((batch_size, self.num_agents, self.hx_size)).to(self.device).detach()
        self.lstm_h = torch.zeros((batch_size, self.num_agents, self.hx_size)).to(self.device).detach()
        self.comm = torch.zeros(batch_size, self.num_agents, self.hx_size).to(self.device).detach()
        self.gates = torch.empty(batch_size, self.num_agents).to(self.device)


    def forward(self, obs):
        obs = obs.to(self.device)
        #batch_size = obs.shape[0]
        batch_size = obs.shape[0]
        #if batch_size != self.batch_size:
        self.init_hidden(batch_size)

        q_values = [torch.empty(batch_size, ).to(self.device)] * self.num_agents
        torch.autograd.set_detect_anomaly(True)

        # get observation encodings for all the agents
        encodings = torch.empty(batch_size, self.num_agents, self.hx_size).to(self.device)
        for agent_i in range(self.num_agents):
            agent_obs = obs[:, agent_i, :].to(self.device)
            agent_encode_net = self.encode_nets[agent_i]
            x = agent_encode_net(agent_obs)
            encodings[:, agent_i, :] = x.squeeze()

        # get the gating for this layer
        for i in range(self.num_agents):
            agent_gate_net = self.gate_nets[i]
            #print(f"ls_h has size: {self.lstm_h[:, :, i].size()}")
            x = agent_gate_net(self.lstm_h[:, i, :])
            #print(f"x has size: {x.size()}")
            #print(f"x: {x}")
            #self.gates[:, i] = torch.argmax(x.squeeze().squeeze().flatten())
            self.gates[:, i] = x[:, 0]

        # pass the encodings through the lstm
        #next_lstm_h = torch.empty(batch_size, self.hx_size, self.num_agents).to(self.device)
        #next_lstm_s = torch.empty(batch_size, self.hx_size, self.num_agents).to(self.device)
        hidden = torch.empty(batch_size, self.num_agents, self.hx_size).to(self.device)
        lstm_s = torch.empty(batch_size, self.num_agents, self.hx_size).to(self.device)
        for i in range(self.num_agents):

            e_i = encodings[:, i, :]
            c_i = self.comm[:, i, :]
            e_c_sum = torch.add(e_i, c_i).detach().unsqueeze(0)

            l_h = self.lstm_h[:, i, :].contiguous().detach().unsqueeze(0)
            l_s = self.lstm_s[:, i, :].contiguous().detach().unsqueeze(0)

            # print(f"e_c_sum has size: {e_c_sum.size()}")
            # print(f"l_h has size: {l_h.size()}")
            # print(f"l_s has size: {l_s.size()}")
            x, (hn, sn) = self.shared_lstm(e_c_sum, (l_h, l_s))
            # print(f"DONE!!!")
            # print(f"x has size: {x.size()}")
            # print(f"hn has size: {hn.size()}")
            # print(f"sn has size: {sn.size()}")

            x = x.squeeze(0)
            hn = hn.squeeze(0)
            sn = sn.squeeze(0)

            #next_lstm_h[:, i, :]  = hn
            #next_lstm_s[:, i, :]  = sn

            lstm_s[:,i,:] = sn
            hidden[:, i, :] = x.squeeze()

        self.lstm_s = lstm_s
        self.lstm_h = hidden


        # perform communication between agents
        #print(f"gates are: {self.gates.size()}")
        #print(f"gates are: {self.gates}")
        mask = torch.stack([self.gates]*4, dim=2) # gates act as a mask for each agent
        #print(f"mask has size: {mask.size()}")
        #print(f"mask is: {mask}")
        self.comm = self.get_communications(mask, hidden)

        # extract action preferences from the resultant hidden layers
        for agent_i in range(self.num_agents):
            agent_decode_net = self.decode_nets[agent_i]
            q_values[agent_i] = agent_decode_net(hidden[:, agent_i, :]).unsqueeze(1)

        return torch.cat(q_values, dim=1).to(self.device)



    def sample_action(self, obs, epsilon):

        #print(f"ABOUT TO CALL forward from sample action!!!")
        #print(f"state is: {obs.size()}")
        out = self.forward(obs).to(self.device)

        mask = (torch.rand((out.shape[0],)).to(self.device) <= epsilon)
        action = torch.empty((out.shape[0], out.shape[1],)).to(self.device)
        action[mask] = torch.randint(0, out.shape[2], action[mask].shape).float().to(self.device)
        action[~mask] = out[~mask].argmax(dim=2).float().to(self.device)

        return action
