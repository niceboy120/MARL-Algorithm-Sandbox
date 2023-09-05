

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.replay_buffer import ReplayBuffer

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

class G2ANET(MALearner):
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

        self.hx_size = 64
        self.num_heads = 2
        self.k = 4

        self.encode_nets = nn.ModuleList()
        self.hard_att_nets = nn.ModuleList()


        self.self_attn = nn.MultiheadAttention(self.hx_size, self.num_heads).to(self.device)

        self.decode_nets = nn.ModuleList()


        for i in range(self.num_agents):
            n_obs = observation_space[i].shape[0]

            # each agents encoding of a feature vector
            encode_net = nn.Sequential(
                nn.Linear(n_obs, 128),
                nn.ReLU(),
                nn.Linear(128, self.hx_size),
                nn.ReLU(),
            ).to(self.device)
            self.encode_nets.append(encode_net)


            # an RNN between each pair of agents for determining hard attention
            for j in range(self.num_agents):
                #if j != i:
                hard_bi_GRU = nn.Sequential(
                    nn.GRU(2 * self.hx_size, 2 * self.hx_size, bidirectional=True),
                    nn.Linear(2 * self.hx_size, self.hx_size),
                    #nn.ReLU(),
                ).to(self.device)
                self.hard_att_nets.append(hard_bi_GRU)

            decode_net = nn.Sequential(
                nn.Linear(self.hx_size * 2, action_space[i].n),
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


    def forward(self, obs):
        batch_size = obs.shape[0]
        q_values = [torch.empty(batch_size, ).to(self.device)] * self.num_agents
        torch.autograd.set_detect_anomaly(True)

        # tensor representing the hidden layers containing encoded observation data for each agent
        hidden = torch.empty(batch_size, 1, self.hx_size, self.num_agents).to(self.device)
        # get observation encodings for all the agents
        for agent_i in range(self.num_agents):
            agent_obs = obs[:, agent_i, :].to(self.device)
            agent_encode_net = self.encode_nets[agent_i]

            # encoded observation data for agent i
            x = agent_encode_net(agent_obs)
            hidden[:, : , :, agent_i] = x.unsqueeze(1)

        # hard attention
        hard_weights = torch.empty(batch_size, self.num_agents, self.num_agents).to(self.device)
        print(f"hard_weights: {hard_weights.size()}")

        input_hard = []
        n = self.num_agents
        for i in range(n):
            hi = hidden[:, :, :, i]
            for j in range(n):
                if j != i:
                    hj = hidden[:, :, :, j]

                    # the input for this agent pair consisting of both their feature vectors
                    hard_att_input = torch.cat((hi, hj), axis=2).to(self.device)
                    print(hard_att_input.size())

                    # the corresponding RNN for this agent pair
                    pair_rnn = self.hard_att_nets[i + j*n]
                    out, e = pair_rnn(hard_att_input)
                    e.to(self.device)
                    #print(f"hard_weights: {hard_weights[:, :, :, i, j].size()}")
                    print(f"out: {out.size()}")
                    print(f"e: {e.size()}")
                    hard_weights[:, i, j] = F.gumbel_softmax(out, tau=1, hard=True)



        # soft attention
        # soft_weights= torch.empty(batch_size, 1, self.hx_size, self.num_agents, self.num_agents-1).to(self.device)

        # input_hard = []
        # n = self.num_agents
        # for i in range(n):
        #     hi = hidden[:, :, :, i]
        #     for j in range(n):
        #         if j != i:
        #             print(f"out: {out.size()}")

        #self attention
        _src, _ = self.self_attn(hidden, hidden, hidden, src_mask)

        #print(hard_weights)

        # get the contribution of other agents from the hard and soft attention weihts
        other_contrib = torch.empty(batch_size, 1, self.hx_size, self.num_agents).to(self.device)
        for i in range(n):
            #x_i = torch.empty(batch_size, 1, self.hx_size, self.num_agents).to(self.device)
            for j in range(n):
                if j != i:
                    hj, w_ij = hidden[:, :, :, j], hard_weights[:, :, :, i, j]
                    print(f"hj: {hj.size()}")
                    print(f"w_ij: {w_ij.size()}")
                    other_contrib[:, :, :, i] = torch.inner(w_ij, hj)
            #print(f"other_contrib: {other_contrib.size()}")
            #print(f"x_i: {x_i.size()}")
            #other_contrib[:, :, :, i] = x_i


        # determine action preference based on feature vector and other's contributions
        for i in range(n):
            agent_decode_net = self.decode_nets[i]
            hi = hidden[:, 0, :, i]
            x_i = other_contrib[:, 0, :, i]

            comm_input = torch.cat((hi, x_i), axis=1).to(self.device)
            q_values[i] = agent_decode_net(comm_input).unsqueeze(1)
        return torch.cat(q_values, dim=1)

    def sample_action(self, obs, epsilon):
        out = self.forward(obs).to(self.device)
        mask = (torch.rand((out.shape[0],)).to(self.device) <= epsilon)
        action = torch.empty((out.shape[0], out.shape[1],)).to(self.device)
        action[mask] = torch.randint(0, out.shape[2], action[mask].shape).float().to(self.device)
        action[~mask] = out[~mask].argmax(dim=2).float().to(self.device)
        return action
