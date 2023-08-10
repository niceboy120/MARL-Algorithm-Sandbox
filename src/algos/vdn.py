
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.replay_buffer import ReplayBuffer

import numpy as np

from common.learner import MALearner

# from torchview import draw_graph

default_options = {
    'lr': 0.001,
    'batch_size': 32,
    'gamma': 0.99,
    'buffer_limit': 50000,
    'update_target_interval': 20,
    'warm_up_steps': 500,
    'update_iter': 10,
    'chunk_size': 10,  # if not recurrent, internally, we use chunk_size of 1 and no gru cell is used.
    'recurrent': True
}

class VDN(MALearner):
    def __init__(self, observation_space, action_space, options=default_options):
        super().__init__(observation_space, action_space, options)

        self.q = QNet(observation_space, action_space, self.recurrent)
        self.q_target = QNet(observation_space, action_space, self.recurrent)
        self.q_target.load_state_dict(self.q.state_dict())


        print("\nInstantiating Prediction Network:")
        self.display_info(self.q)

        self.optimizer = optim.Adam(self.q.parameters(), lr=self.lr)

        self.q_hidden = self.q.init_hidden()

    # def init_hidden(self):
    #     return self.q.init_hidden()


    def sample_action(self, obs, epsilon):
        action = self.q.sample_action(obs, epsilon)
        # action, hidden = self.q.sample_action(obs, self.q_hidden, epsilon)
        # self.q_hidden = hidden
        return action

    def per_step_update(self, state, action, reward, next_state, done):
        self.memory.put((state, action, (np.array(reward)).tolist(), next_state, [int(all(done))]))

    def per_episode_update(self, episode_i):
        if self.memory.size() > self.warm_up_steps:
            self.train(self.gamma, self.batch_size, self.update_iter, self.chunk_size)

        # at every update interval update the target q_network
        if episode_i % self.update_target_interval:
            self.q_target.load_state_dict(self.q.state_dict())


    def train(self, gamma, batch_size, update_iter=10, chunk_size=10, grad_clip_norm=5):
        _chunk_size = chunk_size if self.q.recurrent else 1
        for _ in range(update_iter):
            # gather a sample of the transition data from memory
            s, a, r, s_prime, done = self.memory.sample_chunk(batch_size, _chunk_size)

            # hidden = self.q.init_hidden(batch_size)
            # target_hidden = self.q_target.init_hidden(batch_size)
            loss = 0
            for step_i in range(_chunk_size):
                # estimate the value of the current states and actions
                # q_out, hidden = self.q(s[:, step_i, :, :], hidden)
                q_out = self.q(s[:, step_i, :, :])
                q_a = q_out.gather(2, a[:, step_i, :].unsqueeze(-1).long()).squeeze(-1)
                # sum over all the action values
                sum_q = q_a.sum(dim=1, keepdims=True)


                #shape = tuple(s.shape)
                ##print(shape)
                #draw_graph(self.q, input_size=shape, expand_nested=True, filename='./diagrams/vdn', save_graph=True)


                # estimate the target value as the discounted q value of the optimal actions in the next states
                # max_q_prime, target_hidden = self.q_target(s_prime[:, step_i, :, :], target_hidden.detach())
                max_q_prime = self.q_target(s_prime[:, step_i, :, :])
                max_q_prime = max_q_prime.max(dim=2)[0].squeeze(-1)
                target_q = r[:, step_i, :].sum(dim=1, keepdims=True)
                target_q += gamma * max_q_prime.sum(dim=1, keepdims=True) * (1 - done[:, step_i])

                # calculate loss from the summed action value and next state values
                loss += F.smooth_l1_loss(sum_q, target_q.detach())

                done_mask = done[:, step_i].squeeze(-1).bool()
                # hidden[done_mask] = self.q.init_hidden(len(hidden[done_mask]))
                # target_hidden[done_mask] = self.q_target.init_hidden(len(target_hidden[done_mask]))

            # perform backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), grad_clip_norm, norm_type=2)
            self.optimizer.step()


class QNet(nn.Module):

    def __init__(self, observation_space, action_space, recurrent=False):
        super(QNet, self).__init__()
        self.num_agents = len(observation_space)
        self.recurrent = recurrent
        self.hx_size = 32
        for agent_i in range(self.num_agents):
            n_obs = observation_space[agent_i].shape[0]
            setattr(self, 'agent_feature_{}'.format(agent_i), 
                nn.Sequential(
                    nn.Linear(n_obs, 64),
                    nn.ReLU(),
                    nn.Linear(64, self.hx_size),
                    nn.ReLU(),
                    nn.GRUCell(self.hx_size, self.hx_size),
                    nn.Linear(self.hx_size, action_space[agent_i].n)
                )
            )
            # if recurrent:
            #     setattr(self, 'agent_gru_{}'.format(agent_i), 
            #         nn.GRUCell(self.hx_size, self.hx_size)
            #     )
            # setattr(self, 'agent_q_{}'.format(agent_i), 
            #     nn.Linear(self.hx_size, action_space[agent_i].n)
            # )

    #def forward(self, obs, hidden):
    def forward(self, obs):
        q_values = [torch.empty(obs.shape[0], )] * self.num_agents
        # next_hidden = [torch.empty(obs.shape[0], 1, self.hx_size)] * self.num_agents
        for agent_i in range(self.num_agents):
            # x = getattr(self, 'agent_feature_{}'.format(agent_i))(obs[:, agent_i, :])
            # if self.recurrent:
            #     x = getattr(self, 'agent_gru_{}'.format(agent_i))(x, hidden[:, agent_i, :])
            #     next_hidden[agent_i] = x.unsqueeze(1)
            # q_values[agent_i] = getattr(self, 'agent_q_{}'.format(agent_i))(x).unsqueeze(1)
            q_values[agent_i] = getattr(self, 'agent_feature_{}'.format(agent_i))(obs[:, agent_i, :]).unsqueeze(1)

        #return torch.cat(q_values, dim=1), torch.cat(next_hidden, dim=1)
        return torch.cat(q_values, dim=1)

    # def sample_action(self, obs, hidden, epsilon):
    def sample_action(self, obs, epsilon):
        # out, hidden = self.forward(obs, hidden)
        out = self.forward(obs)
        # print(out.shape)
        mask = (torch.rand((out.shape[0],)) <= epsilon)
        action = torch.empty((out.shape[0], out.shape[1],))
        action[mask] = torch.randint(0, out.shape[2], action[mask].shape).float()
        action[~mask] = out[~mask].argmax(dim=2).float()
        # return action, hidden
        return action

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.num_agents, self.hx_size))
