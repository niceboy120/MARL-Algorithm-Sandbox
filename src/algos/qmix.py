
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.replay_buffer import ReplayBuffer

import numpy as np

from common.learner import MALearner


default_options = {
    'lr': 0.001,
    'batch_size': 32,
    'gamma': 0.99,
    'buffer_limit': 50000,
    'update_target_interval': 20,
    'warm_up_steps': 2000,
    'update_iter': 10,
    'chunk_size': 10,  # if not recurrent, internally, we use chunk_size of 1 and no gru cell is used.
    'recurrent': False
}



class QMix(MALearner):
    def __init__(self, observation_space, action_space, options=default_options):
        super().__init__(observation_space, action_space, options)

        # create networks
        self.q = QNet(observation_space, action_space, self.recurrent)
        self.q_target = QNet(observation_space, action_space, self.recurrent)
        self.q_target.load_state_dict(self.q.state_dict())

        self.mix_net = MixNet(observation_space, recurrent=self.recurrent)
        self.mix_net_target = MixNet(observation_space, recurrent=self.recurrent)
        self.mix_net_target.load_state_dict(self.mix_net.state_dict())

        self.optimizer = optim.Adam([{'params': self.q.parameters()}, {'params': self.mix_net.parameters()}], lr=self.lr)
        self.q_hidden = self.q.init_hidden()

    # def init_hidden(self):
    #     return self.q.init_hidden()

    def sample_action(self, obs, epsilon):
        action, hidden = self.q.sample_action(torch.Tensor(obs).unsqueeze(0), self.q_hidden, epsilon)
        action = action[0].data.cpu().numpy().tolist()
        self.q_hidden = hidden
        return action

    def per_step_update(self, state, action, reward, next_state, done):
        self.memory.put((state, action, (np.array(reward)).tolist(), next_state, [int(all(done))]))

    def per_episode_update(self, episode_i):

        if self.memory.size() > self.warm_up_steps:
            self.train(self.gamma, self.batch_size, self.update_iter, self.chunk_size)

        # at every update interval update the target q_network
        if episode_i % self.update_target_interval:
            self.q_target.load_state_dict(self.q.state_dict())
            self.mix_net_target.load_state_dict(self.mix_net.state_dict())
            self.q_target.load_state_dict(self.q.state_dict())


    def train(self, gamma, batch_size, update_iter=10, chunk_size=10, grad_clip_norm=5):
        _chunk_size = chunk_size if self.q.recurrent else 1
        for _ in range(update_iter):
            s, a, r, s_prime, done = self.memory.sample_chunk(batch_size, _chunk_size)

            hidden = self.q.init_hidden(batch_size)
            target_hidden = self.q_target.init_hidden(batch_size)
            mix_net_target_hidden = self.mix_net_target.init_hidden(batch_size)
            mix_net_hidden = [torch.empty_like(mix_net_target_hidden) for _ in range(_chunk_size + 1)]
            mix_net_hidden[0] = self.mix_net_target.init_hidden(batch_size)

            loss = 0
            for step_i in range(_chunk_size):
                q_out, hidden = self.q(s[:, step_i, :, :], hidden)
                q_a = q_out.gather(2, a[:, step_i, :].unsqueeze(-1).long()).squeeze(-1)
                pred_q, next_mix_net_hidden = self.mix_net(q_a, s[:, step_i, :, :], mix_net_hidden[step_i])

                max_q_prime, target_hidden = self.q_target(s_prime[:, step_i, :, :], target_hidden.detach())
                max_q_prime = max_q_prime.max(dim=2)[0].squeeze(-1)
                q_prime_total, mix_net_target_hidden = self.mix_net_target(max_q_prime, s_prime[:, step_i, :, :],
                                                                    mix_net_target_hidden.detach())
                target_q = r[:, step_i, :].sum(dim=1, keepdims=True) + (gamma * q_prime_total * (1 - done[:, step_i]))
                loss += F.smooth_l1_loss(pred_q, target_q.detach())

                done_mask = done[:, step_i].squeeze(-1).bool()
                hidden[done_mask] = self.q.init_hidden(len(hidden[done_mask]))
                target_hidden[done_mask] = self.q_target.init_hidden(len(target_hidden[done_mask]))
                mix_net_hidden[step_i + 1][~done_mask] = next_mix_net_hidden[~done_mask]
                mix_net_hidden[step_i + 1][done_mask] = self.mix_net.init_hidden(len(mix_net_hidden[step_i][done_mask]))
                mix_net_target_hidden[done_mask] = self.mix_net_target.init_hidden(len(mix_net_target_hidden[done_mask]))

            ## perform backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), grad_clip_norm, norm_type=2)
            torch.nn.utils.clip_grad_norm_(self.mix_net.parameters(), grad_clip_norm, norm_type=2)
            self.optimizer.step()






class MixNet(nn.Module):
    def __init__(self, observation_space, hidden_dim=32, hx_size=64, recurrent=False):
        super(MixNet, self).__init__()
        state_size = sum([_.shape[0] for _ in observation_space])
        self.hidden_dim = hidden_dim
        self.hx_size = hx_size
        self.n_agents = len(observation_space)
        self.recurrent = recurrent

        hyper_net_input_size = state_size
        if self.recurrent:
            self.gru = nn.GRUCell(state_size, self.hx_size)
            hyper_net_input_size = self.hx_size
        self.hyper_net_weight_1 = nn.Linear(hyper_net_input_size, self.n_agents * hidden_dim)
        self.hyper_net_weight_2 = nn.Linear(hyper_net_input_size, hidden_dim)

        self.hyper_net_bias_1 = nn.Linear(hyper_net_input_size, hidden_dim)
        self.hyper_net_bias_2 = nn.Sequential(nn.Linear(hyper_net_input_size, hidden_dim),
                                              nn.ReLU(),
                                              nn.Linear(hidden_dim, 1))

    def forward(self, q_values, observations, hidden):
        batch_size, n_agents, obs_size = observations.shape
        state = observations.view(batch_size, n_agents * obs_size)

        x = state
        if self.recurrent:
            hidden = self.gru(x, hidden)
            x = hidden

        weight_1 = torch.abs(self.hyper_net_weight_1(x))
        weight_1 = weight_1.view(batch_size, self.hidden_dim, n_agents)
        bias_1 = self.hyper_net_bias_1(x).unsqueeze(-1)
        weight_2 = torch.abs(self.hyper_net_weight_2(x))
        bias_2 = self.hyper_net_bias_2(x)

        x = torch.bmm(weight_1, q_values.unsqueeze(-1)) + bias_1
        x = torch.relu(x)
        x = (weight_2.unsqueeze(-1) * x).sum(dim=1) + bias_2
        return x, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.hx_size))


class QNet(nn.Module):
    def __init__(self, observation_space, action_space, recurrent=False):
        super(QNet, self).__init__()
        self.num_agents = len(observation_space)
        self.recurrent = recurrent
        self.hx_size = 32
        for agent_i in range(self.num_agents):
            n_obs = observation_space[agent_i].shape[0]
            setattr(self, 'agent_feature_{}'.format(agent_i), nn.Sequential(nn.Linear(n_obs, 128),
                                                                            nn.ReLU(),
                                                                            nn.Linear(128, self.hx_size),
                                                                            nn.ReLU()))
            if recurrent:
                setattr(self, 'agent_gru_{}'.format(agent_i), nn.GRUCell(self.hx_size, self.hx_size))
            setattr(self, 'agent_q_{}'.format(agent_i), nn.Linear(self.hx_size, action_space[agent_i].n))

    def forward(self, obs, hidden):
        print(f"obs: {obs.shape}")
        q_values = [torch.empty(obs.shape[0], )] * self.num_agents
        next_hidden = [torch.empty(obs.shape[0], 1, self.hx_size, )] * self.num_agents
        for agent_i in range(self.num_agents):
            print(f"agent_i obs: {obs[:, agent_i, :].shape}")
            x = getattr(self, 'agent_feature_{}'.format(agent_i))(obs[:, agent_i, :])
            if self.recurrent:
                x = getattr(self, 'agent_gru_{}'.format(agent_i))(x, hidden[:, agent_i, :])
                next_hidden[agent_i] = x.unsqueeze(1)
            q_values[agent_i] = getattr(self, 'agent_q_{}'.format(agent_i))(x).unsqueeze(1)

        return torch.cat(q_values, dim=1), torch.cat(next_hidden, dim=1)

    def sample_action(self, obs, hidden, epsilon):
        out, hidden = self.forward(obs, hidden)
        mask = (torch.rand((out.shape[0],)) <= epsilon)
        action = torch.empty((out.shape[0], out.shape[1],))
        action[mask] = torch.randint(0, out.shape[2], action[mask].shape).float()
        action[~mask] = out[~mask].argmax(dim=2).float()
        return action, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.num_agents, self.hx_size))
