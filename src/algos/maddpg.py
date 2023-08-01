

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.replay_buffer import ReplayBuffer

import numpy as np


from common.learner import MALearner



default_options = {
    'lr_mu': 0.0005,
    'lr_q': 0.001,
    'batch_size': 32,
    'tau': 0.005,
    'gamma': 0.99,
    'buffer_limit': 50000,
    'warm_up_steps': 2000,
    'update_iter': 10,
    'gumbel_max_temp': 10,
    'gumbel_min_temp': 0.1
}


class MADDPG(MALearner):
    def __init__(self, observation_space, action_space, options=default_options):
        super().__init__(observation_space, action_space, options)

        # Critic network
        self.q = QNet(observation_space, action_space)
        self.q_target = QNet(observation_space, action_space)
        self.q_target.load_state_dict(self.q.state_dict())

        # Actor network
        self.mu = MuNet(observation_space, action_space)
        self.mu_target = MuNet(observation_space, action_space)
        self.mu_target.load_state_dict(self.mu.state_dict())

        # optimizers
        self.mu_optimizer = optim.Adam(self.mu.parameters(), lr=self.lr_mu)
        self.q_optimizer = optim.Adam(self.q.parameters(), lr=self.lr_q)

    def sample_action(self, obs, epsilon):
        action_logits = self.mu(torch.Tensor(obs).unsqueeze(0))
        action_one_hot = F.gumbel_softmax(logits=action_logits.squeeze(0), tau=self.temperature, hard=True)
        action = torch.argmax(action_one_hot, dim=1).data.numpy()
        return action

    def per_step_update(self, state, action, reward, next_state, done):
        self.memory.put((state, action, (np.array(reward)).tolist(), next_state, [int(all(done))]))

    def per_episode_update(self, episode_i):
        self.temperature = max(
            self.gumbel_min_temp,
            self.gumbel_max_temp - (self.gumbel_max_temp - self.gumbel_min_temp) * (episode_i / (0.6 * self.max_episodes))
        )
        
        if self.memory.size() > self.warm_up_steps:
            self.train(self.gamma, self.batch_size, self.update_iter, self.chunk_size)
            self.soft_update(self.mu, self.mu_target, self.tau)
            self.soft_update(self.q, self.q_target, self.tau)

        if self.memory.size() > self.warm_up_steps:
            for i in range(self.update_iter):
                self.train(self.gamma, self.batch_size)
            
        # at every update interval update the target q_network
        if episode_i % self.update_target_interval:
            self.q_target.load_state_dict(self.q.state_dict())


    def soft_update(self, net, net_target, tau):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            

    def train(self, gamma, batch_size, update_iter=10, chunk_size=10, grad_clip_norm=5):
        state, action, reward, next_state, done_mask = self.memory.sample(batch_size)

        next_state_action_logits = self.mu_target(next_state)
        _, n_agents, action_size = next_state_action_logits.shape
        next_state_action_logits = next_state_action_logits.view(batch_size * n_agents, action_size)
        next_state_action = F.gumbel_softmax(logits=next_state_action_logits, tau=0.1, hard=True)
        next_state_action = next_state_action.view(batch_size, n_agents, action_size)

        # Critic Update
        target = reward + gamma * self.q_target(next_state, next_state_action) * done_mask
        q_loss = F.smooth_l1_loss(self.q(state, action), target.detach())
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        state_action_logits = self.mu(state)
        state_action_logits = state_action_logits.view(batch_size * n_agents, action_size)
        state_action = F.gumbel_softmax(logits=state_action_logits, tau=0.1, hard=True)
        state_action = state_action.view(batch_size, n_agents, action_size)

        # Actor Update
        mu_loss = -self.q(state, state_action).mean()  # That's all for the policy loss.
        self.q_optimizer.zero_grad()
        self.mu_optimizer.zero_grad()
        mu_loss.backward()
        self.mu_optimizer.step()




class MuNet(nn.Module):
    def __init__(self, observation_space, action_space):
        super(MuNet, self).__init__()
        self.num_agents = len(observation_space)
        self.action_space = action_space
        for agent_i in range(self.num_agents):
            n_obs = observation_space[agent_i].shape[0]
            num_action = action_space[agent_i].n
            setattr(self, 'agent_{}'.format(agent_i), 
                nn.Sequential(nn.Linear(n_obs, 128),
                    nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, num_action)
                    )
                )

    def forward(self, obs):
        action_logits = [torch.empty(1, _.n) for _ in self.action_space]
        for agent_i in range(self.num_agents):
            x = getattr(self, 'agent_{}'.format(agent_i))(obs[:, agent_i, :]).unsqueeze(1)
            action_logits[agent_i] = x

        return torch.cat(action_logits, dim=1)


class QNet(nn.Module):
    def __init__(self, observation_space, action_space):
        super(QNet, self).__init__()
        self.num_agents = len(observation_space)
        total_action = sum([_.n for _ in action_space])
        total_obs = sum([_.shape[0] for _ in observation_space])
        for agent_i in range(self.num_agents):
            setattr(self, 'agent_{}'.format(agent_i), 
                nn.Sequential(
                    nn.Linear(total_obs + total_action, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )
            )
    def forward(self, obs, action):
        q_values = [torch.empty(obs.shape[0], )] * self.num_agents
        x = torch.cat((obs.view(obs.shape[0], obs.shape[1] * obs.shape[2]),
                       action.view(action.shape[0], action.shape[1] * action.shape[2])), dim=1)
        for agent_i in range(self.num_agents):
            q_values[agent_i] = getattr(self, 'agent_{}'.format(agent_i))(x)

        return torch.cat(q_values, dim=1)

