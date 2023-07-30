
import torch
import torch.nn as nn
import torch.nn.functional as F


class VDN(nn.Module):


    def train(q, q_target, memory, optimizer, gamma, batch_size, update_iter=10, chunk_size=10, grad_clip_norm=5):
        _chunk_size = chunk_size if q.recurrent else 1
        for _ in range(update_iter):
            s, a, r, s_prime, done = memory.sample_chunk(batch_size, _chunk_size)

            hidden = q.init_hidden(batch_size)
            target_hidden = q_target.init_hidden(batch_size)
            loss = 0
            for step_i in range(_chunk_size):
                q_out, hidden = q(s[:, step_i, :, :], hidden)
                q_a = q_out.gather(2, a[:, step_i, :].unsqueeze(-1).long()).squeeze(-1)
                sum_q = q_a.sum(dim=1, keepdims=True)

                max_q_prime, target_hidden = q_target(s_prime[:, step_i, :, :], target_hidden.detach())
                max_q_prime = max_q_prime.max(dim=2)[0].squeeze(-1)
                target_q = r[:, step_i, :].sum(dim=1, keepdims=True)
                target_q += gamma * max_q_prime.sum(dim=1, keepdims=True) * (1 - done[:, step_i])

                loss += F.smooth_l1_loss(sum_q, target_q.detach())

                done_mask = done[:, step_i].squeeze(-1).bool()
                hidden[done_mask] = q.init_hidden(len(hidden[done_mask]))
                target_hidden[done_mask] = q_target.init_hidden(len(target_hidden[done_mask]))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), grad_clip_norm, norm_type=2)
            optimizer.step()


    def test(env, num_episodes, q):
        score = 0
        for episode_i in range(num_episodes):
            state = env.reset()
            done = [False for _ in range(env.n_agents)]
            with torch.no_grad():
                hidden = q.init_hidden()
                while not all(done):
                    action, hidden = q.sample_action(torch.Tensor(state).unsqueeze(0), hidden, epsilon=0)
                    next_state, reward, done, info = env.step(action[0].data.cpu().numpy().tolist())
                    score += sum(reward)
                    state = next_state

        return score / num_episodes


    def __init__(self, observation_space, action_space, recurrent=False):
        super(VDN, self).__init__()
        self.num_agents = len(observation_space)
        self.recurrent = recurrent
        self.hx_size = 32
        for agent_i in range(self.num_agents):
            n_obs = observation_space[agent_i].shape[0]
            setattr(self, 'agent_feature_{}'.format(agent_i), nn.Sequential(nn.Linear(n_obs, 64),
                                                                            nn.ReLU(),
                                                                            nn.Linear(64, self.hx_size),
                                                                            nn.ReLU()))
            if recurrent:
                setattr(self, 'agent_gru_{}'.format(agent_i), nn.GRUCell(self.hx_size, self.hx_size))
            setattr(self, 'agent_q_{}'.format(agent_i), nn.Linear(self.hx_size, action_space[agent_i].n))

    def forward(self, obs, hidden):
        q_values = [torch.empty(obs.shape[0], )] * self.num_agents
        next_hidden = [torch.empty(obs.shape[0], 1, self.hx_size)] * self.num_agents
        for agent_i in range(self.num_agents):
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
