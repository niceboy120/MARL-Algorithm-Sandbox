
import torch
import torch.nn.functional as F
import numpy as np
from common.replay_buffer import ReplayBuffer

default_options = {
    'lr': 0.001,
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
    'recurrent': False,
    'monitor': False,
    'chunk_size': 10,  # if not recurrent, internally, we use chunk_size of 1 and no gru cell is used.
}


class MALearner:
    def __init__(self, observation_space, action_space, options=default_options):
        # update options
        self.unpack_options(options)
        self.memory = ReplayBuffer(self.buffer_limit)


    def unpack_options(self, options):
        # apply all dictionary key-values as object properties
        for option in options:
            setattr(self, option, options[option])

    def test(self, env, num_episodes):
        score = 0
        steps = 0
        for episode_i in range(num_episodes):
            state = env.reset()
            done = [False for _ in range(env.n_agents)]
            with torch.no_grad():
                #hidden = self.q.init_hidden()
                while not all(done):
                    action = self.sample_action(torch.Tensor(state).unsqueeze(0), epsilon=0)
                    next_state, reward, done, info = env.step(action[0].data.cpu().numpy().tolist())
                    score += sum(reward)
                    state = next_state
                    steps += 1

        avg_score = score / num_episodes
        avg_steps = steps / num_episodes
        return avg_score, avg_steps

    def display_info(self, net):
        print("\n Network Type:")
        print(net.type)
        print("\n Parameters are:")
        for param in net.parameters():
            print(type(param.data), param.size())



class StateLearner(MALearner):
    def __init__(self, observation_space, action_space, Net, options=default_options):
        super().__init__(observation_space, action_space, options)
        # instantiate the prediction (q) and target (q_target) networks
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


    def draw_net_graph(self, x, net):
            shape = tuple(x.shape)
            ##print(shape)
            #draw_graph(net, input_size=shape, expand_nested=True, filename='./diagrams/commnet', save_graph=True)


    def train(self, gamma, batch_size, update_iter=10):
        # implement in subclass
        pass
