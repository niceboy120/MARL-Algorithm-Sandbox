

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


import torch
from common.replay_buffer import ReplayBuffer

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
