
import argparse

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from common.replay_buffer import ReplayBuffer
from algos.vdn import VDN




def main(env_name, lr, gamma, batch_size, buffer_limit, log_interval, max_episodes, max_epsilon, min_epsilon, test_episodes, max_steps, warm_up_steps, update_iter, chunk_size, update_target_interval, recurrent):
    env = gym.make(env_name)
    test_env = gym.make(env_name)
    memory = ReplayBuffer(buffer_limit)


    # create networks
    q = VDN(env.observation_space, env.action_space, recurrent)
    q_target = VDN(env.observation_space, env.action_space, recurrent)
    q_target.load_state_dict(q.state_dict())
    optimizer = optim.Adam(q.parameters(), lr=lr)

    score = 0
    result_data = list()
    # perfom each episode
    for episode_i in range(max_episodes):
        epsilon = max(min_epsilon, max_epsilon - (max_epsilon - min_epsilon) * (episode_i / (0.6 * max_episodes)))
        state = env.reset()
        done = [False for _ in range(env.n_agents)]
        with torch.no_grad():
            hidden = q.init_hidden()
            # perfoms steps according to the policy
            steps = 0
            while not all(done) and steps <= max_steps:
                action, hidden = q.sample_action(torch.Tensor(state).unsqueeze(0), hidden, epsilon)
                action = action[0].data.cpu().numpy().tolist()
                next_state, reward, done, info = env.step(action)
                # store the step data in the replay buffer
                memory.put((state, action, (np.array(reward)).tolist(), next_state, [int(all(done))]))

                # collect data for plotting
                score += sum(reward)
                steps += 1
                # increment the state
                state = next_state

        if memory.size() > warm_up_steps:
            VDN.train(q, q_target, memory, optimizer, gamma, batch_size, update_iter, chunk_size)

        # at every update interval update the target q_network
        if episode_i % update_target_interval:
            q_target.load_state_dict(q.state_dict())

        # log relavent data at every log interval
        if (episode_i + 1) % log_interval == 0:
            test_score = VDN.test(test_env, test_episodes, q)
            train_score = score / log_interval
            print("#{:<10}/{} episodes , avg train score : {:.1f}, test score: {:.1f} n_buffer : {}, eps : {:.1f}"
                  .format(episode_i, max_episodes, train_score, test_score, memory.size(), epsilon))

            result_data.append({'episode': episode_i, 'test-score': test_score, 'buffer-size': memory.size(), 'epsilon': epsilon, 'train-score': train_score})
            score = 0


    np.save("./results/data.npy", result_data)
    env.close()
    test_env.close()










if __name__ == '__main__':
    # gather arguments
    parser = argparse.ArgumentParser(description="MARL algorithm examples")
    parser.add_argument('--env-name', required=False, default='ma_gym:TrafficJunction4-v0')
    parser.add_argument('--seed', type=int, default=1, required=False)
    parser.add_argument('--no-recurrent', action='store_true')
    parser.add_argument('--max-episodes', type=int, default=15000, required=False)
    
    # process arguments
    args = parser.parse_args()


    kwargs= {'env_name': args.env_name,
              'lr': 0.001,
              'batch_size': 32,
              'gamma': 0.99,
              'buffer_limit': 50000,
              #'buffer_limit': 5000,
              'update_target_interval': 20,
              'log_interval': 100,
              'max_episodes': args.max_episodes,
              'max_epsilon': 0.9,
              'min_epsilon': 0.1,
              'test_episodes': 5,
              'max_steps': 10000,
              'warm_up_steps': 500,
              'update_iter': 10,
              'chunk_size': 10,  # if not recurrent, internally, we use chunk_size of 1 and no gru cell is used.
              'recurrent': not args.no_recurrent}

    main(**kwargs)
