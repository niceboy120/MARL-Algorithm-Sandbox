
import argparse

import gym
import numpy as np
import torch
import torch.nn.functional as F

from algos.vdn import VDN
from algos.idqn import IDQN
from algos.maddpg import MADDPG

#from algos.commnet import DDPG


def main(env_name, algo, results_dir, log_interval, num_episodes, num_runs, max_epsilon, min_epsilon, test_episodes, max_steps, options):
    env = gym.make(env_name)
    test_env = gym.make(env_name)

    # create networks
    print(f"using algorithm: {algo} on environment: {env_name}")
    match algo:
        case "vdn":
            learner = VDN(env.observation_space, env.action_space, options)
        case "mddpg":
            learner = MADDPG(env.observation_space, env.action_space, options)
        case _:
            learner = IDQN(env.observation_space, env.action_space, options)

    score = 0
    result_data = np.zeros((num_runs, num_episodes, 2))
    # perform runs
    for run_i in range(num_runs):
        print(f"performing run: {run_i}")
        # perfom each episode
        for episode_i in range(num_episodes):
            epsilon = max(min_epsilon, max_epsilon - (max_epsilon - min_epsilon) * (episode_i / (0.6 * num_episodes)))
            state = np.array(env.reset())
            done = [False for _ in range(env.n_agents)]
            with torch.no_grad():
                # hidden = learner.init_hidden()
                # perfoms steps according to the policy
                steps = 0
                score = 0
                while not all(done) and steps <= max_steps:
                    #action, hidden = learner.sample_action(torch.Tensor(state).unsqueeze(0), hidden, epsilon)
                    action = learner.sample_action(torch.Tensor(state).unsqueeze(0), epsilon)
                    action = action[0].data.cpu().numpy().tolist()
                    next_state, reward, done, info = env.step(action)
                    # store the step data in the replay buffer
                    learner.per_step_update(state, action, (np.array(reward)).tolist(), next_state, [int(all(done))])

                    # collect data for plotting
                    score += sum(reward)
                    steps += 1
                    # increment the state
                    state = np.array(next_state)

            learner.per_episode_update(episode_i=episode_i)
            # print(f"episode: {episode_i}/{num_episodes}: score: {score}, steps: {steps}")
            result_data[run_i][episode_i][0] = score
            result_data[run_i][episode_i][1] = steps

            # log relavent data at every log interval
            if (episode_i + 1) % log_interval == 0:
                print(f"episode: {episode_i}/{num_episodes}: score: {score}, steps: {steps}")
            #     test_score, test_steps = learner.test(test_env, test_episodes)
            #     print(f"episode: {episode_i}/{num_episodes}: score: {test_score}, steps: {test_steps}")
            #     result_data[episode_i][0] = score
            #     result_data[episode_i][1] = steps
            #     train_score = score / log_interval
            #     print("#{:<10}/{} episodes , avg train score : {:.1f}, test score: {:.1f}, test steps: {:.1f}, epsilon : {:.1f}"
            #             .format(episode_i, num_episodes, train_score, test_score, test_steps, epsilon))

    np.save(f"{results_dir}/{algo}_{num_episodes}_{num_runs}_{env_name}.npy", result_data)
    env.close()
    test_env.close()



if __name__ == '__main__':
    # gather arguments
    parser = argparse.ArgumentParser(description="MARL algorithm examples")
    parser.add_argument('--env-name', required=False, default='ma_gym:TrafficJunction4-v0')
    parser.add_argument('--algo', required=False, default='idqn')
    parser.add_argument('--results-dir', required=False, default='./results')
    parser.add_argument('--seed', type=int, default=1, required=False)
    parser.add_argument('--no-recurrent', action='store_true')
    parser.add_argument('--num-episodes', type=int, default=15000, required=False)
    parser.add_argument('--num-runs', type=int, default=10, required=False)
    
    #kwargs = {'env_name': 'ma_gym:Switch2-v1',
    # process arguments
    args = parser.parse_args()


    default_options = {
        'lr': 0.001,
        'batch_size': 32,
        'gamma': 0.99,
        'buffer_limit': 50000,
        'update_target_interval': 20,
        'warm_up_steps': 500,
        'update_iter': 10,
        'chunk_size': 10,  # if not recurrent, internally, we use chunk_size of 1 and no gru cell is used.
        'recurrent': False
    }


    kwargs= {
        'env_name': args.env_name,
        'algo': args.algo,
        'results_dir': args.results_dir,
        'log_interval': 100,
        'num_episodes': args.num_episodes,
        'num_runs': args.num_runs,
        'max_epsilon': 0.9,
        'min_epsilon': 0.1,
        'test_episodes': 5,
        'max_steps': 10000,
        'options': default_options
    }

    main(**kwargs)
