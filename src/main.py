
import argparse

import gym
import numpy as np
import torch
import torch.nn.functional as F

from algos.vdn import VDN
from algos.idqn import IDQN
from algos.commnet import COMMNET
from algos.maddpg import MADDPG
from algos.qmix import QMix

#from algos.commnet import DDPG


USE_WANDB = True


def main(env_name, algo, results_dir, log_interval, num_episodes, num_runs, max_epsilon, min_epsilon, test_episodes, max_steps, options=None):
    env = gym.make(env_name)
    test_env = gym.make(env_name)

    # create networks
    print(f"using algorithm: {algo} on environment: {env_name}")
    match algo:
        case "vdn":
            learner = VDN(env.observation_space, env.action_space)
        case "commnet":
            learner = COMMNET(env.observation_space, env.action_space)
        case "maddpg":
            learner = MADDPG(env.observation_space, env.action_space)
        case "qmix":
            learner = QMix(env.observation_space, env.action_space)
        case _:
            learner = IDQN(env.observation_space, env.action_space)

    score = 0
    #result_data = np.zeros((num_runs, num_episodes, 2))
    # result_data = np.zeros((num_episodes // log_interval, 2))
    result_data = torch.zeros([num_episodes // log_interval, 2])
    if torch.cuda.is_available():  
        dev = f'cuda:0' 
        print(f"Using Device: {torch.cuda.get_device_name(0)}")
    else:  
        dev = 'cpu'  
    device = torch.device(dev)  
    result_data = result_data.to(device)
    # perform runs
    # for run_i in range(num_runs):
    #     print(f"performing run: {run_i}")
    # perfom each episode
    for episode_i in range(num_episodes):
        epsilon = max(min_epsilon, max_epsilon - (max_epsilon - min_epsilon) * (episode_i / (0.6 * num_episodes)))
        state = np.array(env.reset())
        done = [False for _ in range(env.n_agents)]
        # perfoms steps according to the policy
        steps = 0
        score = 0
        with torch.no_grad():
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

        # log relavent data at every log interval
        if (episode_i + 1) % log_interval == 0:
            test_score, test_steps = learner.test(test_env, test_episodes)

            train_score = score / log_interval
            print(f"episode: {episode_i}/{num_episodes}: train_score: {train_score}, train_steps: {train_steps}, score: {test_score}, steps: {test_steps}")
            if USE_WANDB:
                wandb.log({'episode': episode_i, 'test-score': test_score, 'buffer-size': learner.memory.size(),
                           'epsilon': epsilon, 'train-score': train_score})
            else:
                result_data[episode_i // log_interval, 0] = test_score
                result_data[episode_i // log_interval, 1] = test_steps
            score, steps = 0, 0

    if not USE_WANDB:
        np.save(f"{results_dir}/{algo}_{num_episodes}_{num_runs}_{env_name}.npy", result_data.cpu())
    env.close()
    test_env.close()

import threading

if __name__ == '__main__':
    # gather arguments
    parser = argparse.ArgumentParser(description="MARL algorithm examples")
    parser.add_argument('--env-name', required=False, default='ma_gym:TrafficJunction4-v0')
    parser.add_argument('--algo', required=False, default='idqn')
    parser.add_argument('--results-dir', required=False, default='./results')
    parser.add_argument('--seed', type=int, default=1, required=False)
    parser.add_argument('--no-recurrent', action='store_true')
    parser.add_argument('--num-episodes', type=int, default=15000, required=False)
    parser.add_argument('--num-tests', type=int, default=5, required=False)
    parser.add_argument('--num-runs', type=int, default=10, required=False)
    
    # process arguments
    args = parser.parse_args()


    # create a new thread for each algorithm in each run
    threads = []
    for i in range(args.num_runs):
        for env in ["ma_gym:TrafficJunction4-v0"]:
            for algo in ["idqn", "vdn", "commnet"]:
                kwargs= {
                    'env_name': args.env_name,
                    'algo': algo,
                    'results_dir': args.results_dir,
                    'log_interval': 100,
                    'num_episodes': args.num_episodes,
                    'max_epsilon': 0.9,
                    'min_epsilon': 0.1,
                    'test_episodes': args.num_tests,
                    'max_steps': 10000,
                }
                # activate wandb if necessary
                if USE_WANDB:
                    import wandb
                    wandb.init(project='marl-algos', config={'algo': args.algo, **kwargs})

                t = threading.Thread(target=main, name=f"{algo}_{env}_{i}", args=(kwargs,))
                threads.append(t)

    for t in threads:
        t.start()

