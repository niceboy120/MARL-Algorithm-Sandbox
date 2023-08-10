import unittest, random, string, math, sys

sys.path.append('.')
sys.path.append('./src')


from src.algos.idqn import IDQN
import gym

def random_word(length): 
   letters = string.ascii_lowercase
   rand_word = ''.join(random.choice(letters) for i in range(length))
   return rand_word


default_options = {
    'lr': 0.0005,
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
    'monitor': False
}

class TestIDQN(unittest.TestCase):

    def test_agent_num_equal_to_num_of_observations(self):
        env = gym.make('ma_gym:TrafficJunction4-v0')
        learner = IDQN(env.observation_space, env.action_space, default_options)
        # it should have an empty array of nodes and edges
        self.assertEqual(learner.q.num_agents, len(env.observation_space))

