import unittest, random, string, math, sys

sys.path.append('.')
sys.path.append('./src')


from src.algos.commnet_range import COMMNET_RANGE
import torch
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

class TestCOMMNET_RANGE(unittest.TestCase):

    def test_agent_num_equal_to_num_of_observations(self):
        env = gym.make('ma_gym:Switch4-v0')
        N = 10 # size of neighborhood
        learner = COMMNET_RANGE(env.observation_space, env.action_space, neighborhood=N)
        # it should have an empty array of nodes and edges
        self.assertEqual(learner.q.num_agents, len(env.observation_space))


    """def test_obs_to_pos(self):
        env = gym.make('ma_gym:Switch4-v0')
        N = 10 # size of neighborhood
        learner = COMMNET_RANGE(env.observation_space, env.action_space, neighborhood=N)
        q_net = learner.q

        batch_size = 4
        num_agents = 5
        obs_size = 2 # position

        mock_obs = torch.zeros(batch_size, num_agents, obs_size)

        agent_pos = q_net.obs_to_pos(mock_obs)
        

        # it should have an empty array of nodes and edges
        self.assertEqual(learner.q.num_agents, len(env.observation_space))"""


    def test_get_nei_mask_returns_1_for_proximal_and_0_for_non_proximal_agents(self):
        env = gym.make('ma_gym:Switch4-v0')
        N = 1 # size of neighborhood
        learner = COMMNET_RANGE(env.observation_space, env.action_space, neighborhood=N)
        q_net = learner.q
        batch_size = 2
        
        mock_obs = torch.tensor([
            [[0.5, 0.5], [0.6, 0.5], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.7, 0.7], [0.7, 0.8]]
        ])
        mask = q_net.get_nei_mask(mock_obs)

        #self.assertEqual(learner.q.num_agents, len(env.observation_space))
        self.assertEqual(mask[0, 0, 1], 1) # batch 0
        self.assertEqual(mask[0, 0, 2], 0) # batch 0
        self.assertEqual(mask[1, 0, 1], 0) # batch 1
        self.assertEqual(mask[1, 2, 3], 1) # batch 1


    def test_get_nei_mask_returns_0_for_own_agent(self):
        env = gym.make('ma_gym:Switch4-v0')
        N = 1 # size of neighborhood
        learner = COMMNET_RANGE(env.observation_space, env.action_space, neighborhood=N)
        q_net = learner.q
        batch_size = 2
        num_agents = 4
        
        mock_obs = torch.tensor([
            [[0.5, 0.5], [0.6, 0.5], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.7, 0.7], [0.7, 0.8]]
        ])
        mask = q_net.get_nei_mask(mock_obs)

        #self.assertEqual(learner.q.num_agents, len(env.observation_space))
        for b in range(batch_size):
            for i in range(num_agents):
                self.assertEqual(mask[b, i, i], 0)


    def test_get_communication(self):
        env = gym.make('ma_gym:Switch4-v0')
        N = 10 # size of neighborhood
        learner = COMMNET_RANGE(env.observation_space, env.action_space, neighborhood=N)
        q_net = learner.q

        batch_size = 2
        num_agents = 4
        obs_size = 2 # position

        
        mock_mask = torch.tensor([
            [
                [0., 0., 0., 0.], 
                [0., 0., 0., 0.], 
                [0., 0., 0., 1.], 
                [0., 0., 1., 0.]
            ],
            [
                [0., 1., 1., 0.], 
                [1., 0., 0., 0.], 
                [1., 0., 0., 0.], 
                [0., 0., 0., 0.]
            ],
        ])
        mock_hidden = torch.tensor([
            [[2.0, 3.0, 1.0], [1.0, 5.0, 1.0], [2.0, 3.0, 1.0], [1.0, 5.0, 1.0]],
            [[5.0, 1.0, 1.0], [2.0, 0.0, 1.0], [2.0, 3.0, 1.0], [1.0, 5.0, 1.0]]
        ])
        
        # co [2.0, 3.0, 1.0], [1.0, 5.0, 1.0]

        comm = q_net.get_communications(mock_mask, mock_hidden)
        print(comm)
        
        agent_i = 4

        # it should have an empty array of nodes and edges
        #self.assertEqual(learner.q.num_agents, len(env.observation_space))
        b = 0
        self.assertEqual(comm[b, 0, :].tolist(), [0., 0., 0.]) # batch 0
        self.assertEqual(comm[b, 2, :].tolist(), mock_hidden[b, 3, :].tolist()) # batch 0
        self.assertNotEqual(comm[0, 1, :].tolist(), mock_hidden[b, 3, :].tolist()) # batch 0
        self.assertEqual(comm[b, 3, :].tolist(), mock_hidden[b, 2, :].tolist()) # batch 0

        b = 1
        avg_h = torch.mul(torch.add(mock_hidden[b, 1, :], mock_hidden[b, 2, :]), 1/2)
        self.assertEqual(comm[b, 0, :].tolist(), avg_h.tolist()) # batch 1
        self.assertEqual(comm[b, 3, :].tolist(), [0., 0., 0.]) # batch 1



