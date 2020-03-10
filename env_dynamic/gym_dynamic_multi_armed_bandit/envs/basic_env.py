import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class BasicEnv(gym.Env):
    """
    Description:

    Source:

    Observation:
        Type: Box(2)

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Choose option 0
        1	Choose option 1

        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
    Reward:

    Starting State:

    Episode Termination:
    """

    def __init__(self):
        # low_bound = np.array(np.repeat(0, 20), dtype=np.float32)
        # high_bound = np.array(np.repeat(np.finfo(np.float32).max, 20), dtype=np.float32)

        low_bound = np.array(np.repeat(0, 2), dtype=np.float32)
        high_bound = np.array(np.repeat(np.finfo(np.float32).max, 2), dtype=np.float32)

        # low_bound = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        # high_bound = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=np.float32)

        self.latent_state = np.random.binomial(1, 0.5, 1)[0]
        
        self.last_actions = None
        self.last_rewards = None

        self.max_step = None
        self.step_count = None
        self.reward_high = 50
        self.reward_low = 0

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low_bound, high_bound, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    @staticmethod
    def _get_new_latent_state(init_state):
        new_state = (1 - init_state) * np.random.binomial(1, 0.01, 1)[0] + \
                    init_state * np.random.binomial(1, 0.99, 1)[0]
        return new_state

    def _compute_reward(self, env_latent_state, action):
        reward = (action == env_latent_state) * np.random.normal(self.reward_high, 1, 1)[0] + \
                 (action != env_latent_state) * np.random.normal(self.reward_low, 1, 1)[0]
        return reward

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # compute the reward
        reward = self._compute_reward(self.latent_state, action)

        # dynamic latent state
        latent_state = self.latent_state
        self.latent_state = self._get_new_latent_state(latent_state)

        # stack the action
        self.last_actions.append(action)
        self.last_actions.pop(0)

        # stack the reward
        self.last_rewards.append(reward)
        self.last_rewards.pop(0)

        # self.state = [round(x / self.reward_high, 2) for x in self.last_rewards] + self.last_actions
        self.state = self.last_rewards[-1]/self.reward_high, self.last_actions[-1]
        # self.state = self.last_actions
        # self.state = np.array(round(self.last_rewards[-1] / self.reward_high), self.last_actions[-1])

        # def if the episode is done
        self.step_count += 1
        done = self.step_count >= self.max_step
        done = bool(done)

        return np.array(self.state), reward, done, {"latent_state": latent_state}

    def reset(self):
        self.latent_state = self._get_new_latent_state(self.latent_state)
        self.last_actions = list(np.random.binomial(1, 0.5, 10))
        self.last_rewards = [self._compute_reward(self._get_new_latent_state(self.latent_state),
                                                  self.last_actions[x]) for x in range(10)]

        self.max_step = 10
        self.step_count = 0
        self.state = np.array([self.last_actions[-1]/self.reward_high, self.last_rewards[-1]])
        # self.state = np.array(self.last_actions)
        # self.state = [round(x / self.reward_high, 2) for x in self.last_rewards] + self.last_actions
        # self.state = np.array(round(self.last_rewards[-1] / self.reward_high), self.last_actions[-1])

        return np.array(self.state)

    def render(self, mode='human'):
        return None

