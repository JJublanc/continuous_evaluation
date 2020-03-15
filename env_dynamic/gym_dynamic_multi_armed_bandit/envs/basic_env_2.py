import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class BasicEnv2(gym.Env):
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

        self.latent_state_A = 0
        self.latent_state_B = 0
        
        self.last_actions = [0]
        self.last_rewards = [0]

        self.max_step = None
        self.step_count = None
        self.time_count = 0

        self.reward_high = 25
        self.reward_low = -5

        self.period_B = 500
        self.period_A = 400

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low_bound, high_bound, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    @staticmethod
    def get_season_value(i, period, std=2):
        return 10 + 10 * np.cos(2 * np.pi * (i % period) / period) + np.random.normal(0, std, 1)[0]

    def _compute_reward(self, action):
        reward = action * self.latent_state_A + (1 - action) * self.latent_state_B
        return reward

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # compute the reward
        reward = self._compute_reward(action)

        # dynamic latent state
        self.latent_state_A = self.get_season_value(self.time_count, self.period_A, std=2)
        self.latent_state_B = self.get_season_value(self.time_count, self.period_B, std=2)

        # stack the action
        self.last_actions.append(action)
        self.last_actions.pop(0)

        # stack the reward
        self.last_rewards.append(reward)
        self.last_rewards.pop(0)

        # self.state = [round(x / self.reward_high, 2) for x in self.last_rewards] + self.last_actions
        scaled_reward = (self.last_rewards[-1] - self.reward_low) / (self.reward_high - self.reward_low)
        self.state = scaled_reward, self.last_actions[-1]
        # self.state = self.last_actions
        # self.state = np.array(round(self.last_rewards[-1] / self.reward_high), self.last_actions[-1])

        # def if the episode is done
        self.step_count += 1
        self.time_count += 1
        done = self.step_count >= self.max_step
        done = bool(done)

        return np.array(self.state), reward, done, {"latent_state_A": self.latent_state_A,
                                                    "latent_state_B": self.latent_state_B}

    def reset(self):
        self.max_step = 10
        self.step_count = 0

        scaled_reward = (self.last_rewards[-1] - self.reward_low) / (self.reward_high - self.reward_low)
        self.state = scaled_reward, self.last_actions[-1]

        return np.array(self.state)

    def render(self, mode='human'):
        return None

