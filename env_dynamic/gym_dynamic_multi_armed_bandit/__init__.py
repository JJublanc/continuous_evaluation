from gym.envs.registration import register

register(
    id='basic-v0',
    entry_point='gym_dynamic_multi_armed_bandit.envs:BasicEnv'
    )

