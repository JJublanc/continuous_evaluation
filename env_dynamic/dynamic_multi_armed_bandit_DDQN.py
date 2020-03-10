import os
from rl_coach.base_parameters import TaskParameters
from rl_coach.core_types import EnvironmentSteps
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.agents.ddqn_agent import DDQNAgentParameters
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.schedules import LinearSchedule


####################
# Graph Scheduling #
####################
schedule = ScheduleParameters()
schedule.improve_steps = EnvironmentSteps(20000)
schedule.steps_between_evaluation_periods = EnvironmentSteps(1000)
schedule.evaluation_steps = EnvironmentSteps(1000)
schedule.heatup_steps = EnvironmentSteps(0)

#########
# Agent #
#########
agent_params = DDQNAgentParameters()
agent_params.network_wrappers['main'].learning_rate = 0.025
agent_params.exploration.epsilon_schedule = LinearSchedule(1, 0, 500)

###############
# Environment #
###############

level = 'gym_dynamic_multi_armed_bandit.envs:BasicEnv'
env_params = GymVectorEnvironment(level)

########################
# Create Graph Manager #
########################

graph_manager = BasicRLGraphManager(agent_params=agent_params, 
                                    env_params=env_params,
                                    schedule_params=schedule)


#######################
# add task parameters #
#######################

log_path = './experiments/log'  # training logs are saved
checkpoint_sec = 60  # checkpoints are used to restore the model
if not os.path.exists(log_path):
    os.makedirs(log_path)

task_parameters = TaskParameters(evaluate_only=False,
                                 experiment_path=log_path,
                                 checkpoint_save_secs=checkpoint_sec
                                 )

graph_manager.create_graph(task_parameters)


##################
# start training #
##################

