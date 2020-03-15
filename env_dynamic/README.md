# To install the environements Gym proceed as follow

In a terminal go to the root of the project and launch the following command

`
pip install -e ./env_dynamic
`

If you want to train a DDQN agent to adopt the best actions in a basic multiarmed 
bandit environment you can run the following command

`
python dynamic_multi_armed_bandit_DDQN.py
`

The checkpoints and results are stored in the folder `./experiments`