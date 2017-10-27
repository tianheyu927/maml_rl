
CHEETAH_GOALS_LOCATION = '/home/rosen/maml_rl/saved_goals/cheetah/goals_pool1.pkl'
CHEETAH_GOALS_LOCATION_EC2 = '/root/code/rllab/saved_goals/cheetah/goals_pool1.pkl'

ENV_OPTIONS = {
    '':'half_cheetah.xml',

}

default_cheetah_env_option = ''

EXPERT_TRAJ_LOCATION_DICT = {
    ".local": "/home/rosen/maml_rl/saved_expert_traj/cheetah/CH-ET4-individual/",
    ".ec2": "/root/code/rllab/saved_expert_traj/cheetah/CH-ET4-individual/",
    ".local.noise0.1": "/home/rosen/maml_rl/saved_expert_traj/cheetah/CH-ET-E1.6-noise0.1/",
    ".ec2.noise0.1": "/root/code/rllab/saved_expert_traj/cheetah/CH-ET-E1.6-noise0.1/",
}

