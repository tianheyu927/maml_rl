
R7DOF_GOALS_LOCATION = '/home/rosen/maml_rl/saved_goals/R7DOF/goals_pool1_1000_40.pkl'
R7DOF_GOALS_LOCATION = '/home/rosen/maml_rl/saved_goals/R7DOF/goals_pool1_100_40.pkl'
R7DOF_GOALS_LOCATION_EC2 = '/root/code/rllab/saved_goals/R7DOF/goals_pool1.pkl'

ENV_OPTIONS = {
#    '':'R7DOF.xml',

}

default_r7dof_env_option = ''

EXPERT_TRAJ_LOCATION_DICT = {
    ".local": "/home/rosen/maml_rl/saved_expert_traj/R7DOF/R7-ET-individual_noise0.1/",
    ".local_100": "/home/rosen/maml_rl/saved_expert_traj/R7DOF/R7-ET-noise0.1-100/",
    ".local_200": "/home/rosen/maml_rl/saved_expert_traj/R7DOF/R7-ET-noise0.1-200/",
    ".ec2": "/root/code/rllab/saved_expert_traj/R7DOF/R7-ET-individual_noise0.1/",

}

