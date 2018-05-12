
# PUSHER_GOALS_LOCATION = '/home/rosen/maml_rl/saved_goals/PUSHER/goals_pool1_1000_40.pkl'
# PUSHER_GOALS_LOCATION = '/home/rosen/maml_rl/saved_goals/PUSHER/goals_pool1_100_40.pkl'
# PUSHER_GOALS_LOCATION_EC2 = '/root/code/rllab/saved_goals/PUSHER/goals_pool1.pkl'

PUSHER_GOALS_LOCATION = '/home/rosen/maml_rl/saved_goals/PUSHER/goals_pool1_200_40.pkl'


ENV_OPTIONS = {
#    '':'PUSHER.xml',

}

default_pusher_env_option = ''

EXPERT_TRAJ_LOCATION_DICT = {
    ".local":          "/home/rosen/maml_rl/saved_expert_traj/PUSHER/main_1/",
#     ".local_1000_40":  "/home/rosen/maml_rl/saved_expert_traj/PUSHER/R7-ET-individual_noise0.1/",
#     ".local_100_40_1":   "/home/rosen/maml_rl/saved_expert_traj/PUSHER/R7-ET-noise0.1-100-40/",  # 100 goals, 40 goals per itr, 40 et per goal
#     ".local_200_40_1":   "/home/rosen/maml_rl/saved_expert_traj/PUSHER/R7-ET-noise0.1-200-40-1/", # 200 goals, 40 goals per itr, 40 et per goal
#     ".local_200_40_1dist":   "/home/rosen/maml_rl/saved_expert_traj/PUSHER/R7-ET-noise0.1-200-40-1dist/", # 200 goals, 40 goals per itr, 40 et per goal
#     ".local_200_40_1_1":   "/home/rosen/maml_rl/saved_expert_traj/PUSHER/R7-ET-noise0.1-200-40-1_1/", # 200 goals, 40 goals per itr, 40 et per goal
#     ".local_200_40_1dist_1":   "/home/rosen/maml_rl/saved_expert_traj/PUSHER/R7-ET-noise0.1-200-40-1dist_1/", # 200 goals, 40 goals per itr, 40 et per goal
#     ".local_200_40_1_5":   "/home/rosen/maml_rl/saved_expert_traj/PUSHER/R7-ET-noise0.1-200-40-1_5/", # 200 goals, 40 goals per itr, 40 et per goal
#     ".local_200_40_1dist_5":   "/home/rosen/maml_rl/saved_expert_traj/PUSHER/R7-ET-noise0.1-200-40-1dist_5/", # 200 goals, 40 goals per itr, 40 et per goal
#     ".local_200_40_2":   "/home/rosen/maml_rl/saved_expert_traj/PUSHER/R7-ET-noise0.1-200-40-2/", # 200 goals, 40 goals per itr, 40 et per goal
#     ".local_200_40_3":   "/home/rosen/maml_rl/saved_expert_traj/PUSHER/R7-ET-noise0.1-200-40-3/", # 200 goals, 40 goals per itr, 40 et per goal
#     ".local_200_40_4":   "/home/rosen/maml_rl/saved_expert_traj/PUSHER/R7-ET-noise0.1-200-40-4/", # 200 goals, 40 goals per itr, 40 et per goal
#     ".ec2_1000_40":      "/root/code/rllab/saved_expert_traj/PUSHER/R7-ET-individual_noise0.1/",
#     ".ec2_100_40_1":     "/root/code/rllab/saved_expert_traj/PUSHER/R7-ET-noise0.1-100-40/", # 100 goals, 40 goals per itr, 40 et per goal
#     ".ec2_200_40_1":     "/root/code/rllab/saved_expert_traj/PUSHER/R7-ET-noise0.1-200-40/", # 200 goals, 40 goals per itr, 40 et per goal
#     ".ec2_200_40_1dist": "/home/rosen/maml_rl/saved_expert_traj/PUSHER/R7-ET-noise0.1-200-40/",
# # 200 goals, 40 goals per itr, 40 et per goal
#     ".ec2_200_40_1_1": "/home/rosen/maml_rl/saved_expert_traj/PUSHER/R7-ET-noise0.1-200-40/",
# # 200 goals, 40 goals per itr, 40 et per goal
#     ".ec2_200_40_1dist_1": "/home/rosen/maml_rl/saved_expert_traj/PUSHER/R7-ET-noise0.1-200-40/",
# # 200 goals, 40 goals per itr, 40 et per goal
#     ".ec2_200_40_1_5": "/home/rosen/maml_rl/saved_expert_traj/PUSHER/R7-ET-noise0.1-200-40/",
# # 200 goals, 40 goals per itr, 40 et per goal
#     ".ec2_200_40_1dist_5": "/home/rosen/maml_rl/saved_expert_traj/PUSHER/R7-ET-noise0.1-200-40/",
# # 200 goals, 40 goals per itr, 40 et per goal
#     ".ec2_200_40_2":     "/root/code/rllab/saved_expert_traj/PUSHER/R7-ET-noise0.1-200-40-2/",
#     ".ec2_200_40_3":     "/root/code/rllab/saved_expert_traj/PUSHER/R7-ET-noise0.1-200-40-3/",
#     ".ec2_200_40_4":     "/root/code/rllab/saved_expert_traj/PUSHER/R7-ET-noise0.1-200-40-4/",

}

