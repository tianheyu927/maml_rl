
#EXPERT_TRAJ_LOCATION = "/home/rosen/maml_rl_data/saved_expert_traj/reacher10/test7/"
GOALS_LOCATION = '/home/rosen/maml_rl_data/saved_goals/reach/B1_saved_goals_10_3.pkl'

ENV_OPTIONS ={
    '':'/home/rosen/gym/gym/envs/mujoco/assets/reacher.xml',
    'g200l0.05':'/home/rosen/gym/gym/envs/mujoco/assets/reacher_gear200_limit0.05.xml',
    'g200l0.15':'/home/rosen/gym/gym/envs/mujoco/assets/reacher_gear200_limit0.15.xml',
    'g200l0.25':'/home/rosen/gym/gym/envs/mujoco/assets/reacher_gear200_limit0.25.xml',
    'g200l0.35':'/home/rosen/gym/gym/envs/mujoco/assets/reacher_gear200_limit0.35.xml',
    'g200nfj':'/home/rosen/gym/gym/envs/mujoco/assets/reacher_gear200_nofreejoint.xml',
    'g10':'/home/rosen/gym/gym/envs/mujoco/assets/reacher_gear10.xml', # gear 10 seems to have too much friction
    'g1nfj': '/home/rosen/gym/gym/envs/mujoco/assets/reacher_gear1_nofreejoint.xml',
    'g2nfj': '/home/rosen/gym/gym/envs/mujoco/assets/reacher_gear2_nofreejoint.xml',
    'g3nfj': '/home/rosen/gym/gym/envs/mujoco/assets/reacher_gear3_nofreejoint.xml',
    'g10nfj': '/home/rosen/gym/gym/envs/mujoco/assets/reacher_gear10_nofreejoint.xml',
    'g1':'/home/rosen/gym/gym/envs/mujoco/assets/reacher_gear1.xml',
    'g200nfj.st':'/home/rosen/gym/gym/envs/mujoco/assets/reacher_gear200_nofreejoint_stubby.xml',
    'g200l0.05nfj.st':'/home/rosen/gym/gym/envs/mujoco/assets/reacher_gear200_l0.05_nofreejoint_stubby.xml',
    'g50l0.2nfj.st':'/home/rosen/gym/gym/envs/mujoco/assets/reacher_gear50_l0.2_nofreejoint_stubby.xml',
    'g100l0.25nfj':'/home/rosen/gym/gym/envs/mujoco/assets/reacher_gear100_l0.25_nofreejoint.xml',
    'g200l0.25nfj':'/home/rosen/gym/gym/envs/mujoco/assets/reacher_gear200_l0.25_nofreejoint.xml',
    'g50l0.25nfj.st':'/home/rosen/gym/gym/envs/mujoco/assets/reacher_gear50_l0.25_nofreejoint_stubby.xml',
}
EXPERT_TRAJ_LOCATION_DICT={
    "":"", # blank means we haven't recorded expert traj yet
    "g200nfj":"/home/rosen/maml_rl_data/saved_expert_traj/g200nfj/test1/",
    'g10nfj':"/home/rosen/maml_rl_data/saved_expert_traj/reacher10/test7/",
}