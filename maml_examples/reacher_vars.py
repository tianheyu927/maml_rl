
# GOALS_LOCATION = '/home/kevin/maml_rl_data/saved_goals/reach/goals_E1_test1.pkl'
# GOALS_LOCATION = '/home/kevin/maml_rl_data/saved_goals/reach/g200nfj_goals_pool.pkl'
# GOALS_LOCATION = '/home/kevin/maml_rl_data/saved_goals/reach/g200nfj_goals_pool_100.pkl'
GOALS_LOCATION = '/home/kevin/maml_rl_data/saved_goals/reach/g200nfj_goals_pool_g1000_40.pkl'

ENV_OPTIONS = {
    '':'reacher.xml',
    'g200l0.05':'reacher_versions/reacher_gear200_limit0.05.xml',
    'g200l0.15':'reacher_versions/reacher_gear200_limit0.15.xml',
    'g200l0.25':'reacher_versions/reacher_gear200_limit0.25.xml',
    'g200l0.35':'reacher_versions/reacher_gear200_limit0.35.xml',
    'g200nfj':'reacher_versions/reacher_gear200_nofreejoint.xml',
    'g10':'reacher_versions/reacher_gear10.xml', # gear 10 seems to have too much friction
    'g1nfj': 'reacher_versions/reacher_gear1_nofreejoint.xml',
    'g2nfj': 'reacher_versions/reacher_gear2_nofreejoint.xml',
    'g3nfj': 'reacher_versions/reacher_gear3_nofreejoint.xml',
    'g10nfj': 'reacher_versions/reacher_gear10_nofreejoint.xml',
    'g1':'reacher_versions/reacher_gear1.xml',
    'g200nfj.st':'reacher_versions/reacher_gear200_nofreejoint_stubby.xml',
    'g200l0.05nfj.st':'reacher_versions/reacher_gear200_l0.05_nofreejoint_stubby.xml',
    'g50l0.2nfj.st':'reacher_versions/reacher_gear50_l0.2_nofreejoint_stubby.xml',
    'g100l0.25nfj':'reacher_versions/reacher_gear100_l0.25_nofreejoint.xml',
    'g200l0.25nfj':'reacher_versions/reacher_gear200_l0.25_nofreejoint.xml',
    'g50l0.25nfj.st':'reacher_versions/reacher_gear50_l0.25_nofreejoint_stubby.xml',
}

default_reacher_env_option = 'g200nfj'

EXPERT_TRAJ_LOCATION_DICT = {
    "":"", # blank means we haven't recorded expert traj yet
    # "g200nfj.ec2":"/root/code/rllab/saved_expert_traj/reacher/test_E1_test2_noise0.1/",
    # "g200nfj.local":"/home/kevin/maml_rl/saved_expert_traj/reacher_g200nfj/test_E1_test2_noise0.1/",
    "g200nfj.ec2":"/root/code/rllab/saved_expert_traj/test_C1_randgoal_noise0.1_MINI/",
    "g200nfj.local":"/home/kevin/maml_rl/saved_expert_traj/test_C1_randgoal_noise0.1_MINI/",
    # "g200nfj.local":"/home/kevin/maml_rl_data/test_B1_randgoal_noise0.1/",
    # note, .small didn't seem to work
    # "g200nfj.local.small":"/home/kevin/maml_rl_data/saved_expert_traj/reacher_g200nfj/test_E3.3_test1_noise0.1_small/",
    # "g200nfj.ec2.small":"/root/code/rllab/saved_expert_traj/reacher_g200nfj/test_E3.3_200_40_noise0.1_small/",
    "g200nfj.ec2.small": "/root/code/rllab/saved_expert_traj/reacher_g200nfj/test_E3.3_1000_40_d40_noise0.1/",
    # "g200nfj.local.small":"/home/kevin/maml_rl_data/saved_expert_traj/reacher_g200nfj/test_E3.3_200_40_noise0.1_small/",
    "g200nfj.local.small":"/home/kevin/maml_rl_data/saved_expert_traj/reacher_g200nfj/test_E3.3_1000_40_d40_noise0.1/",
    "g200nfj.local.nonoise":"/home/kevin/maml_rl_data/saved_expert_traj/reacher/test_E1_test3_noise0.0/",
  #  "g200nfj":"/home/kevin/maml_rl_data/saved_expert_traj/reacher_g200nfj/test_B1_randgoal_noise0.1/",
    'g10nfj':"/home/kevin/maml_rl_data/saved_expert_traj/reacher10/test7/",
}