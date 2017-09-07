# # TO DELETE THIS FILE:
#
# # We should be using batch_polopt and normal non-maml experiments to generate the trajectories
# # And then use batch_maml_polopt with the added options around goals_to_load, expert_trajs to load
# # the expert policy's trajectories.
#
#
# # Intended to work with batch_maml_polopt. Use elsewhere at your own risk
#
#
# from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
# from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
# from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
# from rllab.baselines.zero_baseline import ZeroBaseline
# from maml_examples.point_env_randgoal_expert import PointEnvRandGoalExpert
# from maml_examples.point_env_randgoal_oracle import PointEnvRandGoalOracle
# from rllab.envs.normalized_env import normalize
# from rllab.misc.instrument import stub, run_experiment_lite
# from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy
# from sandbox.rocky.tf.envs.base import TfEnv
#
# import tensorflow as tf
# import time
#
#
# fast_batch_size = 20  # 20 # 10 works for [0.1, 0.2], 20 doesn't improve much for [0,0.2]  #inner grad update size
# meta_batch_size = 40  # 40 @ 10 also works, but much less stable, 20 is fairly stable, 40 is more stable
# max_path_length = 100  # 100
# meta_step_size = 0.01
# l2loss_std_mult_list = [1.0]
# env_options = ["box"]
#
# policy = MAMLGaussianMLPPolicy # oracle because it sees the goal? Need to rewrite its functions
#
# def generateExpertTrajectories(
#     policy,
#     env,
#     training_n_itr=1000,
#     fast_batch_size = 20,  # 20 # 10 works for [0.1, 0.2], 20 doesn't improve much for [0,0.2]  #inner grad update size
#     meta_batch_size = 40, # 40 @ 10 also works, but much less stable, 20 is fairly stable, 40 is more stable
#     max_path_length = 100,  # 100
#     meta_step_size = 0.01,
#     env_options = None,
#     **kwargs,
# ):
#     # thought: isn't it easier to just pass the goal as an observation, both during training
#     # and during sampling
#     # and then to just scrub the resulting paths from the goal?
#     #test
#     algo = MAMLTRPO()
#     env = env.with_goal_as_obs
#     policy.train(algo=algo, n_itr=training_n_itr, goals = env.sample_goals(training_n_itr))
#
# k
#     output = {}
#     env=env.without_goal_as_obs
#     batch_size = fast_batch_size * max_path_length * meta_batch_size
#
#     filled_batch_size =
#     # I guess i can just sample as regular
#     for goal in range(meta_batch_size):
#         output[goal] = {}
#         for trajnum in range(fast_batch_size):
#             output[goal][trajnum] = []
#             init_obs = env.reset(newgoal)
#             obs = init_obs
#
#             while not done:
#                 step =policy.feedforward(obs)
#                 output[goal][trajnum].append(step)
#                 observation = step['obs']
#                 done = step['done']
#
#     #scrub the resulting paths from the goal
#     for goal in range(meta_batch_size) and for trajnum:
#         output[goal][trajnum] = scrub(env.spec, env.with_goal_as_obs.spec, output[goal][trajnum])
