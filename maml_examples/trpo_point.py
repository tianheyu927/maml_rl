
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.algos.vpg import VPG
from sandbox.rocky.tf.policies.minimal_gauss_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from maml_examples.point_env_randgoal_oracle import PointEnvRandGoalOracle
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

import tensorflow as tf

env = TfEnv(normalize(PointEnvRandGoalOracle()))

policy = GaussianMLPPolicy(
    name='policy',
    env_spec=env.spec,
    hidden_nonlinearity=tf.nn.relu,
    hidden_sizes=(100, 100)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=20*100,
    max_path_length=100,
    start_itr=-100,
    n_itr=31,
    discount=0.99,
    step_size=0.01,
    #plot=True,
    itrs_to_pickle=list(range(0, 31)),
    save_expert_traj_dir="/home/rosen/maml_rl/saved_expert_traj/9_6_test4/",
    goals_to_load='/home/rosen/maml_rl/saved_goals/point/saved_goals_9_6.pkl',

)

run_experiment_lite(
    algo.train(),
    # Number of parallel workers for sampling
    n_parallel=4,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    exp_prefix='trpo_point100_oracle',
    exp_name='oracleenv2',
    #plot=True,
)
