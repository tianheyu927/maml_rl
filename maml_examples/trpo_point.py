
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.algos.vpg import VPG
from sandbox.rocky.tf.policies.minimal_gauss_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from maml_examples.point_env_randgoal_oracle import PointEnvRandGoalOracle
from maml_examples.point_env_randgoal_oracle_noise import PointEnvRandGoalOracleNoise
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

import tensorflow as tf
from maml_examples.point_vars import POINT_GOALS_LOCATION, EXPERT_TRAJ_LOCATION_DICT



env = TfEnv(normalize(PointEnvRandGoalOracleNoise()))

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
    batch_size=400*100,
    batch_size_expert_traj=40*100,
    max_path_length=100,
    start_itr=-100,
    n_itr=1000,
    discount=0.99,
    step_size=0.01,
    #plot=True,
    action_noise_train=0.0,
    action_noise_test=0.00,
    save_expert_traj_dir=EXPERT_TRAJ_LOCATION_DICT[".local"],
    goals_pool_to_load=POINT_GOALS_LOCATION[".local"],

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
    exp_prefix='PR_ET_E1',
    exp_name='PR_ET_E1',
    #plot=True,
)
