from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.envs.mujoco.half_cheetah_env_rand import HalfCheetahEnvRand
from rllab.envs.mujoco.half_cheetah_env_oracle import HalfCheetahEnvOracle
from maml_examples.cheetah_vars import EXPERT_TRAJ_LOCATION_DICT, ENV_OPTIONS, GOALS_LOCATION, \
    default_cheetah_env_option
import pickle

#from rllab.envs.gym_env import GymEnv
#from gym.envs.mujoco import mujoco_env
import tensorflow as tf
from rllab.misc.mujoco_render import pusher

from rllab.misc.instrument import VariantGenerator, variant

import glob
import random
local = True

DOCKER_CODE_DIR = "/root/code/rllab/"
LOCAL_CODE_DIR = '/home/rosen/maml_rl_data/'
if local:
    DOCKER_CODE_DIR = LOCAL_CODE_DIR
    mode = 'local'
else:
    mode = 'ec2'

class VG(VariantGenerator):
    @variant
    def seed(self):
        # 1003 is pretraining policy 3.
        return [1000]
        #return range(1,101) #102)


variants = VG().variants()

env_option = default_cheetah_env_option

def run_task(v):
    env = TfEnv(normalize(HalfCheetahEnvOracle()))
    policy = GaussianMLPPolicy(
       name="policy",
       env_spec=env.spec,
       hidden_nonlinearity=tf.nn.relu,
       hidden_sizes=(256, 256),
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        # policy=None,
       # load_policy='/home/rosen/maml_rl/data/local/CH-ET-D5.6/CH_ET_D5.6_2017_10_25_09_42_27_0001/itr_-720.pkl',  # if you want to use this you need to comment out the definition of policy above
        baseline=baseline,
        batch_size=400*200,  # we divide this by #envs on every iteration
        batch_size_expert_traj=40*200,
        max_path_length=200,
        start_itr=-400,
        n_itr=1001,  # actually last iteration number, not total iterations
        discount=0.99,
        step_size=0.01,  # 0.01
        force_batch_sampler=True,
        # optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)),
        action_noise_train=0.0,
        action_noise_test=0.0,
        expert_traj_itrs_to_pickle=list(range(0, 801)),
        save_expert_traj_dir=EXPERT_TRAJ_LOCATION_DICT[env_option],
        goals_to_load=GOALS_LOCATION,
    )
    algo.train()

for v in variants:

    run_experiment_lite(
        #algo.train(),
        run_task,
        # Number of parallel workers for sampling
        n_parallel=10, #10,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="gap",
        snapshot_gap=20,
        exp_prefix='CH_ET_D5.6',
        python_command='python3',
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=79,
        variant=v,
        # mode="ec2",
        # mode="local_docker",
        mode='local',
        sync_s3_pkl=True,
        # plot=True,
    )

