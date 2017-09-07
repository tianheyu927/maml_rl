from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.envs.mujoco.pusher_env import PusherEnv
from maml_examples.reacher_env import ReacherEnv

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
LOCAL_CODE_DIR = '/home/rosen/maml_rl/'
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


def run_task(v):


    env = TfEnv(normalize(ReacherEnv()))

    policy = GaussianMLPPolicy(
       name="policy",
       env_spec=env.spec,
       hidden_nonlinearity=tf.nn.relu,
       hidden_sizes=(128, 128)
    )



    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        #load_policy='/home/rosen/maml_rl/data/local/rllab-fixed-reach-experts/rllab_fixed_reach_experts_2017_08_24_19_24_05_0001/itr_280.pkl',
        #load_policy='vendor/pretraining_policy3/itr_300.pkl',
        baseline=baseline,
        batch_size=25*100, #100*500,
        max_path_length=100,
        n_itr=301, #301,
        discount=0.99,
        step_size=0.01,
        force_batch_sampler=True,
        # optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)),
        save_expert_trajectories="/home/rosen/maml_rl/saved_expert_traj/test_reach6fast3.pkl"
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
        exp_prefix='rllab_fixed_reach_experts',
        python_command='python3',
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=79,
        variant=v,
        # mode="ec2",
        # mode="local_docker",
        mode='local',
        confirm_remote=False,
        sync_s3_pkl=True,
        # plot=True,
    )

