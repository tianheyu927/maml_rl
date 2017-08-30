from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.envs.mujoco.pusher_env import PusherEnv
import pickle

#from rllab.envs.gym_env import GymEnv
#from gym.envs.mujoco import mujoco_env

from rllab.misc.mujoco_render import pusher

from rllab.misc.instrument import VariantGenerator, variant

import glob
import random
local = True

DOCKER_CODE_DIR = "/root/code/rllab/"
LOCAL_CODE_DIR = '/home/rosen/rllab_copy/'
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

    if local:
        #xml_filepath = DOCKER_CODE_DIR + 'vendor/local_mujoco_models/pusher' + str(v['seed']) + '.xml'
        xml_filepath = DOCKER_CODE_DIR + 'vendor/local_mujoco_models/ensure_woodtable_distractor_pusher' + str(v['seed']) + '.xml'
    else:
        xml_filepath = DOCKER_CODE_DIR + 'vendor/mujoco_models/ensure_woodtable_distractor_pusher' + str(v['seed']) + '.xml'
    exp_log_info = {'xml': xml_filepath}

    gym_env = PusherEnv(xml_file=xml_filepath) #**{'xml_file': xml_filepath}) #, 'distractors': True})
    #gym_env = GymEnv('Pusher-v0', force_reset=True, record_video=False)
    # TODO - this is hacky...
    #mujoco_env.MujocoEnv.__init__(gym_env.env.env.env, xml_filepath, 5)
    env = TfEnv(normalize(gym_env))

    policy = GaussianMLPPolicy(
       name="policy",
       env_spec=env.spec,
       hidden_sizes=(128, 128)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        #load_policy='/home/rosen/rllab_copy/data/local/rllab-fixed-push-experts/pretraining_policy3/itr_300.pkl',
        #load_policy='vendor/pretraining_policy3/itr_300.pkl',
        baseline=baseline,
        batch_size=100*500,
        max_path_length=100,
        n_itr=301,
        discount=0.99,
        step_size=0.01,
        force_batch_sampler=True,
        # optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
        exp_log_info=exp_log_info,
    )
    algo.train()

for v in variants:

    run_experiment_lite(
        #algo.train(),
        run_task,
        # Number of parallel workers for sampling
        n_parallel=8,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="gap",
        snapshot_gap=20,
        exp_prefix='rllab_fixed_push_experts',
        python_command='python3',
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=79,
        variant=v,
        # mode="ec2",
        # mode="local_docker",
        mode=mode, #'local',
        confirm_remote=False,
        sync_s3_pkl=True,
        # plot=True,
    )
