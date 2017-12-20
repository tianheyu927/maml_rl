from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
from sandbox.rocky.tf.algos.maml_il import MAMLIL
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

from rllab.envs.gym_env import GymEnv
from maml_examples.reacher_env import ReacherEnv
from rllab.envs.mujoco.pusher_env import PusherEnv
from rllab.envs.mujoco.half_cheetah_env_rand import HalfCheetahEnvRand
from maml_examples.cheetah_vars import EXPERT_TRAJ_LOCATION_DICT, ENV_OPTIONS, default_cheetah_env_option
#from examples.trpo_push_obj import

import tensorflow as tf
import time

beta_adam_steps_list = [(1,125)]

fast_learning_rates = [0.1]
baselines = ['linear']
env_option = ''
mode = "local"

fast_batch_size = 20  # 20 # 10 works for [0.1, 0.2], 20 doesn't improve much for [0,0.2]  #inner grad update size
meta_batch_size = 40  # 40 @ 10 also works, but much less stable, 20 is fairly stable, 40 is more stable
max_path_length = 200  # 100
num_grad_updates = 1
meta_step_size = 0.01
pre_std_modifier_list = [1.0]
post_std_modifier_train_list = [0.00001]
post_std_modifier_test_list = [0.00001]
l2loss_std_mult_list = [1.0]

use_maml = True

for l2loss_std_mult in l2loss_std_mult_list:
    for post_std_modifier_train in post_std_modifier_train_list:
        for post_std_modifier_test in post_std_modifier_test_list:
            for pre_std_modifier in pre_std_modifier_list:
                for fast_learning_rate in fast_learning_rates:
                    for beta_steps, adam_steps in beta_adam_steps_list:
                        for bas in baselines:
                            stub(globals())

                            seed = 1
                            env = TfEnv(normalize(HalfCheetahEnvRand()))

                            policy = MAMLGaussianMLPPolicy(
                                name="policy",
                                env_spec=env.spec,
                                grad_step_size=fast_learning_rate,
                                hidden_nonlinearity=tf.nn.relu,
                                hidden_sizes=(200, 200),
                                std_modifier=pre_std_modifier,
                            )
                            if bas == 'zero':
                                baseline = ZeroBaseline(env_spec=env.spec)
                            elif 'linear' in bas:
                                baseline = LinearFeatureBaseline(env_spec=env.spec)
                            else:
                                baseline = GaussianMLPBaseline(env_spec=env.spec)
                            algo = MAMLIL(
                                env=env,
                                policy=policy,
                                baseline=baseline,
                                batch_size=fast_batch_size,  # number of trajs for alpha grad update
                                max_path_length=max_path_length,
                                meta_batch_size=meta_batch_size,  # number of tasks sampled for beta grad update
                                num_grad_updates=num_grad_updates,  # number of alpha grad updates
                                n_itr=800, #100
                                use_maml=use_maml,
                                use_pooled_goals=True,
                                step_size=meta_step_size,
                                plot=False,
                                beta_steps=beta_steps,
                                adam_steps=adam_steps,
                                pre_std_modifier=pre_std_modifier,
                                l2loss_std_mult=l2loss_std_mult,
                                post_std_modifier_train=post_std_modifier_train,
                                post_std_modifier_test=post_std_modifier_test,
                                expert_trajs_dir=EXPERT_TRAJ_LOCATION_DICT[env_option+"."+mode+".noise0.1.small"],
                            )
                            run_experiment_lite(
                                algo.train(),
                                n_parallel=1,
                                snapshot_mode="last",
                                python_command='python3',
                                seed=seed,
                                exp_prefix='CH_IL_E3.3',
                                exp_name='CH_IL_E3.3'
                                         # + str(int(use_maml))
                                         #     +'_fbs'+str(fast_batch_size)
                                         #     +'_mbs'+str(meta_batch_size)
                                         + '_flr_' + str(fast_learning_rate)
                                         #     +'metalr_'+str(meta_step_size)
                                         #     +'_ngrad'+str(num_grad_updates)
                                         + "_bs" + str(beta_steps)
                                         + "_as" + str(adam_steps)
                                         + "_l2m" + str(l2loss_std_mult)
                                         #  + "_prsm" + str(pre_std_modifier)
                                        # + "_pstr" + str(post_std_modifier_train)
                                         #+ "_posm" + str(post_std_modifier_test)
                                       #  + "_l2m" + str(l2loss_std_mult)
                                         + "_" + time.strftime("%D_%H_%M").replace("/", "."),
                                plot=False,
                                sync_s3_pkl=True,
                                mode=mode,
                                terminate_machine=False,
                            )

