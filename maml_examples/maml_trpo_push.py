from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from maml_examples.point_env_randgoal_expert import PointEnvRandGoalExpert
from maml_examples.point_env_randgoal_oracle import PointEnvRandGoalOracle
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

from rllab.envs.gym_env import GymEnv
from maml_examples.reacher_env import ReacherEnv
from rllab.envs.mujoco.pusher_env import PusherEnv

import tensorflow as tf
import time

beta_steps_list = [1]  # Not implemented for TRPO

fast_learning_rates = [0.5]  # RK this seems to be
baselines = ['linear']
fast_batch_size = 20  # 10 works for [0.1, 0.2], 20 doesn't improve much for [0,0.2]
meta_batch_size = 40  # 10 also works, but much less stable, 20 is fairly stable, 40 is more stable
max_path_length = 100
num_grad_updates = 1
meta_step_size = 0.01  ## is this beta?
pre_std_modifier_list = [1.0]
post_std_modifier_train_list = [1.0]
post_std_modifier_test_list = [0.001]
l2loss_std_mult_list = [1.0]  # not needed here


use_maml = True


for l2loss_std_mult in l2loss_std_mult_list:
    for post_std_modifier_train in post_std_modifier_train_list:
        for post_std_modifier_test in post_std_modifier_test_list:
            for pre_std_modifier in pre_std_modifier_list:
                for fast_learning_rate in fast_learning_rates:
                    for beta_steps in beta_steps_list:
                        for bas in baselines:
                            stub(globals())

                            seed = 1
                            env = TfEnv(normalize(GymEnv("Pusher-v0", force_reset=True, record_video=False)))
                            #xml_filepath ='home/rosen/rllab_copy/vendor/local_mujoco_models/ensure_woodtable_distractor_pusher%s.xml' % seed
                            #env = TfEnv(normalize(PusherEnv(xml_file=xml_filepath)))



                            policy = MAMLGaussianMLPPolicy(
                                name="policy",
                                env_spec=env.spec,
                                grad_step_size=fast_learning_rate,
                                hidden_nonlinearity=tf.nn.relu,
                                hidden_sizes=(100, 100),
                            )
                            if bas == 'zero':
                                baseline = ZeroBaseline(env_spec=env.spec)
                            elif 'linear' in bas:
                                baseline = LinearFeatureBaseline(env_spec=env.spec)
                            else:
                                baseline = GaussianMLPBaseline(env_spec=env.spec)
                            algo = MAMLTRPO(
                                env=env,
                                policy=policy,
                                baseline=baseline,
                                batch_size=fast_batch_size,  # number of trajs for alpha grad update
                                max_path_length=max_path_length,
                                meta_batch_size=meta_batch_size,  # number of tasks sampled for beta grad update
                                num_grad_updates=num_grad_updates,  # number of alpha grad updates
                                n_itr=100,
                                use_maml=use_maml,
                                step_size=meta_step_size,
                                plot=False,
                                pre_std_modifier=pre_std_modifier,
                                post_std_modifier_train=post_std_modifier_train,
                                post_std_modifier_test=post_std_modifier_test
                            )
                            run_experiment_lite(
                                algo.train(),
                                n_parallel=1,
                                snapshot_mode="last",
                                python_command='python3',
                                seed=seed,
                                exp_prefix='maml_trpo_push100',
                                exp_name='trpomaml'
                                         + str(int(use_maml))
                                         #     +'_fbs'+str(fast_batch_size)
                                         #     +'_mbs'+str(meta_batch_size)
                                         + '_flr_' + str(fast_learning_rate)
                                         #     +'metalr_'+str(meta_step_size)
                                         #     +'_ngrad'+str(num_grad_updates)
                                         + "_prsm" + str(pre_std_modifier)
                                         + "_pstr" + str(post_std_modifier_train)
                                         + '_posm' + str(post_std_modifier_test)
                                         + "_" + time.strftime("%D.%H:%M").replace("/", "."),
                                plot=False,
                            )
