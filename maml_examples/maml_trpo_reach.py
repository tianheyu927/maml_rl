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
from gym.wrappers.monitoring import Monitor
from maml_examples.maml_experiment_vars import HIDDEN_NONLINEARITY, OUTPUT_NONLINEARITY
from maml_examples.reacher_vars import GOALS_LOCATION
import tensorflow as tf
import time

baselines = ['linear']
env_option = 'g100l0.25nfj'
nonlinearity_option = 'reluh'  #A1=relu, A2=reluh
net_size = 200
fast_learning_rates = [0.15]  # A1=0.3, A2=0.15
fast_batch_size = 40  # A1=40
meta_batch_size = 40  # A1=40
num_grad_updates = 1  #1
n_itr = 200  #100
max_path_length = 100
meta_step_size = 0.01  ## it was 0.01
pre_std_modifier_list = [1.0]
post_std_modifier_train_list = [1.0]
post_std_modifier_test_list = [0.0001]


l2loss_std_mult_list = [1.0]  # not needed here


use_maml = True


for l2loss_std_mult in l2loss_std_mult_list:
    for post_std_modifier_train in post_std_modifier_train_list:
        for post_std_modifier_test in post_std_modifier_test_list:
            for pre_std_modifier in pre_std_modifier_list:
                for fast_learning_rate in fast_learning_rates:
                    for bas in baselines:
                        stub(globals())

                        seed = 1
                        #env = TfEnv(normalize(GymEnv("Pusher-v0", force_reset=True, record_video=False)))  #TODO: force_reset was True
                        #xml_filepath ='home/rosen/rllab_copy/vendor/local_mujoco_models/ensure_woodtable_distractor_pusher%s.xml' % seed
                      #  env = Monitor(TfEnv(normalize(ReacherEnv(option=env_option))), '/home/rosen/temp/')
                        env = TfEnv(normalize(ReacherEnv(option=env_option)))
                      #  env = TfEnv(normalize(GymEnv("Reacher-v1", force_reset=True, record_video=True)))

                        policy = MAMLGaussianMLPPolicy(
                            name="policy",
                            env_spec=env.spec,
                            grad_step_size=fast_learning_rate,
                            hidden_nonlinearity=HIDDEN_NONLINEARITY[nonlinearity_option],
                            hidden_sizes=(net_size, net_size),
                            output_nonlinearity=OUTPUT_NONLINEARITY[nonlinearity_option],
                            std_modifier=pre_std_modifier,
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
                            batch_size=fast_batch_size,
                            max_path_length=max_path_length,
                            meta_batch_size=meta_batch_size,
                            num_grad_updates=num_grad_updates,
                            n_itr=n_itr,
                            use_maml=use_maml,
                            step_size=meta_step_size,
                            plot=False,
                            pre_std_modifier=pre_std_modifier,
                            post_std_modifier_train=post_std_modifier_train,
                            post_std_modifier_test=post_std_modifier_test,
                           # goals_to_load=GOALS_LOCATION,
                            # goals_pickle_to=GOALS_LOCATION,

                        )
                        run_experiment_lite(
                            algo.train(),
                            n_parallel=1, #10, If you use more than 1, your std modifiers may not work
                            snapshot_mode="last",
                            python_command='python3',
                            seed=seed,
                            exp_prefix='RE_TR_1',
                            exp_name='RE_TR_1'
                                     # + ".itr." + str(n_itr)
                                     # + str(int(use_maml))
                                     #     +'_fbs'+str(fast_batch_size)
                                     #     +'_mbs'+str(meta_batch_size)
                                     + env_option
                                     + nonlinearity_option
                                     + "2x" + str(net_size)
                                     + '.f' + str(fast_learning_rate)
                                     # +'metalr_'+str(meta_step_size)
                                     # +'_ngrad'+str(num_grad_updates)
                                     #  + "_prsm" + str(pre_std_modifier)
                                     #  + "_pstr" + str(post_std_modifier_train)
                                     # + '_posm' + str(post_std_modifier_test)
                                     + "_" + time.strftime("%D_%H:%M").replace("/", ""),
                            plot=False,
                        )
