from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
from sandbox.rocky.tf.algos.maml_il import MAMLIL
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.maml_gaussian_mlp_baseline import MAMLGaussianMLPBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy
from sandbox.rocky.tf.optimizers.quad_dist_expert_optimizer import QuadDistExpertOptimizer
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer

from sandbox.rocky.tf.envs.base import TfEnv
# import lasagne.nonlinearities as NL
import sandbox.rocky.tf.core.layers as L

from rllab.envs.gym_env import GymEnv
from maml_examples.reacher_env import ReacherEnv
from rllab.envs.mujoco.pusher_env import PusherEnv
from maml_examples.r7dof_env import Reacher7DofMultitaskEnv
from maml_examples.r7dof_vars import EXPERT_TRAJ_LOCATION_DICT, ENV_OPTIONS, default_r7dof_env_option
from maml_examples.maml_experiment_vars import MOD_FUNC


#from examples.trpo_push_obj import


import tensorflow as tf
import time

beta_adam_steps_list = [(1,125),]

fast_learning_rates = [1.0]
baselines = ['MAMLGaussianMLP']  #['linear'] GaussianMLP MAMLGaussianMLP zero
env_option = ''
# mode = "ec2"
mode = "local"

fast_batch_size_list = [20]  # 20 # 10 works for [0.1, 0.2], 20 doesn't improve much for [0,0.2]  #inner grad update size
meta_batch_size = 40  # 40 @ 10 also works, but much less stable, 20 is fairly stable, 40 is more stable
max_path_length = 100  # 100
num_grad_updates = 1
meta_step_size = 0.01
pre_std_modifier_list = [1.0]
post_std_modifier_train_list = [0.00001]
post_std_modifier_test_list = [0.00001]
l2loss_std_mult_list = [1.0]
importance_sampling_modifier_list = ['clip0.5_']
limit_expert_traj_num_list = [40]  # 40
test_goals_mult = 1
bas_lr = 0.00  # baseline learning rate
bas_hnl = tf.identity
# bas_onl = lambda x: x*0.0 + tf.constant(-5.0)
baslayers_list = [(1,), ]
basas = 200 # baseline adam steps




use_maml = True
for baslayers in baslayers_list:
    for fast_batch_size in fast_batch_size_list:
        for ism in importance_sampling_modifier_list:
            for limit_expert_traj_num in limit_expert_traj_num_list:
                for l2loss_std_mult in l2loss_std_mult_list:
                    for post_std_modifier_train in post_std_modifier_train_list:
                        for post_std_modifier_test in post_std_modifier_test_list:
                            for pre_std_modifier in pre_std_modifier_list:
                                for fast_learning_rate in fast_learning_rates:
                                    for beta_steps, adam_steps in beta_adam_steps_list:
                                        for bas in baselines:
                                            stub(globals())

                                            seed = 1
                                            env = TfEnv(normalize(Reacher7DofMultitaskEnv()))

                                            policy = MAMLGaussianMLPPolicy(
                                                name="policy",
                                                env_spec=env.spec,
                                                grad_step_size=fast_learning_rate,
                                                hidden_nonlinearity=tf.nn.relu,
                                                hidden_sizes=(100, 100),
                                                std_modifier=pre_std_modifier,
                                                # metalearn_baseline=(bas == "MAMLGaussianMLP"),
                                            )
                                            if bas == 'zero':
                                                baseline = ZeroBaseline(env_spec=env.spec)
                                            elif bas == 'MAMLGaussianMLP':
                                                baseline = MAMLGaussianMLPBaseline(env_spec=env.spec,
                                                                                   learning_rate=bas_lr,
                                                                                   hidden_sizes=baslayers,
                                                                                   hidden_nonlinearity=bas_hnl,
                                                                                   # learn_std=False,
                                                                                   # use_trust_region=False,
                                                                                   # optimizer=QuadDistExpertOptimizer(
                                                                                   #      name="bas_optimizer",
                                                                                   #     #  tf_optimizer_cls=tf.train.GradientDescentOptimizer,
                                                                                   #     #  tf_optimizer_args=dict(
                                                                                   #     #      learning_rate=bas_lr,
                                                                                   #     #  ),
                                                                                   #     # # tf_optimizer_cls=tf.train.AdamOptimizer,
                                                                                   #     # max_epochs=200,
                                                                                   #     # batch_size=None,
                                                                                   #      adam_steps=basas
                                                                                   #     )
                                                                                   )

                                            elif bas == 'linear':
                                                baseline = LinearFeatureBaseline(env_spec=env.spec)
                                            elif "GaussianMLP" in bas:
                                                baseline = GaussianMLPBaseline(env_spec=env.spec,
                                                                                   regressor_args=dict(
                                                                                   hidden_sizes=baslayers,
                                                                                   hidden_nonlinearity=bas_hnl,
                                                                                   learn_std=False,
                                                                                   # use_trust_region=False,
                                                                                   # normalize_inputs=False,
                                                                                   # normalize_outputs=False,
                                                                                   optimizer=QuadDistExpertOptimizer(
                                                                                        name="bas_optimizer",
                                                                                       #  tf_optimizer_cls=tf.train.GradientDescentOptimizer,
                                                                                       #  tf_optimizer_args=dict(
                                                                                       #      learning_rate=bas_lr,
                                                                                       #  ),
                                                                                       # # tf_optimizer_cls=tf.train.AdamOptimizer,
                                                                                       # max_epochs=200,
                                                                                       # batch_size=None,
                                                                                        adam_steps=basas
                                                                                       ))
                                                                                   )
                                            algo = MAMLIL(
                                                env=env,
                                                policy=policy,
                                                baseline=baseline,
                                                batch_size=fast_batch_size,  # number of trajs for alpha grad update
                                                max_path_length=max_path_length,
                                                meta_batch_size=meta_batch_size,  # number of tasks sampled for beta grad update
                                                num_grad_updates=num_grad_updates,  # number of alpha grad updates
                                                n_itr=20, #100
                                                make_video=True,
                                                use_maml=use_maml,
                                                use_pooled_goals=True,
                                                metalearn_baseline=(bas=="MAMLGaussianMLP"),
                                                limit_expert_traj_num=limit_expert_traj_num,
                                                test_goals_mult=test_goals_mult,
                                                step_size=meta_step_size,
                                                plot=False,
                                                beta_steps=beta_steps,
                                                adam_steps=adam_steps,
                                                pre_std_modifier=pre_std_modifier,
                                                l2loss_std_mult=l2loss_std_mult,
                                                importance_sampling_modifier=MOD_FUNC[ism],
                                                post_std_modifier_train=post_std_modifier_train,
                                                post_std_modifier_test=post_std_modifier_test,
                                                expert_trajs_dir=EXPERT_TRAJ_LOCATION_DICT[env_option+"."+mode],
                                            )
                                            run_experiment_lite(
                                                algo.train(),
                                                n_parallel=1,
                                                snapshot_mode="last",
                                                python_command='python3',
                                                seed=seed,
                                                exp_prefix='R7_IL_D0.2',
                                                exp_name='R7_IL_D0.2'
                                                         # + str(int(use_maml))
                                                             +'_fbs'+str(fast_batch_size)
                                                             +'_mbs'+str(meta_batch_size)
                                                         + '_flr_' + str(fast_learning_rate)
                                                         + '_demo' + str(limit_expert_traj_num)
                                                         #+ '_tgm' + str(test_goals_mult)
                                                         #     +'metalr_'+str(meta_step_size)
                                                         #     +'_ngrad'+str(num_grad_updates)
                                                         + "_bs" + str(beta_steps)
                                                         + "_as" + str(adam_steps)
                                                         #+"_net" + str(net_size[0])
                                                         # +"_L2m" + str(l2loss_std_mult)
                                                         + "_prsm" + str(pre_std_modifier)
                                                         # + "_pstr" + str(post_std_modifier_train)
                                                         # + "_posm" + str(post_std_modifier_test)
                                                         #  + "_l2m" + str(l2loss_std_mult)
                                                         + "_ism" + ism
                                                         + "_bas" + bas[0]
                                                        +"_tfbe" # TF backend for baseline
                                                        +"_qdo" # quad dist optimizer
                                                        +("_bi" if bas_hnl == tf.identity else ("_brel" if bas_hnl == tf.nn.relu else "_bth"))  # identity or relu or tanh for baseline
                                                         +"_" + str(baslayers) #size
                                                        +"_basas" + str(basas) #baseline adam steps
                                                         + "_" + time.strftime("%D_%H_%M").replace("/", "."),
                                                plot=False,
                                                sync_s3_pkl=True,
                                                mode=mode,
                                                terminate_machine=False,
                                            )

