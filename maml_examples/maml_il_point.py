from sandbox.rocky.tf.algos.maml_il import MAMLIL
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from maml_examples.point_env_randgoal_expert import PointEnvRandGoalExpert
from maml_examples.point_env_randgoal import PointEnvRandGoal
from maml_examples.point_env_randgoal_oracle import PointEnvRandGoalOracle
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

import tensorflow as tf
import time
from maml_examples.point_vars import POINT_GOALS_LOCATION, EXPERT_TRAJ_LOCATION_DICT
from maml_examples.maml_experiment_vars import MOD_FUNC
import numpy as np
import random as rd
beta_adam_steps_list = [(1,3)] #,(1,100)]  # , ## maybe try 1 and 10 to compare, we know that 1 is only slightly worse than 5

fast_learning_rates = [1.0]  #1.0 seems to work best
baselines = ['linear']

fast_batch_size = 20  # 20 # 10 works for [0.1, 0.2], 20 doesn't improve much for [0,0.2]  #inner grad update size
meta_batch_size = 40  # 40 @ 10 also works, but much less stable, 20 is fairly stable, 40 is more stable
max_path_length = 100  # 100
num_grad_updates = 1
meta_step_size = 0.01 # 0.01
pre_std_modifier_list = [1.0]
post_std_modifier_train_list = [0.00001]
post_std_modifier_test_list = [0.00001]
l2loss_std_mult_list = [1.0]
env_options = ["box"]

use_maml = True
seeds=[1]  #,2,3,4,5,6,7,8]
for seed in seeds:
    for env_option in env_options:
        for l2loss_std_mult in l2loss_std_mult_list:
            for post_std_modifier_train in post_std_modifier_train_list:
                for post_std_modifier_test in post_std_modifier_test_list:
                    for pre_std_modifier in pre_std_modifier_list:
                        for fast_learning_rate in fast_learning_rates:
                            for beta_steps, adam_steps in beta_adam_steps_list:
                                for bas in baselines:
                                    stub(globals())
                                    tf.set_random_seed(seed)
                                    np.random.seed(seed)
                                    rd.seed(seed)

                                    ###
                                    seed %= 4294967294
                                    global seed_
                                    seed_ = seed
                                    rd.seed(seed)
                                    np.random.seed(seed)
                                    try:
                                        import tensorflow as tf

                                        tf.set_random_seed(seed)
                                    except Exception as e:
                                        print(e)
                                    print('using seed %s' % (str(seed)))
                                    env = TfEnv(normalize(PointEnvRandGoal()))
                                    policy = MAMLGaussianMLPPolicy(
                                        name="policy",
                                        env_spec=env.spec,
                                        grad_step_size=fast_learning_rate,
                                        hidden_nonlinearity=tf.nn.relu,
                                        hidden_sizes=(100, 100),
                                        std_modifier=pre_std_modifier,
                                    )
                                    if bas == 'zero':
                                        baseline = ZeroBaseline(env_spec=env.spec)
                                    elif 'linear' in bas:
                                        baseline = LinearFeatureBaseline(env_spec=env.spec)
                                    else:
                                        baseline = GaussianMLPBaseline(env_spec=env.spec)
                                    #expert_policy = PointEnvExpertPolicy(env_spec=env.spec)
                                    algo = MAMLIL(
                                        env=env,
                                        policy=policy,
                                        baseline=baseline,
                                        #expert_policy=expert_policy,  TODO: we will want to define the expert policy here
                                        batch_size=fast_batch_size, ## number of trajs for alpha grad update
                                        max_path_length=max_path_length,
                                        meta_batch_size=meta_batch_size, ## number of tasks sampled for beta grad update
                                        num_grad_updates=num_grad_updates, ## number of alpha grad updates per beta update
                                        n_itr=100, #100
                                        use_maml=use_maml,
                                        use_pooled_goals=True,
                                        step_size=meta_step_size,
                                        plot=False,
                                        beta_steps=beta_steps,
                                        adam_steps=adam_steps,
                                        pre_std_modifier=pre_std_modifier,
                                        l2loss_std_mult=l2loss_std_mult,
                                        importance_sampling_modifier=MOD_FUNC[""],
                                        post_std_modifier_train=post_std_modifier_train,
                                        post_std_modifier_test=post_std_modifier_test,
                                        expert_trajs_dir=EXPERT_TRAJ_LOCATION_DICT[".ec2"],
                                    )

                                    run_experiment_lite(
                                        algo.train(),
                                        n_parallel=1,
                                        snapshot_mode="all",
                                        python_command='python3',
                                        seed=1,
                                        exp_prefix='PR_IL_',
                                        exp_name='PR_IL_'
                                        +str(seed)
                                                 # +str(int(use_maml))
                                                # +'_fbs'+str(fast_batch_size)
                                                # +'_mbs'+str(meta_batch_size)
                                                 +'_flr'+str(fast_learning_rate)
                                                # +'_mlr'+str(meta_step_size)
                                                # +'_ngrad'+str(num_grad_updates)
                                                #  +"_bs" + str(beta_steps)
                                                 + "_as" + str(adam_steps)
                                                 # +"_prsm"+ str(pre_std_modifier)
                                                 # +"_pstr"+ str(post_std_modifier_train)
                                                 # +"_posm" + str(post_std_modifier_test)
                                                 # +"_l2m" + str(l2loss_std_mult)
                                                 #+"_env" + str(env_option)
                                                 +"_"+time.strftime("%D_%H_%M").replace("/","."),
                                        plot=False,mode='ec2',
                                    )



