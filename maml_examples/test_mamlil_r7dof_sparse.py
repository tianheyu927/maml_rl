from sandbox.rocky.tf.algos.vpg import VPG
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.policies.minimal_gauss_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from maml_examples.r7dof_sparse_env import Reacher7DofMultitaskEnvSparse
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler




import joblib
import os
import csv
import numpy as np
import random as rd
import pickle
import tensorflow as tf

# files, descriptions, step_sizes = [[]]*4, [[]]*4, [[]]*4
files, descriptions, step_sizes = [[]], [[]], [[]]

#
# files[0], descriptions[0], step_sizes[0] = 'data/local/R7-IL-SPARSE-0619/R7_IL_SPARSE_200_40_1_1_dem40_ei5_as10_basl_1906_11_55/itr_88.pkl' ,"our_algo",1.0
# files[1], descriptions[1], step_sizes[1] ='data/local/R7-IL-SPARSE-0619/R7_IL_SPARSE_200_40_1_1_dem40_ei5_as10_basl_1906_11_55/itr_0.pkl'  ,"0_metaitr", 1.0
#
# files[2], descriptions[2], step_sizes[2] = 'data/local/R7-IL-SPARSE-0622/R7_IL_SPARSE_200_40_1_1_flr0.0_dem40_ei5_as10_basl_2206_03_30/itr_16.pkl' , "0_flr", 1.0
files[0], descriptions[0], step_sizes[0] ='data/local/R7-TR-SPARSE/R7_TR_SPARSErelu2x100.f1.0_062318_15_00/itr_42.pkl' , "trpo", 1.0

make_video = False  # generate results if False, run code to make video if True
run_id = 1  # for if you want to run this script in multiple terminals (need to have different ids)

temp_env = TfEnv(normalize(Reacher7DofMultitaskEnvSparse(envseed=0)))

if not make_video:
    np.random.seed(1)
    rd.seed(1)
    tf.set_random_seed(1)
    test_num_goals = 40
    goals = temp_env.wrapped_env.wrapped_env.sample_goals(test_num_goals)
else:
    np.random.seed(1)
    rd.seed(1)
    tf.set_random_seed(1)
    test_num_goals = 1
    goals = temp_env.wrapped_env.wrapped_env.sample_goals(test_num_goals)
    file_ext = 'gif'  # can be mp4 or gif

stub(globals())
env = TfEnv(normalize(Reacher7DofMultitaskEnvSparse(envseed=0)))


gen_name = 'r7dof_sparse_eval_'
names = ['mamlil'] #,'pretrain','random', 'oracle']
exp_names = [gen_name + name for name in names]

initial_params_files = files
n_itrs_list = [2,3,4,5,6]  # 1 through 5 grad steps
# all_avg_returns = []
for step_size, initial_params_file, desc in zip(step_sizes, initial_params_files, descriptions):
    ret_means_for_algo = []
    ret_stds_for_algo = []
    for n_itr in n_itrs_list:
        # avg_returns = []
        final_returns = []  # final returns for algo, n_itr
        for goalnum, goal in enumerate(goals):
            final_return = None  # final return for algo, n_itr, goal
            goal = goal.tolist()
            policy = GaussianMLPPolicy(  # random policy
                name='policy',
                env_spec=env.spec,
                hidden_nonlinearity=tf.nn.relu,
                hidden_sizes=(100, 100),
                extra_input_dim=5,
            )

            if initial_params_file is not None:
                policy = None
            make_video1 = True if goalnum in [0,1,2] else False
            baseline = LinearFeatureBaseline(env_spec=env.spec)
            algo = VPG(
                env=env,
                policy=policy,
                load_policy=initial_params_file,
                baseline=baseline,
                batch_size=2000,
                max_path_length=100,
                n_itr=n_itr,
                #step_size=10.0,
                sampler_cls=VectorizedSampler, # added by RK 6/19
                sampler_args = dict(n_envs=1),
                reset_arg=goal,
                optimizer=None,
                optimizer_args={'init_learning_rate': step_size, 'tf_optimizer_args': {'learning_rate': 0.5*step_size}, 'tf_optimizer_cls': tf.train.GradientDescentOptimizer},
                # extra_input="onehot_exploration", # added by RK 6/19
                # extra_input_dim=5, # added by RK 6/19
                make_video=make_video1
            )
            exp_name='mamlil'+desc+str(run_id)+"_n_itr"+str(n_itr)+"_goal"+str(goalnum)

            run_experiment_lite(
                algo.train(),
                # Number of parallel workers for sampling
                n_parallel=1,
                # Only keep the snapshot parameters for the last iteration
                snapshot_mode="all",
                # Specifies the seed for the experiment. If this is not provided, a random seed
                # will be used
                seed=1, # don't set the seed for oracle, since it's already deterministic.
                exp_prefix='R7DOF_SPARSE_EVAL',
                exp_name=exp_name,
                plot=True,
            )
            # get return from the experiment
            with open('data/local/R7DOF-SPARSE-EVAL/'+exp_name+'/progress.csv', 'r') as f:
                reader = csv.reader(f, delimiter=',')
                i = 0
                row = None
                returns = []
                for row in reader:
                    i+=1
                    if i ==1:
                        ret_idx = row.index('AverageReturn')
                    else:
                        returns.append(float(row[ret_idx]))
                        final_return = float(row[ret_idx])
                # avg_returns.append(returns)
                if final_return is not None:
                    final_returns.append(final_return)
            if make_video:
                data_loc = 'data/local/R7DOF-SPARSE-EVAL/mamlil'+str(run_id)+'/'
                save_loc = 'data/local/R7DOF-SPARSE-EVAL//videos/'
                param_file = initial_params_file
                save_prefix = save_loc + names[step_i] + '_goal_' + str(goalnum)
                video_filename = save_prefix + 'prestep.' + file_ext
                import os
                os.system('python scripts/sim_policy.py ' + param_file + ' --speedup=4 --max_path_length=100 --video_filename='+video_filename)
                for itr_i in range(3):
                    param_file = data_loc + 'itr_' + str(itr_i)  + '.pkl'
                    video_filename = save_prefix + 'step_'+str(itr_i)+'.'+file_ext
                    os.system('python scripts/sim_policy.py ' + param_file + ' --speedup=4 --max_path_length=100 --video_filename='+video_filename)
        ret_mean_for_algo_n_itr = np.mean(final_returns)
        ret_std_for_algo_n_itr = np.std(final_returns)
        ret_means_for_algo.append(ret_mean_for_algo_n_itr)
        ret_stds_for_algo.append(ret_std_for_algo_n_itr)
    print("means", ret_means_for_algo)
    print("stds", ret_stds_for_algo)
    if not os.path.exists('data/local/R7DOF-SPARSE-EVAL/mamlil'+desc):
        os.makedirs('data/local/R7DOF-SPARSE-EVAL/mamlil'+desc)
    with open('data/local/R7DOF-SPARSE-EVAL/summary_'+desc+'.csv', "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["n_itr","AverageReturn","ReturnStd"])
        print("debug", ret_means_for_algo)
        print("debug", ret_stds_for_algo)
        for i, (mean, std) in enumerate(zip(ret_means_for_algo,ret_stds_for_algo)):
            print("debug", [str(i),str(mean),str(std)])
            writer.writerow([str(i),str(mean),str(std)])
    # all_avg_returns.append(avg_returns)
    #
    #
    # task_avg_returns = []
    # for itr in range(len(all_avg_returns[step_i][0])):
    #     task_avg_returns.append([ret[itr] for ret in all_avg_returns[step_i]])
    #
    # if not make_video:
    #     results = {'task_avg_returns': task_avg_returns}
    #     with open(exp_names[step_i] + '.pkl', 'wb') as f:
    #         pickle.dump(results, f)


for i in range(len(initial_params_files)):
    returns = []
    std_returns = []
    # returns.append(np.mean([ret[itr] for ret in all_avg_returns[i]]))
    # std_returns.append(np.std([ret[itr] for ret in all_avg_returns[i]]))
    print(initial_params_files[i])
    print(returns) #np.mean(all_avg_returns[i]), np.std(all_avg_returns[i])
    print(std_returns) #np.mean(all_avg_returns[i]), np.std(all_avg_returns[i])


