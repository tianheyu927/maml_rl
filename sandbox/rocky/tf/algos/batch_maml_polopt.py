import matplotlib
matplotlib.use('Pdf')

import matplotlib.pyplot as plt
import os.path as osp
import rllab.misc.logger as logger
import rllab.plotter as plotter
import tensorflow as tf
import time
import numpy as np

from rllab.algos.base import RLAlgorithm
from sandbox.rocky.tf.policies.base import Policy
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
from rllab.sampler.stateful_pool import singleton_pool

class BatchMAMLPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods, with maml.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            scope=None,
            n_itr=500,
            start_itr=0,
            # Note that the number of trajectories for grad upate = batch_size
            # Defaults are 10 trajectories of length 500 for gradient update
            ## If default is 10 traj-s, why batch_size=100?
            batch_size=100,
            max_path_length=500,
            meta_batch_size=100,
            num_grad_updates=1,
            discount=0.99,
            gae_lambda=1,
            plot=False,
            pause_for_plot=False,
            center_adv=True,
            positive_adv=False,
            store_paths=False,
            whole_paths=True,
            fixed_horizon=False,
            sampler_cls=None,
            sampler_args=None,
            force_batch_sampler=False,
            use_maml=True,
            load_policy=None,
            pre_std_modifier=1.0,
            post_std_modifier_train=1.0,
            post_std_modifier_test=0.001,
            off_policy_meta_learning = False,
            goals_to_use = None,
            expert_trajs = None,
            **kwargs
    ):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.  #
        :param max_path_length: Maximum length of a single rollout.
        :param meta_batch_size: Number of tasks sampled per meta-update
        :param num_grad_updates: Number of fast gradient updates
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :return:
        """
        self.env = env
        self.policy = policy
        self.load_policy=load_policy
        self.baseline = baseline
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr
        # batch_size is the number of trajectories for one fast grad update.
        # self.batch_size is the number of total transitions to collect.
        self.batch_size = batch_size * max_path_length * meta_batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        self.meta_batch_size = meta_batch_size # number of tasks
        self.num_grad_updates = num_grad_updates # number of gradient steps during training
        self.pre_std_modifier = pre_std_modifier
        self.post_std_modifier_train = post_std_modifier_train
        self.post_std_modifier_test = post_std_modifier_test
        self.off_policy_meta_learning = off_policy_meta_learning
        self.expert_trajs = expert_trajs
        if goals_to_use is None and expert_trajs is not None:
            goals_to_use = {}
            for itr in expert_trajs.keys():
                # a bit tricky since expert_traj organizes tasks in dict while goals uses a list
                goals_to_use[itr] = []
                for goalnum in expert_trajs[itr].keys():
                    goals_to_use[itr].append(expert_trajs[itr][goalnum]['goal_or_state'])
        self.goals_to_use = goals_to_use

        if sampler_cls is None:
            if singleton_pool.n_parallel > 1:
                sampler_cls = BatchSampler
                print("Using Batch Sampler")
            else:
                sampler_cls = VectorizedSampler
                print("Using Vectorized Sampler")
        if sampler_args is None:
            sampler_args = dict()
        sampler_args['n_envs'] = self.meta_batch_size
        self.sampler = sampler_cls(self, **sampler_args)

    def start_worker(self):
        self.sampler.start_worker()
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def obtain_samples(self, itr, reset_args=None, log_prefix=''):
        # This obtains samples using self.policy, and calling policy.get_actions(obses)
        # return_dict specifies how the samples should be returned (dict separates samples
        # by task)
        paths = self.sampler.obtain_samples(itr, reset_args, return_dict=True, log_prefix=log_prefix)
        assert type(paths) == dict
        return paths

    def obtain_expert_samples(self, itr, expert_trajs, meta_batch_size, batch_size):
        paths = {}
        for goalnum in expert_trajs[itr].keys():
            temp_traj_list = expert_trajs[itr][goalnum]['expert_traj_list']
            paths[goalnum] = [] # paths[goalnum] will be a list of trajectories
            for
                #test

    def process_samples(self, itr, paths, prefix='', log=True):
        return self.sampler.process_samples(itr, paths, prefix=prefix, log=log)

    def train(self):
        # TODO - make this a util
        flatten_list = lambda l: [item for sublist in l for item in sublist]

        with tf.Session() as sess:
            # Code for loading a previous policy. Somewhat hacky because needs to be in sess.
            if self.load_policy is not None:
                import joblib
                self.policy = joblib.load(self.load_policy)['policy']
            self.init_opt()
            # initialize uninitialized vars  (only initialize vars that were not loaded)
            uninit_vars = []
            for var in tf.global_variables():
                # note - this is hacky, may be better way to do this in newer TF.
                try:
                    sess.run(var)
                except tf.errors.FailedPreconditionError:
                    uninit_vars.append(var)
            sess.run(tf.variables_initializer(uninit_vars))

            self.start_worker()
            start_time = time.time()

            # generating goals ahead of time, the idea being to pickle them eventually
            if self.goals_to_use is not None:
                # goals_to_use is a dictionary where each key (itr) corresponds to a set of goals
                # to be used during a meta-training step
                # see the 'else' statement below for the structure we want to follow
                itr_list = self.goals_to_use.keys()
                assert itr_list == range(self.start_itr, self.n_itr), "algo meta iterations do not start or end at same place as goals_to_use"
                assert len(self.goals_to_use[itr_list[0]]) >= self.meta_batch_size, "number of goals provided per meta iteration not enough"
                learner_env_goals_dict = self.goals_to_use

            else:
                learner_env_goals_dict = {}
                for itr in range(self.start_itr, self.n_itr):
                    with logger.prefix('itr #%d | ' % itr):
                        logger.log("Sampling set of tasks/goals for this meta-batch...")

                        env = self.env
                        while 'sample_goals' not in dir(env):
                            env = env.wrapped_env
                        learner_env_goals = env.sample_goals(self.meta_batch_size)
                        learner_env_goals_dict[itr] = learner_env_goals

            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                with logger.prefix('itr #%d | ' % itr):
                    self.policy.switch_to_init_dist()  # Switch to pre-update policy
                    self.policy.std_modifier = self.pre_std_modifier
                    all_samples_data, all_paths = [], []
                    for step in range(self.num_grad_updates+1):
                        #if step > 0:
                        #    import pdb; pdb.set_trace() # test param_vals functions.
                        logger.log('** Step ' + str(step) + ' **')
                        logger.log("Obtaining samples...")
                        if self.off_policy_meta_learning and step == self.num_grad_updates and itr % 2 == 0: # only train on even itr, sample as regular on odd to see performance
                            # this extracts the paths we want to be working with: observations, rewards, expert actions
                            # and adds
                            paths = self.obtain_expert_samples(itr, self.expert_trajs, self.meta_batch_size, self.batch_size)
                        else:
                            # this obtains a dictionary paths, one dict entry for each env/goal
                            # I believe observations, rewards, actions, env_infos
                            paths = self.obtain_samples(itr, reset_args=learner_env_goals_dict[itr],  log_prefix=str(step))  # there was a reset_args here, TODO revert

                            # for my education:
                            if step == 0 and itr ==0:
                                print("debug8", paths.keys())
                                print("debug8", np.shape(paths[0]))
                                print("debug8", paths[0][0].keys()) #task, trajnum, infotype
                                print("debug8", np.shape(paths[0][0]['observations']))
                                print("debug8", np.shape(paths[0][0]['env_infos']['goal']))
                                print("debug8", np.shape(paths[0][0]['env_infos']['expert_actions']))
                                print("debug8", np.shape(paths[0][0]['agent_infos']['log_std']))
                                print("debug8", np.shape(paths[0][0]['agent_infos']['mean']))
                                print("debug8", np.shape(paths[0][0]['rewards']))
                                print("debug8", np.shape(paths[0][0]['actions']))

                        all_paths.append(paths) # all paths is not used for anything except visualization
                        logger.log("Processing samples...")
                        samples_data = {}
                        for key in paths.keys():  # the keys are the tasks
                            # don't log because this will spam the console with every task.
                            if self.off_policy_meta_learning and step == self.num_grad_updates and itr % 2 == 0:
                                samples_data[key] = self.process_expert_samples(itr, paths[key], log=False)
                            else:
                                # this takes info from the paths and adds returns, advantages, expert_actions,
                                #
                                samples_data[key] = self.process_samples(itr, paths[key], log=False)
                        all_samples_data.append(samples_data)
                        # for logging purposes only
                        self.process_samples(itr, flatten_list(paths.values()), prefix=str(step), log=True)
                        logger.log("Logging diagnostics...")
                        self.log_diagnostics(flatten_list(paths.values()), prefix=str(step))
                        if step < self.num_grad_updates:
                            logger.log("Computing policy updates...")
                            if itr %2 == 0:
                                self.policy.std_modifier = self.post_std_modifier_train
                            else:
                                self.policy.std_modifier = self.post_std_modifier_test
                            self.policy.compute_updated_dists(samples_data)
                        if step == 0 and itr == 0:
                            # this is how we know what a samples_data should look like once it comes out of
                            # process_expert_samples:
                            print("debug7", np.shape(all_samples_data[step][0]['observations'])) # inner grad step, task,
                            print("debug7", np.shape(all_samples_data[step][0]['returns']))
                            print("debug7", np.shape(all_samples_data[step][0]['agent_infos']['log_std']))
                            print("debug7", np.shape(all_samples_data[step][0]['agent_infos']['mean']))
                            print("debug7", np.shape(all_samples_data[step][0]['advantages']))
                            print("debug7", np.shape(all_samples_data[step][0]['rewards']))
                            print("debug7", np.shape(all_samples_data[step][0]['expert_actions']))
                            print("debug7", np.shape(all_samples_data[step][0]['paths']))
                            print("debug7", np.shape(all_samples_data[step][0]['actions']))
                            print("debug7", np.shape(all_samples_data[step][0]['env_infos']['expert_actions']))
                            print("debug7", np.shape(all_samples_data[step][0]['env_infos']['goal']))




                    logger.log("Optimizing policy...")
                    # This needs to take all samples_data so that it can construct graph for meta-optimization.
                    self.optimize_policy(itr, all_samples_data)
                    logger.log("Saving snapshot...")
                    params = self.get_itr_snapshot(itr, all_samples_data[-1])  # , **kwargs)
                    if self.store_paths:
                        params["paths"] = all_samples_data[-1]["paths"]
                    logger.save_itr_params(itr, params)
                    logger.log("Saved")
                    logger.record_tabular('Time', time.time() - start_time)
                    logger.record_tabular('ItrTime', time.time() - itr_start_time)

                    logger.dump_tabular(with_prefix=False)

                    # The rest is some example plotting code.
                    # Plotting code is useful for visualizing trajectories across a few different tasks.
                    if False and (itr-1) % 10 == 0 and self.env.observation_space.shape[0] <= 4: # point-mass
                        logger.log("Saving visualization of paths")
                        for ind in range(min(5, self.meta_batch_size)):
                            plt.clf()
                            plt.plot(learner_env_goals_dict[itr][ind][0], learner_env_goals_dict[itr][ind][1], 'k*', markersize=10)
                            plt.hold(True)

                            preupdate_paths = all_paths[0]
                            postupdate_paths = all_paths[-1]

                            pre_points = preupdate_paths[ind][0]['observations']
                            post_points = postupdate_paths[ind][0]['observations']
                            plt.plot(pre_points[:,0], pre_points[:,1], '-r', linewidth=2)
                            plt.plot(post_points[:,0], post_points[:,1], '-b', linewidth=1)

                            pre_points = preupdate_paths[ind][1]['observations']
                            post_points = postupdate_paths[ind][1]['observations']
                            plt.plot(pre_points[:,0], pre_points[:,1], '--r', linewidth=2)
                            plt.plot(post_points[:,0], post_points[:,1], '--b', linewidth=1)

                            pre_points = preupdate_paths[ind][2]['observations']
                            post_points = postupdate_paths[ind][2]['observations']
                            plt.plot(pre_points[:,0], pre_points[:,1], '-.r', linewidth=2)
                            plt.plot(post_points[:,0], post_points[:,1], '-.b', linewidth=1)

                            plt.plot(0,0, 'k.', markersize=5)
                            plt.xlim([-0.9, 0.9])  # TODO: revert these to -0.8 to 0.8
                            plt.ylim([-0.9, 0.9])
                            plt.legend(['goal', 'preupdate path', 'postupdate path'])
                            plt.savefig(osp.join(logger.get_snapshot_dir(), 'prepost_path' + str(ind) + '_' + str(itr) + '.png'))
                            print(osp.join(logger.get_snapshot_dir(), 'prepost_path' + str(ind) + '_' + str(itr) + '.png'))
                    elif False and itr % 2 == 0:  # swimmer or cheetah
                        logger.log("Saving visualization of paths")
                        for ind in range(min(5, self.meta_batch_size)):
                            plt.clf()
                            goal_vel = learner_env_goals_dict[itr][ind]
                            plt.title('Swimmer paths, goal vel='+str(goal_vel))
                            plt.hold(True)

                            prepathobs = all_paths[0][ind][0]['observations']
                            postpathobs = all_paths[-1][ind][0]['observations']
                            plt.plot(prepathobs[:,0], prepathobs[:,1], '-r', linewidth=2)
                            plt.plot(postpathobs[:,0], postpathobs[:,1], '--b', linewidth=1)
                            plt.plot(prepathobs[-1,0], prepathobs[-1,1], 'r*', markersize=10)
                            plt.plot(postpathobs[-1,0], postpathobs[-1,1], 'b*', markersize=10)
                            plt.xlim([-1.0, 5.0])
                            plt.ylim([-1.0, 1.0])

                            plt.legend(['preupdate path', 'postupdate path'], loc=2)
                            plt.savefig(osp.join(logger.get_snapshot_dir(), 'swim1d_prepost_itr' + str(itr) + '_id' + str(ind) + '.pdf'))
        self.shutdown_worker()

    def log_diagnostics(self, paths, prefix):
        self.env.log_diagnostics(paths, prefix)
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)
