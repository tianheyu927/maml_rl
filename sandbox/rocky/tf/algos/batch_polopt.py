import time

import tensorflow as tf
import numpy as np
import rllab.misc.logger as logger
import rllab.plotter as plotter
from rllab.algos.base import RLAlgorithm
from sandbox.rocky.tf.policies.base import Policy
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler

import joblib
import matplotlib.pyplot as plt
import os.path as osp
from rllab.sampler.utils import rollout, joblib_dump_safe


class BatchPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods.
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
            batch_size=5000,
            batch_size_expert_traj=5000,
            max_path_length=500,
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
            load_policy=None,
            make_video=True,
            action_noise_train=0.0,
            action_noise_test=0.0,
            reset_arg=None,
            save_expert_traj_dir=None,
            expert_traj_itrs_to_pickle=[],
            goals_to_load=None,
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
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
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
        self.load_policy = load_policy
        self.baseline = baseline
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size
        self.batch_size_train = batch_size
        self.batch_size_expert_traj = batch_size_expert_traj
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
        if sampler_cls is None:
            #if self.policy.vectorized and not force_batch_sampler:
            #sampler_cls = VectorizedSampler
            #else:
            sampler_cls = BatchSampler
        if sampler_args is None:
            sampler_args = dict()
        self.sampler = sampler_cls(self, **sampler_args)
        self.reset_arg = reset_arg
        self.action_noise_train = action_noise_train
        self.action_noise_test = action_noise_test
        self.make_video = make_video
        self.save_expert_traj_dir = save_expert_traj_dir
        self.expert_traj_itrs_to_pickle = expert_traj_itrs_to_pickle
        if goals_to_load is not None:
            self.goals_to_use_dict = joblib.load(goals_to_load)
        else:
            self.goals_to_use_dict = {}
        if len(self.expert_traj_itrs_to_pickle) > 0:
            assert save_expert_traj_dir is not None,\
                "please provide a filename to save expert trajectories"
            assert set(self.expert_traj_itrs_to_pickle).issubset(set(range(self.start_itr,self.n_itr))),\
                "Not going to go through all itrs that need to be pickled"
            assert set(self.expert_traj_itrs_to_pickle).issubset(set(self.goals_to_use_dict.keys())),\
                "Haven't loaded goals for all expert trajectories"
            Path(self.save_expert_traj_dir).mkdir(parents=True, exist_ok=True)
            logger.log("Pickling goals...")
            joblib_dump_safe(self.goals_to_use_dict, self.save_expert_traj_dir+"goals.pkl")
            # I know this was redundant, but we are doing it to make sure the goals stay with the expert trajs


        elif save_expert_traj_dir is not None and len(self.expert_traj_itrs_to_pickle) == 0:
            assert False, "please provide expert_traj_itrs_to_pickle"

            # note, in batch_polopt, goals_to use can be a subset of all iterations/goals used,
            # while in batch_maml_polopt, once we use goals_to_load, we enforce that all iterations
            # and meta batch sizes are covered by those goals.
            # The intuition is that for every task used by batch_maml_polopt, we want to have an expert policy
            # that has covered said situation; at the same time, we want an expert policy to pre-train
            # before performing on the set of tasks that we'll use for MAML




    def start_worker(self):
        self.sampler.start_worker()
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def obtain_samples(self, itr, reset_args=None):
        if reset_args is None:
            reset_args = self.reset_arg
        return self.sampler.obtain_samples(itr, reset_args=reset_args)

    def process_samples(self, itr, paths):
        return self.sampler.process_samples(itr, paths)

    def train(self):
        with tf.Session() as sess:
            if self.load_policy is not None:
                self.policy = joblib.load(self.load_policy)['policy']
            self.init_opt()
            # initialize uninitialized vars (I know, it's ugly)
            uninit_vars = []
            for var in tf.all_variables():
                try:
                    sess.run(var)
                except tf.errors.FailedPreconditionError:
                    uninit_vars.append(var)
            sess.run(tf.initialize_variables(uninit_vars))
            #sess.run(tf.initialize_all_variables())
            self.start_worker()
            start_time = time.time()
            for itr in range(self.start_itr, self.n_itr):
                if itr in self.goals_to_use_dict.keys():
                    goals = self.goals_to_use_dict[itr]
                    noise = self.action_noise_test
                    self.batch_size = self.batch_size_expert_traj
                else:
                    goals = [None]
                    noise = self.action_noise_train
                    self.batch_size = self.batch_size_train
                if itr in self.expert_traj_itrs_to_pickle:
                    paths_to_save = {}
                itr_start_time = time.time()
                with logger.prefix('itr #%d | ' % itr):

                    logger.log("Obtaining samples...")
                    paths = []
                    for goalnum, goal in enumerate(goals):
                        paths_for_goal = self.obtain_samples(itr=itr, reset_args=[{'goal': goal, 'noise': noise}])
                        paths.extend(paths_for_goal)  # we need this to be flat because we process all of them together
                        # TODO: there's a bunch of sample processing happening below that we should abstract away
                        if itr in self.expert_traj_itrs_to_pickle:
                            logger.log("Saving trajectories...")
                            paths_no_goalobs = self.clip_goal_from_obs(paths_for_goal)
                            [path.pop('agent_infos') for path in paths_no_goalobs]
                            paths_to_save[goalnum] = paths_no_goalobs
                    if itr in self.expert_traj_itrs_to_pickle:
                        logger.log("Pickling trajectories...")
                        joblib_dump_safe(paths_to_save, self.save_expert_traj_dir+str(itr)+".pkl")
                        logger.log("Fast-processing returns...")
                        undiscounted_returns = [sum(path['rewards']) for path in paths]
                        logger.record_tabular('AverageReturn', np.mean(undiscounted_returns))

                    else:
                        logger.log("Processing samples...")
                        samples_data = self.process_samples(itr, paths)
                        logger.log("Logging diagnostics...")
                        self.log_diagnostics(paths)
                        logger.log("Optimizing policy...")
                        self.optimize_policy(itr, samples_data)
                        #new_param_values = self.policy.get_variable_values(self.policy.all_params)
                        logger.log("Saving snapshot...")
                        params = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
                        if self.store_paths:
                            params["paths"] = samples_data["paths"]

                    logger.save_itr_params(itr, params)
                    logger.log("Saved")
                    logger.record_tabular('Time', time.time() - start_time)
                    logger.record_tabular('ItrTime', time.time() - itr_start_time)

                    if True and (itr % 16 == 0) and self.env.observation_space.shape[0] < 12:  # ReacherEnvOracleNoise
                        logger.log("Saving visualization of paths")
                        plt.clf()
                        plt.hold(True)

                        goal = paths[0]['observations'][0][-2:]
                        plt.plot(goal[0], goal[1], 'k*', markersize=10)

                        goal = paths[1]['observations'][0][-2:]
                        plt.plot(goal[0], goal[1], 'k*', markersize=10)

                        goal = paths[2]['observations'][0][-2:]
                        plt.plot(goal[0], goal[1], 'k*', markersize=10)

                        points = np.array([obs[6:8] for obs in paths[0]['observations']])
                        plt.plot(points[:, 0], points[:, 1], '-r', linewidth=2)

                        points = np.array([obs[6:8] for obs in paths[1]['observations']])
                        plt.plot(points[:, 0], points[:, 1], '--r', linewidth=2)

                        points = np.array([obs[6:8] for obs in paths[2]['observations']])
                        plt.plot(points[:, 0], points[:, 1], '-.r', linewidth=2)

                        plt.plot(0, 0, 'k.', markersize=5)
                        plt.xlim([-0.25, 0.25])
                        plt.ylim([-0.25, 0.25])
                        plt.legend(['path'])
                        plt.savefig(osp.join(logger.get_snapshot_dir(),
                                             'path' + str(0) + '_' + str(itr) + '.png'))
                        print(osp.join(logger.get_snapshot_dir(),
                                       'path' + str(0) + '_' + str(itr) + '.png'))

                        if self.make_video and itr % 80 == 0 and itr in self.goals_to_use_dict.keys() == 0:
                            logger.log("Saving videos...")
                            self.env.reset(reset_args=self.goals_to_use_dict[itr][0])
                            video_filename = osp.join(logger.get_snapshot_dir(), 'post_path_%s.mp4' % itr)
                            rollout(env=self.env, agent=self.policy, max_path_length=self.max_path_length,
                                    animated=True, speedup=2, save_video=True, video_filename=video_filename,
                                    reset_arg=self.goals_to_use_dict[itr][0],
                                    use_maml=False,)

                    # debugging
                    """
                    if itr % 1 == 0:
                        logger.log("Saving visualization of paths")
                        import matplotlib.pyplot as plt;
                        for ind in range(5):
                            plt.clf(); plt.hold(True)
                            points = paths[ind]['observations']
                            plt.plot(points[:,0], points[:,1], '-r', linewidth=2)
                            plt.xlim([-1.0, 1.0])
                            plt.ylim([-1.0, 1.0])
                            plt.legend(['path'])
                            plt.savefig('/home/cfinn/path'+str(ind)+'.png')
                    """
                    # end debugging

                    logger.dump_tabular(with_prefix=False)
                    if self.plot:
                        self.update_plot()
                        if self.pause_for_plot:
                            input("Plotting evaluation run: Press Enter to "
                                  "continue...")









        self.shutdown_worker()

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths)
        self.policy.log_diagnostics(paths)
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

    def clip_goal_from_obs(self, paths):
        env = self.env
        while 'clip_goal_from_obs' not in dir(env):
            env = env.wrapped_env
        return env.clip_goal_from_obs(paths)