import tensorflow as tf
import numpy as np
import rllab.misc.logger as logger
from rllab.misc import ext
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.algos.batch_maml_polopt import BatchMAMLPolopt
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.optimizers.quad_dist_expert_optimizer import QuadDistExpertOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from maml_examples.maml_experiment_vars import TESTING_ITRS

class MAMLIL(BatchMAMLPolopt):

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            step_size=0.01,
            use_maml=True,
            beta_steps=1,
            adam_steps=1,
            l2loss_std_mult=1.0,
            importance_sampling_modifier=tf.identity,
            metalearn_baseline=False,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict(min_penalty=1e-8)
            optimizer = QuadDistExpertOptimizer("name1", adam_steps=adam_steps)  #  **optimizer_args)
        self.optimizer = optimizer
        self.step_size = step_size
        self.use_maml = use_maml
        self.kl_constrain_step = -1
        self.l2loss_std_multiplier = l2loss_std_mult
        self.ism = importance_sampling_modifier
        self.metalearn_baseline = metalearn_baseline
        super(MAMLIL, self).__init__(optimizer=optimizer, beta_steps=beta_steps, use_maml_il=True, metalearn_baseline=metalearn_baseline, **kwargs)


    def make_vars(self, stepnum='0'):
        # lists over the meta_batch_size
        # We should only need the last stepnum for meta-optimization.
        obs_vars, action_vars, adv_vars, rewards_vars, returns_vars, path_lengths_vars, expert_action_vars = [], [], [], [], [], [], []
        for i in range(self.meta_batch_size):
            obs_vars.append(self.env.observation_space.new_tensor_variable(
                'obs' + stepnum + '_' + str(i),
                extra_dims=1,
            ))
            action_vars.append(self.env.action_space.new_tensor_variable(
                'action' + stepnum + '_' + str(i),
                extra_dims=1,
            ))
            adv_vars.append(tensor_utils.new_tensor(
                    'advantage' + stepnum + '_' + str(i),
                    ndim=1, dtype=tf.float32,
                ))
            if self.metalearn_baseline:
                rewards_vars.append(tensor_utils.new_tensor(
                    'rewards' + stepnum + '_' + str(i),
                    ndim=1, dtype=tf.float32,
                ))
                returns_vars.append(tensor_utils.new_tensor(
                    'returns' + stepnum + '_' + str(i),
                    ndim=1, dtype=tf.float32,
                ))
                # path_lengths_vars.append(tensor_utils.new_tensor(
                #     'path_lengths' + stepnum + '_' + str(i),
                #     ndim=1, dtype=tf.float32,
                # ))
            expert_action_vars.append(self.env.action_space.new_tensor_variable(
                name='expert_actions' + stepnum + '_' + str(i),
                extra_dims=1,
            ))
        if not self.metalearn_baseline:
            return obs_vars, action_vars, adv_vars, expert_action_vars
        else:
            return obs_vars, action_vars, adv_vars, rewards_vars, returns_vars, expert_action_vars # path_lengths_vars before expert action


    @overrides
    def init_opt(self):
        assert not int(self.policy.recurrent)  # not supported
        assert self.use_maml  # only maml supported

        dist = self.policy.distribution

        old_dist_info_vars, old_dist_info_vars_list = [], []  # TODO: I think this should be in the make_vars function
        for i in range(self.meta_batch_size):
            old_dist_info_vars.append({
                k: tf.placeholder(tf.float32, shape=[None] + list(shape), name='old_%s_%s' % (i, k))
                for k, shape in dist.dist_info_specs
                })
            old_dist_info_vars_list += [old_dist_info_vars[i][k] for k in dist.dist_info_keys]

        theta0_dist_info_vars, theta0_dist_info_vars_list = [], []
        for i in range(self.meta_batch_size):
            theta0_dist_info_vars.append({
                k: tf.placeholder(tf.float32, shape=[None] + list(shape), name='theta0_%s_%s' % (i, k))
                for k, shape in dist.dist_info_specs
                })
            theta0_dist_info_vars_list += [theta0_dist_info_vars[i][k] for k in dist.dist_info_keys]

        theta_l_dist_info_vars, theta_l_dist_info_vars_list = [], []  #theta_l is the current beta step's pre-inner grad update params
        for i in range(self.meta_batch_size):
            theta_l_dist_info_vars.append({
                k: tf.placeholder(tf.float32, shape=[None] + list(shape), name='theta_l_%s_%s' % (i, k))
                for k, shape in dist.dist_info_specs
                })
            theta_l_dist_info_vars_list += [theta_l_dist_info_vars[i][k] for k in dist.dist_info_keys]


        state_info_vars, state_info_vars_list = {}, []  # TODO: is this needed?

        all_surr_objs, input_vars_list, inner_input_vars_list = [], [], []
        new_params = []


        input_vars_list += tuple(theta0_dist_info_vars_list) + tuple(theta_l_dist_info_vars_list)
        inner_input_vars_list += tuple(theta0_dist_info_vars_list) + tuple(theta_l_dist_info_vars_list)

        for grad_step in range(self.num_grad_updates):  # we are doing this for all but the last step
            if not self.metalearn_baseline:
                obs_vars, action_vars, adv_vars, expert_action_vars = self.make_vars(str(grad_step))
            else:
                obs_vars, action_vars, adv_vars, rewards_vars, returns_vars, expert_action_vars = self.make_vars(str(grad_step))  # path_lengths_vars before expert actions

            inner_surr_objs, inner_surr_objs_sym = [], []  # surrogate objectives

            new_params = []
            kls = []

            for i in range(self.meta_batch_size):  # for training task T_i
                adv = adv_vars[i]
                if self.metalearn_baseline:
                    predicted_returns_sym, _ = self.baseline.predict_sym(obs_vars=obs_vars[i], all_params=self.baseline.all_params)
                    predicted_returns_means_sym = predicted_returns_sym['mean']
                    predicted_returns_log_std_sym = predicted_returns_sym['log_std']
                    baseline_pred_loss_i = tf.nn.l2_loss(predicted_returns_means_sym + predicted_returns_log_std_sym -returns_vars[i]) - 0.0 * tf.reduce_sum(predicted_returns_log_std_sym)
                    adv_sym = self.baseline.build_adv_sym(obs_vars=obs_vars[i],
                                                      rewards_vars=rewards_vars[i],
                                                      returns_vars=returns_vars[i],
                                                      baseline_pred_loss=baseline_pred_loss_i,
                                                      # path_lengths_vars=path_lengths_vars[i],
                                                      all_params=self.baseline.all_params)

                dist_info_vars_i, params = self.policy.dist_info_sym(obs_vars[i], state_info_vars, all_params=self.policy.all_params)
                if self.kl_constrain_step == 0:
                    kl = dist.kl_sym(old_dist_info_vars[i], dist_info_vars_i)
                    kls.append(kl)
                new_params.append(params)
                logli_i = dist.log_likelihood_sym(action_vars[i], dist_info_vars_i)
                lr = dist.likelihood_ratio_sym(action_vars[i], theta0_dist_info_vars[i], theta_l_dist_info_vars[i])
                # lr1 = dist.likelihood_ratio_sym(action_vars[i], theta0_dist_info_vars[i], dist_info_vars_i)
                # lr = tf.clip_by_value(lr,0.5,2.0)
                lr = self.ism(lr)
                # formulate a minimization problem
                # The gradient of the surrogate objective is the policy gradient
                inner_surr_objs.append(-tf.reduce_mean(tf.multiply(tf.multiply(logli_i, lr,"debug2"), adv, "debug3")))
                if self.metalearn_baseline:
                    inner_surr_objs_sym.append(-tf.reduce_mean(tf.multiply(tf.multiply(logli_i, lr,"debug4"), adv_sym, "debug5")))
                # inner_surr_objs.append(-tf.reduce_mean(lr * adv_vars[i]))
            inner_input_vars_list += obs_vars + action_vars + adv_vars
            if not self.metalearn_baseline:
                input_vars_list += obs_vars + action_vars + adv_vars
            else:
                input_vars_list += obs_vars + action_vars + rewards_vars + returns_vars  # + path_lengths_vars
            # For computing the fast update for sampling
            # At this point, inner_input_vars_list is theta0 + theta_l + obs + action + adv
            self.policy.set_init_surr_obj(inner_input_vars_list, inner_surr_objs)

            input_vars_list += expert_action_vars # TODO: is this pre-update expert action vars? Should we kill this?
            if not self.metalearn_baseline:
                all_surr_objs.append(inner_surr_objs)
            else:
                all_surr_objs.append(inner_surr_objs_sym)

        # last inner grad step
        if not self.metalearn_baseline:
            obs_vars, action_vars, _, expert_action_vars = self.make_vars('test')  # adv_vars was here instead of _
        else:
            obs_vars, action_vars, _, _, _, expert_action_vars = self.make_vars('test')
        surr_objs = []
        for i in range(self.meta_batch_size):  # here we cycle through the last grad update but for validation tasks (i is the index of a task)
            dist_info_vars_i, _ = self.policy.updated_dist_info_sym(task_id=i,surr_obj=all_surr_objs[-1][i],new_obs_var=obs_vars[i], params_dict=new_params[i])
            if self.kl_constrain_step == -1:  # if we only care about the kl of the last step, the last item in kls will be the overall
                kl = dist.kl_sym(old_dist_info_vars[i], dist_info_vars_i)
                kls.append(kl)  # we either get kl from here or from kl_constrain_step =0

            # here we define the loss for meta-gradient
            e = expert_action_vars[i]
            s = dist_info_vars_i["log_std"]
            m = dist_info_vars_i["mean"]
            surr_objs.append(tf.reduce_mean(self.l2loss_std_multiplier*(tf.square(tf.exp(s)))+tf.square(m)-2*tf.multiply(m,e)))
            # surr_objs.append(tf.reduce_mean(tf.exp(s)**2+m**2-2*m*e))
            # surr_objs.append(tf.reduce_mean(m**2-2*m*e))

        surr_obj = tf.reduce_mean(tf.stack(surr_objs, 0))  # mean over all the different tasks
        input_vars_list += obs_vars + action_vars + expert_action_vars + old_dist_info_vars_list  # +adv_vars # TODO: kill action_vars from this list, and if we're not doing kl, kill old_dist_info_vars_list too

        mean_kl = tf.reduce_mean(tf.concat(kls, 0))

        self.optimizer.update_opt(
            loss=surr_obj,
            target=None,
            # target=(self.policy, self.baseline),
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_vars_list,
            constraint_name="mean_kl"
        )
        return dict()


#######################################
    @overrides
    def optimize_policy(self, itr, all_samples_data):
        assert len(all_samples_data) == self.num_grad_updates + 1  # we collected the rollouts to compute the grads and then the test!
        assert self.use_maml

        input_vals_list = []

        # Code to account for off-policy sampling when more than 1 beta steps
        theta0_dist_info_list = []
        for i in range(self.meta_batch_size):
            if 'agent_infos_orig' not in all_samples_data[0][i].keys():
                assert False, "agent_infos_orig is missing--this should have been handled in batch_maml_polopt"
            else:
                agent_infos_orig = all_samples_data[0][i]['agent_infos_orig']
            theta0_dist_info_list += [agent_infos_orig[k] for k in self.policy.distribution.dist_info_keys]
        input_vals_list += tuple(theta0_dist_info_list)

        theta_l_dist_info_list = []
        for i in range(self.meta_batch_size):
            agent_infos = all_samples_data[0][i]['agent_infos']
            theta_l_dist_info_list += [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        input_vals_list += tuple(theta_l_dist_info_list)

        for step in range(self.num_grad_updates):
            obs_list, action_list, adv_list, rewards_list, returns_list, path_lengths_list, expert_action_list = [], [], [], [], [], [], []
            for i in range(self.meta_batch_size):  # for each task
                if not self.metalearn_baseline:
                    inputs = ext.extract(
                        all_samples_data[step][i],
                        "observations", "actions", "advantages", "expert_actions",
                    )
                    obs_list.append(inputs[0])
                    action_list.append(inputs[1])
                    adv_list.append(inputs[2])
                    expert_action_list.append(inputs[3])
                else:
                    inputs = ext.extract(
                        all_samples_data[step][i],
                        "observations", "actions", "rewards", "returns", "expert_actions", "paths"
                    )
                    obs_list.append(inputs[0])
                    action_list.append(inputs[1])
                    rewards_list.append(inputs[2])
                    returns_list.append(inputs[3])
                    expert_action_list.append(inputs[4])
                    # path_lengths_list.append([len(p['rewards']) for p in inputs[5]])
            if not self.metalearn_baseline:
                input_vals_list += obs_list + action_list + adv_list + expert_action_list
            else:
                input_vals_list += obs_list + action_list + rewards_list + returns_list  + expert_action_list  #+ path_lengths_list before expert action list


        for step in [self.num_grad_updates]:  # last step
            obs_list, action_list, expert_action_list = [], [], []  # last step's adv_list not currently used in maml_il
            for i in range(self.meta_batch_size):  # for each task
                inputs = ext.extract(
                    all_samples_data[step][i],
                    "observations", "actions", "expert_actions",
                )
                obs_list.append(inputs[0])
                action_list.append(inputs[1])
                expert_action_list.append(inputs[2])

            input_vals_list += obs_list + action_list + expert_action_list

        # Code to compute the kl distance, kind of pointless on non-testing iterations as agent_infos are zeroed out on expert traj samples
        dist_info_list = []
        for i in range(self.meta_batch_size):
            agent_infos = all_samples_data[self.kl_constrain_step][i]['agent_infos']  ##kl_constrain_step default is -1, meaning post all alpha grad updates
            dist_info_list += [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        input_vals_list += tuple(dist_info_list)  # This populates old_dist_info_vars_list

      #  logger.log("Computing KL before")
      #  mean_kl_before = self.optimizer.constraint_val(input_vals_list)  # TODO: need to make sure the input list has the correct form. Maybe start naming the input lists based on what they're needed for

        logger.log("Computing loss before")
       # loss_before = self.optimizer.loss(input_vals_list)
        if itr not in TESTING_ITRS:
            logger.log("Optimizing")
            self.optimizer.optimize(input_vals_list)
        else:
            logger.log("Not Optimizing")
        logger.log("Computing loss after")
      #  loss_after = self.optimizer.loss(input_vals_list)
      #  logger.log("Computing KL after")
       # mean_kl = self.optimizer.constraint_val(input_vals_list)
       # logger.record_tabular('MeanKLBefore', mean_kl_before)
       # logger.record_tabular('MeanKL', mean_kl)
      #  logger.record_tabular('LossBefore', loss_before)
      #  logger.record_tabular('LossAfter', loss_after)
      #  logger.record_tabular('dLoss', loss_before - loss_after)
        # getting rid of the above because of issues with tabular
        return dict()


    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )






