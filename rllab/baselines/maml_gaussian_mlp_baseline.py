import numpy as np

from rllab.core.serializable import Serializable
from rllab.core.parameterized import Parameterized
from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides
# from rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from sandbox.rocky.tf.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
from sandbox.rocky.tf.optimizers.quad_dist_expert_optimizer import QuadDistExpertOptimizer

from collections import OrderedDict
from sandbox.rocky.tf.misc import tensor_utils

import tensorflow as tf

class MAMLGaussianMLPBaseline(Baseline, Parameterized, Serializable):

    def __init__(
            self,
            env_spec,
            subsample_factor=1.,
            num_seq_inputs=1,
            regressor_args=None,
            learning_rate=0.01,
            algo_discount=0.99,
    ):
        Serializable.quick_init(self, locals())
        super(MAMLGaussianMLPBaseline, self).__init__(env_spec)
        if regressor_args is None:
            regressor_args = dict()

        self._regressor = GaussianMLPRegressor(
            input_shape=(env_spec.observation_space.flat_dim * num_seq_inputs,),
            output_dim=1,
            name="vf",
            **regressor_args
        )
        self.learning_rate = learning_rate
        self.algo_discount = algo_discount
        # self._preupdate_params = None
        with tf.Session() as sess:
            with tf.variable_scope("vf"):
                # initialize uninitialized vars  (only initialize vars that were not loaded)
                uninit_vars = []
                for var in tf.global_variables():
                    # note - this is hacky, may be better way to do this in newer TF.
                    try:
                        sess.run(var)
                    except tf.errors.FailedPreconditionError:
                        uninit_vars.append(var)
                sess.run(tf.variables_initializer(uninit_vars))
                self.all_params = self._regressor.get_param_values()

        self.all_params = OrderedDict({x.name:x for x in self._regressor.get_params()})
        print("debug23,", self.all_params.keys())
        print("debug23,", self.all_params)

        self.all_param_vals = None

    @overrides
    def fit(self, paths, log=True):
        # self._preupdate_params = self._regressor.get_param_values()
        observations = np.concatenate([p["observations"] for p in paths])
        returns = np.concatenate([p["returns"] for p in paths])
        self._regressor.fit(observations, returns.reshape((-1, 1)), log=log)



    @overrides
    def predict(self, path):
        return self._regressor.predict(path["observations"]).flatten()

    @overrides
    def get_param_values(self, **tags):
        return self._regressor.get_param_values(**tags)

    @overrides
    def set_param_values(self, flattened_params, **tags):
        self._regressor.set_param_values(flattened_params, **tags)
    #
    # def revert(self):
    #     # assert self._preupdate_params is not None, "already reverted"
    #     if self._preupdate_params is None:
    #         return
    #     else:
    #         self._regressor.set_param_values(self._preupdate_params)
    #         self._preupdate_params = None

    # def compute_updated_baseline(self, samples):
    #     """ Compute fast gradients once per iteration and pull them out of tensorflow for sampling with the post-update policy.
    #     """
    #     num_tasks = len(samples)
    #     param_keys = self.all_params.keys()
    #     update_param_keys = param_keys
    #     no_update_param_keys = []
    #
    #     sess = tf.get_default_session()
    #
    #
    #
    #     for i in range(num_tasks):
    #
    #
    #     self._cur_f_dist = tensor_utils.compile_function


    def predict_sym(self, obs_vars, all_params=None):
        """equivalent of dist_info_sym"""
        return_params = True
        if all_params is None:
            return_params = False
            all_params = self.all_params
            if self.all_params is None:
                assert False, "too bad"

        predicted_returns_vars = self._regressor._f_predict_sym(xs=obs_vars, params=all_params)
        # TODO: regressor will predict the rewards, not the returns

        if return_params:
            return predicted_returns_vars, all_params
        else:
            return predicted_returns_vars

    # @overrides
    # def fit(self, paths, log=True):  # aka compute updated baseline
    #     # self._preupdate_params = self._regressor.get_param_values()
    #
    #     param_keys = self.all_params.keys()
    #     update_param_keys = param_keys
    #     no_update_param_keys = []
    #     sess = tf.get_default_session()
    #
    #     observations = np.concatenate([p["observations"] for p in paths])
    #     returns = np.concatenate([p["returns"] for p in paths])
    #
    #     inputs = observations + returns
    #
    #
    #     learning_rate = self.learning_rate
    #     if self.all_param_vals is not None:
    #         self.assign_params(self.all_params, self.all_param_vals)
    #
    #     if "fit_tensor" not in dir(self):
    #         gradients = dict(zip(update_param_keys, tf.gradients(self._regressor.loss_sym, [self.all_params[key] for key in update_param_keys])))
    #         self.fit_tensor = OrderedDict(zip(update_param_keys,
    #                                              [self.all_params[key] - learning_rate * gradients[key] for key in
    #                                               update_param_keys]))
    #         for k in no_update_param_keys:
    #             self.fit_tensor[k] = self.all_params[k]
    #
    #     self.all_param_vals = sess.run(self.fit_tensor, feed_dict = dict(list(zip(self.input_list_for_grad, inputs))))
    #
    #
    #     inputs = self.input_tensor
    #     task_inp = inputs
    #     output = self.predict_sym(task_inp, dict(),all_params=self.all_param_vals, is_training=False)
    #
    #
    #     self._regressor._f_predict = tensor_utils.compile_function(inputs=[self.input_tensor], outputs=output)


    def updated_predict_sym(self, baseline_pred_obj, obs_vars, params_dict=None):
        """ symbolically create post-fitting baseline predict_sym, to be used for meta-optimization.
        Equivalent of updated_dist_info_sym"""
        old_params_dict = params_dict


        if old_params_dict is None:
            old_params_dict = self.all_params
        param_keys = self.all_params.keys()

        update_param_keys = param_keys
        no_update_param_keys = []
        grads = tf.gradients(baseline_pred_obj, [old_params_dict[key] for key in update_param_keys])

        gradients = dict(zip(update_param_keys, grads))
        params_dict = dict(zip(update_param_keys, [old_params_dict[key] - self.learning_rate * gradients[key] for key in update_param_keys]))
        for k in no_update_param_keys:
            params_dict[k] = old_params_dict[k]
        return self.predict_sym(obs_vars=obs_vars, all_params = params_dict)

    def build_adv_sym(self,obs_vars,rewards_vars, returns_vars, all_params):  # path_lengths_vars was before all_params

        baseline_pred_obj = self._regressor.loss_sym  # baseline prediction objective

        predicted_returns_vars, _ = self.updated_predict_sym(baseline_pred_obj=baseline_pred_obj, obs_vars=obs_vars, params_dict=all_params)
        # TODO: predicted_returns_vars should be a list of predicted returns organized by path
        organized_rewards = tf.reshape(rewards_vars, [-1,100])
        organized_pred_returns = tf.reshape(predicted_returns_vars, [-1,100])
        organized_pred_returns_ = tf.concat((organized_pred_returns[:,1:], tf.reshape(tf.zeros(tf.shape(organized_pred_returns[:,0])),[-1,1])),axis=1)
        # organized_pred_returns = tf.map_fn(lambda x: discount_cumsum_sym(x, self.algo_discount), organized_pred_rewards)

        deltas = organized_rewards + self.algo_discount * organized_pred_returns_ - organized_pred_returns
        adv_vars = tf.map_fn(lambda x: discount_cumsum_sym(x, self.algo_discount), deltas)

        adv_vars = tf.reshape(adv_vars, [-1])

        return adv_vars



def discount_cumsum_sym(var, discount):
    # y[0] = x[0] + discount * x[1] + discount**2 * x[2] + ...
    # y[1] = x[1] + discount * x[2] + discount**2 * x[3] + ...
    discount = tf.cast(discount, tf.float32)
    range_ = tf.cast(tf.range(tf.size(var)), tf.float32)
    var_ = var * tf.pow(discount, range_)
    return tf.cumsum(var_,reverse=True) * tf.pow(discount,-range_)
