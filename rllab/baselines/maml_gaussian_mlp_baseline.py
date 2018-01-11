import numpy as np

from rllab.core.serializable import Serializable
from rllab.core.parameterized import Parameterized
from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides
# from rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from sandbox.rocky.tf.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
from collections import OrderedDict

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
            hidden_sizes=(1,),
            hidden_nonlinearity=tf.identity,
            optimizer=FirstOrderOptimizer(
                learning_rate=learning_rate,
            ),
            use_trust_region=False,
            learn_std=False,
            init_std=1.0,
            name="vf",
            **regressor_args
        )
        self.learning_rate = learning_rate
        self.algo_discount = algo_discount
        self._preupdate_params = None
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
        print("debug23,", type(self.all_params))

    @overrides
    def fit(self, paths, log=True):
        self._preupdate_params = self._regressor.get_param_values()
        observations = np.concatenate([p["observations"] for p in paths])
        returns = np.concatenate([p["returns"] for p in paths])
        self._regressor.fit(observations, returns.reshape((-1, 1)))

    @overrides
    def predict(self, path):
        return self._regressor.predict(path["observations"]).flatten()

    @overrides
    def get_param_values(self, **tags):
        return self._regressor.get_param_values(**tags)

    @overrides
    def set_param_values(self, flattened_params, **tags):
        self._regressor.set_param_values(flattened_params, **tags)

    def revert(self):
        # assert self._preupdate_params is not None, "already reverted"
        if self._preupdate_params is None:
            return
        else:
            self._regressor.set_param_values(self._preupdate_params)
            self._preupdate_params = None

    def updated_baseline_sym(self, baseline_pred_obj, obs_vars, params_dict=None):
        """ symbolically create post-fitting baseline params, to be used for meta-optimization.
        NOTE this function generates predicted rewards, which then need to be organized into predicted returns"""
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

    def predict_sym(self, obs_vars, all_params=None):
        return_params = True
        if all_params is None:
            return_params = False
            all_params = self.all_params

        predicted_rewards_vars = self._regressor._f_predict_sym(xs=obs_vars, params=all_params)
        # TODO: regressor will predict the rewards, not the returns

        if return_params:
            return predicted_rewards_vars, all_params
        else:
            return predicted_rewards_vars

    def build_adv_sym(self,obs_vars,rewards_vars, returns_vars, path_lengths_vars, all_params):

        baseline_pred_obj = self._regressor.loss_sym  # baseline prediction objective

        predicted_rewards_vars, _ = self.updated_baseline_sym(baseline_pred_obj=baseline_pred_obj, obs_vars=obs_vars, params_dict=all_params)
        # TODO: predicted_returns_vars should be a list of predicted returns organized by path
        organized_pred_rewards = tf.reshape(predicted_rewards_vars, [-1,path_lengths_vars])
        organized_pred_rewards_ = tf.concat((organized_pred_rewards[:,1:], tf.zeros(tf.shape(organized_pred_rewards)[:-1])),axis=1)
        organized_pred_returns = tf.map_fn(discount_cumsum_sym, organized_pred_rewards)



        adv_vars = []






        for i, path_var in enumerate(path_rewards_vars):
            print("debug24", predicted_rewards_vars)
            print("debug24", predicted_rewards_vars[i])

            predicted_returns_var = discount_cumsum_sym(predicted_rewards_vars[i],self.algo_discount)
            print("debug25", predicted_returns_var)
            print("debug25", predicted_returns_var[1:])

            predicted_returns_var_ = tf.concat([predicted_returns_var[1:], tf.constant([0.0])],1)
            deltas_var = rewards_var + self.algo_discount * predicted_returns_var_ - predicted_returns_var
            adv_var = discount_cumsum_sym(deltas_var, self.algo_discount)
            adv_vars.append(adv_var)

        return adv_vars

def discount_cumsum_sym(var, discount):
    # y[0] = x[0] + discount * x[1] + discount**2 * x[2] + ...
    # y[1] = x[1] + discount * x[2] + discount**2 * x[3] + ...
    discount = tf.cast(discount, tf.float32)
    range_ = tf.cast(tf.range(tf.size(var)), tf.float32)
    var_ = var * tf.pow(discount, range_)
    return tf.cumsum(var_,reverse=True) * tf.pow(discount,-range_)
