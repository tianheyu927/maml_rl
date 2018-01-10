import numpy as np

from rllab.core.serializable import Serializable
from rllab.core.parameterized import Parameterized
from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides
# from rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from sandbox.rocky.tf.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer

import tensorflow as tf

class MAMLGaussianMLPBaseline(Baseline, Parameterized, Serializable):

    def __init__(
            self,
            env_spec,
            subsample_factor=1.,
            num_seq_inputs=1,
            regressor_args=None,
            learning_rate=0.01,
    ):
        Serializable.quick_init(self, locals())
        super(MAMLGaussianMLPBaseline, self).__init__(env_spec)
        if regressor_args is None:
            regressor_args = dict()

        self._regressor = GaussianMLPRegressor(
            input_shape=(env_spec.observation_space.flat_dim * num_seq_inputs,),
            output_dim=1,
            optimizer=FirstOrderOptimizer(
                learning_rate=learning_rate,
            ),
            use_trust_region=False,
            learn_std=False,
            init_std=0.0,
            name="vf",
            **regressor_args
        )
        self.learning_rate = learning_rate
        self._preupdate_params = None
        self.all_params = self._regressor.get_param_values()

        self.all_params = self.create_MLP(  # TODO: this should not be a method of the policy! --> helper
            name="vf",
            input_
        output_dim = output_dim,
                     hidden_sizes = hidden_sizes,
        )
        print("debug23," type(self.all_params))

    @overrides
    def fit(self, paths, log=True):
        self._preupdate_params = self._regressor.get_param_values()
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

    def revert(self):
        assert self._preupdate_params is not None, "already reverted"
        self._regressor.set_param_values(self._preupdate_params)
        self._preupdate_params = None

    def updated_baseline_sym(self, baseline_pred_obj, obs_var, params_dict=None, is_training=True):
        """ symbolically create post-fitting baseline params, to be used for meta-optimization"""
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

        return self.predict_sym(obs_var, all_params = params_dict, is_training=is_training)

    def predict_sym(self, obs_var, all_params=None, is_training=True):
        return_params = True
        if all_params is None:
            return_params = False
            all_params = self.all_params

        predicted_returns_var = self._regressor._f_predict_sym(obs_var, all_params) # TODO
        if return_params:
            return predicted_returns_var, all_params
        else:
            return predicted_returns_var

    def build_adv_sym(self,obs_vars,rewards_vars, returns_vars):
        deltas_vars = rewards_vars + self.algo.discount * predicted_returns_vars_shifted_by_1 - predicted_returns_vars

        adv_vars = discount_cumsum_sym(deltas, self.algo.discount)