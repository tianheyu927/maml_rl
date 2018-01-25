import numpy as np

from rllab.core.serializable import Serializable
from rllab.core.parameterized import Parameterized
from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.regressors.maml_gaussian_mlp_regressor import MAMLGaussianMLPRegressor
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
from sandbox.rocky.tf.optimizers.quad_dist_expert_optimizer import QuadDistExpertOptimizer
from sandbox.rocky.tf.distributions.diagonal_gaussian import DiagonalGaussian  # This is just a util class. No params.

from collections import OrderedDict
from sandbox.rocky.tf.misc import tensor_utils
from tensorflow.contrib.layers.python import layers as tf_layers
from rllab.misc.tensor_utils import flatten_tensors, unflatten_tensors

from sandbox.rocky.tf.core.utils import make_input, make_dense_layer, forward_dense_layer, make_param_layer, \
    forward_param_layer

import tensorflow as tf

class MAMLGaussianMLPBaseline(Baseline, Parameterized, Serializable):

    def __init__(
            self,
            env_spec,
            subsample_factor=1.,
            num_seq_inputs=1,
            learning_rate=0.01,
            algo_discount=0.99,
            hidden_sizes=(32,32),
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.identity,
            init_std=1.01,

    ):
        Serializable.quick_init(self, locals())

        self.env_spec = env_spec
        obs_dim = env_spec.observation_space.flat_dim
        self.action_dim = env_spec.action_space.flat_dim
        self.n_hidden = len(hidden_sizes)
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity
        self.input_shape = (None, obs_dim,)
        self.learning_rate = learning_rate
        self.algo_discount = algo_discount



        self.all_params = self.create_MLP(
            name="mean_baseline_network",
            output_dim=1,
            hidden_sizes=hidden_sizes,
        )
        self.input_tensor, _ = self.forward_MLP('mean_baseline_network', self.all_params, reuse=None)
        forward_mean = lambda x, params, is_train: self.forward_MLP('mean_baseline_network',all_params=params, input_tensor=x, is_training=is_train)[1]

        init_log_std = np.log(init_std)
        self.all_params['std_param'] = make_param_layer(
            num_units=1,
            param=tf.constant_initializer(init_log_std),
            name="output_bas_std_param",
            trainable=False,
        )
        forward_std = lambda x, params: forward_param_layer(x, params['std_param'])
        self.all_param_vals = None

        self._forward = lambda obs, params, is_train: (forward_mean(obs, params, is_train), forward_std(obs, params))

        self._dist = DiagonalGaussian(1)

        self._cached_params = {}

        super(MAMLGaussianMLPBaseline, self).__init__(env_spec)

        predict_sym = self.predict_sym(obs_vars=self.input_tensor)
        mean_var = predict_sym['mean']
        log_std_var = predict_sym['log_std']

        self._init_f_dist = tensor_utils.compile_function(
            inputs=[self.input_tensor],
            outputs=[mean_var,log_std_var],
        )
        self._cur_f_dist = self._init_f_dist


    @property
    def vectorized(self):
        return True


    def set_init_surr_obj(self, input_list, surr_obj_tensor):
        """ Set the surrogate objectives used the update the policy
        """
        self.input_list_for_grad = input_list
        self.surr_obj = surr_obj_tensor

    @overrides
    def fit(self, paths, log=True):
        # return True

        if 'surr_obj' not in dir(self):
            assert False, "why didn't we define it already"


        """Equivalent of compute_updated_dists"""
        param_keys = self.all_params.keys()
        update_param_keys = param_keys
        no_update_param_keys = []

        sess = tf.get_default_session()

        obs = np.concatenate([p["observations"] for p in paths])
        print("debug43", np.shape(obs))
        returns = np.concatenate([p["returns"] for p in paths])

        inputs = [obs] + [returns]

        init_param_values = None
        if self.all_param_vals is not None:
            init_param_values = self.get_variable_values(self.all_params)

        if self.all_param_vals is not None:
            self.assign_params(self.all_params,self.all_param_vals)



        if 'all_fast_params_tensor' not in dir(self):
            # make computation graph once
            self.all_fast_params_tensor = []
            gradients = dict(zip(update_param_keys, tf.gradients(self.surr_obj, [self.all_params[key] for key in update_param_keys])))
            fast_params_tensor = OrderedDict(zip(update_param_keys, [self.all_params[key] - self.learning_rate*gradients[key] for key in update_param_keys]))
            for k in no_update_param_keys:
                fast_params_tensor[k] = self.all_params[k]
            self.all_fast_params_tensor.append(fast_params_tensor)

            # pull new param vals out of tensorflow, so gradient computation only done once
            # first is the vars, second the values
            # these are the updated values of the params after the gradient step
        self.all_param_vals = sess.run(self.all_fast_params_tensor,
                                           feed_dict=dict(list(zip(self.input_list_for_grad, inputs))))
        print("debug57", type(self.all_param_vals))

        if init_param_values is not None:
            self.assign_params(self.all_params, init_param_values)

        inputs = tf.split(self.input_tensor, 1, 0)  #TODO: how to convert this since we don't need to calculate multiple updates simultaneously
        task_inp = inputs
        info, _ = self.predict_sym(obs_vars=dict(), all_params=self.all_param_vals,is_training=False)

        outputs = [info['mean'], info['log_std']]

        self._cur_f_dist = tensor_utils.compile_function(
            inputs=[self.input_tensor],
            outputs=outputs,
        )
        #  logger.record_tabular("ComputeUpdatedDistTime", total_time)

        # # self._preupdate_params = self._regressor.get_param_values()
        # observations = np.concatenate([p["observations"] for p in paths])
        # returns = np.concatenate([p["returns"] for p in paths])
        # TODO self.fit(observations, returns.reshape((-1, 1)), log=log)



    def get_variable_values(self, tensor_dict):
        sess = tf.get_default_session()
        result = sess.run(tensor_dict)
        return result

    def assign_params(self, tensor_dict, param_values):
        print("debug44", type(tensor_dict))
        print("debug45", type(param_values))
        if 'assign_placeholders' not in dir(self):
            # make computation graph, if it doesn't exist; then cache it for future use.
            self.assign_placeholders = {}
            self.assign_ops = {}
            for key in tensor_dict.keys():
                self.assign_placeholders[key] = tf.placeholder(tf.float32)
                self.assign_ops[key] = tf.assign(tensor_dict[key], self.assign_placeholders[key])

        feed_dict = {self.assign_placeholders[key]:param_values[key] for key in tensor_dict.keys()}
        sess = tf.get_default_session()
        sess.run(self.assign_ops, feed_dict)

    @overrides
    def predict(self, path):
        # flat_obs = self.env_spec.observation_space.flatten_n(path['observations'])
        obs = path['observations']
        result = self._cur_f_dist(obs)
        if len(result) == 2:
            means, log_stds = result
        else:
            raise NotImplementedError('Not supported.')
        return np.reshape(means, [-1])


    @property
    def distribution(self):
        return self._dist

    def get_params_internal(self, all_params=False, **tags):
        if tags.get('trainable', False):
            params = tf.trainable_variables()
        else:
            params = tf.global_variables()

        params = [p for p in params if p.name.startswith('mean_baseline_network') or p.name.startswith('output_bas_std_param')]
        params = [p for p in params if 'Adam' not in p.name]

        return params


        # This makes all of the parameters.
    def create_MLP(self, name, output_dim, hidden_sizes,
                   hidden_W_init=tf_layers.xavier_initializer(), hidden_b_init=tf.zeros_initializer(),
                   output_W_init=tf_layers.xavier_initializer(), output_b_init=tf.zeros_initializer(),
                   weight_normalization=False,
                   ):
        all_params = OrderedDict()

        cur_shape = self.input_shape
        with tf.variable_scope(name):
            for idx, hidden_size in enumerate(hidden_sizes):
                W, b, cur_shape = make_dense_layer(
                    cur_shape,
                    num_units=hidden_size,
                    name="hidden_%d" % idx,
                    W=hidden_W_init,
                    b=hidden_b_init,
                    weight_norm=weight_normalization,
                )
                all_params['W' + str(idx)] = W
                all_params['b' + str(idx)] = b
            W, b, _ = make_dense_layer(
                cur_shape,
                num_units=output_dim,
                name='output',
                W=output_W_init,
                b=output_b_init,
                weight_norm=weight_normalization,
            )
            all_params['W' + str(len(hidden_sizes))] = W
            all_params['b' + str(len(hidden_sizes))] = b

        return all_params

    def forward_MLP(self, name, all_params, input_tensor=None,
                    batch_normalization=False, reuse=True, is_training=False):
        # is_training and reuse are for batch norm, irrelevant if batch_norm set to False
        # set reuse to False if the first time this func is called.
        with tf.variable_scope(name):
            if input_tensor is None:
                l_in = make_input(shape=self.input_shape, input_var=None, name='input')
            else:
                l_in = input_tensor

            l_hid = l_in

            for idx in range(self.n_hidden):
                l_hid = forward_dense_layer(l_hid, all_params['W' + str(idx)], all_params['b' + str(idx)],
                                            batch_norm=batch_normalization,
                                            nonlinearity=self.hidden_nonlinearity,
                                            scope=str(idx), reuse=reuse,
                                            is_training=is_training
                                            )
            output = forward_dense_layer(l_hid, all_params['W' + str(self.n_hidden)],
                                         all_params['b' + str(self.n_hidden)],
                                         batch_norm=False, nonlinearity=self.output_nonlinearity,
                                         )
            return l_in, output



    def get_params(self, all_params=False, **tags):
        """
        Get the list of parameters (symbolically), filtered by the provided tags.
        Some common tags include 'regularizable' and 'trainable'
        """
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_params:
            self._cached_params[tag_tuple] = self.get_params_internal(all_params, **tags)
        return self._cached_params[tag_tuple]

    def get_param_values(self, all_params=False, **tags):
        params = self.get_params(all_params, **tags)
        param_values = tf.get_default_session().run(params)
        return flatten_tensors(param_values)

    def switch_to_init_dist(self):
        # switch cur baseline distribution to pre-update baseline
        self._cur_f_dist = self._init_f_dist
        self.all_param_vals = None

    def predict_sym(self, obs_vars, all_params=None, is_training=True):
        """equivalent of dist_info_sym, this function constructs the tf graph, only called
        during beginning of meta-training"""
        return_params = True
        if all_params is None:
            return_params = False
            all_params = self.all_params
            if self.all_params is None:
                assert False, "Shouldn't get here"

        mean_var, std_param_var = self._forward(obs=obs_vars, params=all_params, is_train=is_training)

        if return_params:
            return dict(mean=mean_var, log_std=std_param_var), all_params
        else:
            return dict(mean=mean_var, log_std=std_param_var)

    def updated_predict_sym(self, baseline_pred_loss, obs_vars, params_dict=None):
        """ symbolically create post-fitting baseline predict_sym, to be used for meta-optimization.
        Equivalent of updated_dist_info_sym"""
        old_params_dict = params_dict

        if old_params_dict is None:
            old_params_dict = self.all_params
        param_keys = self.all_params.keys()

        update_param_keys = param_keys
        print("debug11", param_keys)
        no_update_param_keys = []
        grads = tf.gradients(baseline_pred_loss, [old_params_dict[key] for key in update_param_keys])
        print("debug12", grads)
        gradients = dict(zip(update_param_keys, grads))
        params_dict = dict(zip(update_param_keys, [old_params_dict[key] - self.learning_rate * gradients[key] for key in update_param_keys]))
        for k in no_update_param_keys:
            params_dict[k] = old_params_dict[k]
        return self.predict_sym(obs_vars=obs_vars, all_params=params_dict)

    def build_adv_sym(self,obs_vars,rewards_vars, returns_vars, baseline_pred_loss, all_params):  # path_lengths_vars was before all_params


        predicted_returns_vars, _ = self.updated_predict_sym(baseline_pred_loss=baseline_pred_loss, obs_vars=obs_vars, params_dict=all_params)


        organized_rewards = tf.reshape(rewards_vars, [-1,100])
        organized_pred_returns = tf.reshape(predicted_returns_vars['mean'] + 0.0 * predicted_returns_vars['log_std'], [-1,100])
        organized_pred_returns_ = tf.concat((organized_pred_returns[:,1:], tf.reshape(tf.zeros(tf.shape(organized_pred_returns[:,0])),[-1,1])),axis=1)

        deltas = organized_rewards + self.algo_discount * organized_pred_returns_ - organized_pred_returns
        adv_vars = tf.map_fn(lambda x: discount_cumsum_sym(x, self.algo_discount), deltas)

        adv_vars = tf.reshape(adv_vars, [-1])
        adv_vars = (adv_vars - tf.reduce_mean(adv_vars))/tf.sqrt(tf.reduce_mean(adv_vars**2))  # centering advantages

        return adv_vars

    @overrides
    def set_param_values(self, flattened_params, **tags):
        raise NotImplementedError("todo")

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



def discount_cumsum_sym(var, discount):
    # y[0] = x[0] + discount * x[1] + discount**2 * x[2] + ...
    # y[1] = x[1] + discount * x[2] + discount**2 * x[3] + ...
    discount = tf.cast(discount, tf.float32)
    range_ = tf.cast(tf.range(tf.size(var)), tf.float32)
    var_ = var * tf.pow(discount, range_)
    return tf.cumsum(var_,reverse=True) * tf.pow(discount,-range_)




