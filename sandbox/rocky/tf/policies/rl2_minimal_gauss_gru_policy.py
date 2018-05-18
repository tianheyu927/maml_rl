import time
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers

from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc import logger
from rllab.misc.overrides import overrides
from rllab.misc.tensor_utils import flatten_tensors, unflatten_tensors
from sandbox.rocky.tf.core.utils import make_input, make_dense_layer, forward_dense_layer, make_param_layer, \
    forward_param_layer, make_gru_layer
from sandbox.rocky.tf.distributions.recurrent_diagonal_gaussian import RecurrentDiagonalGaussian
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.spaces.box import Box
from tensorflow.python.framework import dtypes

load_params = True

class RL2GaussianGRUPolicy(StochasticPolicy, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            hidden_dim=32,
            fc_hidden_sizes=[]
            feature_network=None,
            state_include_action=True,
            state_include_reward_done=True,
            hidden_nonlinearity=tf.tanh,
            learn_std=True,
            init_std=1.0,
            output_nonlinearity=None,
            horizon=100,
            num_episodes=6
    ):
        """
        :param env_spec: A spec for the env.
        :param hidden_dim: dimension of hidden layer
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :return:
        """
        with tf.variable_scope(name):
            Serializable.quick_init(self, locals())
            super(RL2GaussianGRUPolicy, self).__init__(env_spec)

            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim

            if state_include_action:
                input_dim = obs_dim + action_dim
                if state_include_reward_done:
                    input_dim += 2
            else:
                input_dim = obs_dim
            
            self.action_dim = action_dim
            self.num_units = hidden_dim
            self.n_hidden = len(fc_hidden_sizes)
            self.hidden_nonlinearity = hidden_nonlinearity
            self.output_nonlinearity = output_nonlinearity
            self.input_shape = (None, None, input_dim)
            self.horizon = horizon
            self.num_episodes = num_episodes

            l_input = L.InputLayer(
                shape=self.input_shape,
                name="input"
            )

            if feature_network is None:
                feature_dim = input_dim
                l_flat_feature = None
                l_feature = l_input
            else:
                feature_dim = feature_network.output_layer.output_shape[-1]
                l_flat_feature = feature_network.output_layer
                l_feature = L.OpLayer(
                    l_flat_feature,
                    extras=[l_input],
                    name="reshape_feature",
                    op=lambda flat_feature, input: tf.reshape(
                        flat_feature,
                        tf.pack([tf.shape(input)[0], tf.shape(input)[1], feature_dim])
                    ),
                    shape_op=lambda _, input_shape: (input_shape[0], input_shape[1], feature_dim)
                )
            
            # create network
            if mean_network is None:
                self.all_params = self.create_GRU(
                    name='mean_network',
                    num_units=hidden_dim,
                    hidden_nonlinearity=tf.nn.relu,
                    hidden_init_trainable=True,
                    weight_normalization=True
                )
                MLP_params = self.create_MLP(  # TODO: this should not be a method of the policy! --> helper
                    name="mean_network",
                    output_dim=self.action_dim,
                    hidden_sizes=fc_hidden_sizes,
                )
                self.all_params.update(MLP_params)
                self.input_tensors, _, _ = self.forward_GRU('mean_network', self.all_params,
                    horizon=horizon, state=self.h0
                    reuse=None # Need to run this for batch norm
                )
                forward_mean = lambda x, hidden, params, is_train: self.forward_GRU('mean_network', state=state,
                    all_params=params, horizon=horizon, input_tensor=x, is_training=is_train)
            else:
                raise NotImplementedError('Not supported.')
    
            if std_network is not None:
                raise NotImplementedError('Not supported.')
            else:
                if adaptive_std:
                    raise NotImplementedError('Not supported.')
                else:
                    if std_parametrization == 'exp':
                        init_std_param = np.log(init_std)
                    elif std_parametrization == 'softplus':
                        init_std_param = np.log(np.exp(init_std) - 1)
                    else:
                        raise NotImplementedError
                    self.all_params['std_param'] = make_param_layer(
                        num_units=self.action_dim,
                        param=tf.constant_initializer(init_std_param),
                        name="output_std_param",
                        trainable=learn_std,
                    )
                    forward_std = lambda x, params: forward_param_layer(x, params['std_param'])
                self.all_param_vals = None
    
                # unify forward mean and forward std into a single function
                self._forward = lambda obs, hidden, params, is_train: (
                        forward_mean(obs, hidden, params, is_train), forward_std(obs, params))
    
                self.std_parametrization = std_parametrization
    
                if std_parametrization == 'exp':
                    min_std_param = np.log(min_std)
                    max_std_param = np.log(max_std)
                elif std_parametrization == 'softplus':
                    min_std_param = np.log(np.exp(min_std) - 1)
                    max_std_param = np.log(np.exp(max_std) - 1)
                else:
                    raise NotImplementedError
    
                self.min_std_param = min_std_param  # TODO: change these to min_std_param_raw
                self.max_std_param = max_std_param
                self.std_modifier = np.float64(std_modifier)
                #print("initializing max_std debug4", self.min_std_param, self.max_std_param)
    
    
                self._dist = RecurrentDiagonalGaussian(self.action_dim)
    
                self._cached_params = {}
    
                super(RL2GaussianGRUPolicy, self).__init__(env_spec)
    
                dist_info_sym, hidden = self.dist_info_sym(self.input_tensor, hidden=self.hprev, dict(), is_training=False)
                mean_var = dist_info_sym["mean"]
                log_std_var = dist_info_sym["log_std"]
    
                # pre-update policy
                self._init_f_dist = tensor_utils.compile_function(
                    inputs=[self.input_tensor, self.hprev],
                    outputs=[mean_var, log_std_var, hidden],
                )
                self._cur_f_dist = self._init_f_dist
                
    def compute_updated_dists(self, samples):
        """ Roll out the rnn at test time.
        """
        start = time.time()
        num_tasks = len(samples)

        outputs = []
        inputs = tf.split(self.input_tensor, num_tasks, 0)
        for i in range(num_tasks):
            # TODO - use a placeholder to feed in the params, so that we don't have to recompile every time.
            task_inp = inputs[i]
            info, hidden = self.dist_info_sym(obs_var=task_inp, hidden=self.hprev, state_info_vars=dict(),
                    is_training=False)

            outputs.append([info['mean'], info['log_std'], hidden])

        self._cur_f_dist = tensor_utils.compile_function(
            inputs=[self.input_tensor, self.hprev],
            outputs=outputs,
        )
        total_time = time.time() - start

    @overrides
    def dist_info_sym(self, obs_var, hidden, state_info_vars, all_params=None, is_training=True):
        if all_params is None:
            all_params = self.all_params
        n_batches = tf.shape(obs_var)[0]
        n_steps = tf.shape(obs_var)[1]
        obs_var = tf.reshape(obs_var, tf.pack([n_batches, n_steps, -1]))
        # if self.state_include_action:
        #     prev_action_var = state_info_vars["prev_action"]
        #     prev_reward_var = state_info_vars["prev_reward"]
        #     prev_done_var = state_info_vars["prev_done"]
        #     all_input_var = tf.concat(axis=2, values=[obs_var, prev_action_var, prev_reward_var, prev_done_var])
        # else:
            # all_input_var = obs_var
        mean_var, std_param_var = self._forward(obs_var, hidden, all_params, is_training)
        _, mean_var, hidden = mean_var
        if self.min_std_param is not None:
            std_param_var = tf.maximum(std_param_var, self.min_std_param)
        if self.max_std_param is not None:
            std_param_var = tf.minimum(std_param_var, self.max_std_param)
        if self.std_parametrization == 'exp':
            log_std_var = std_param_var + np.log(self.std_modifier)
        elif self.std_parametrization == 'softplus':
            log_std_var = tf.log(tf.log(1. + tf.exp(std_param_var))) + np.log(self.std_modifier)
        else:
            raise NotImplementedError
            
        return dict(mean=mean_var, log_std=log_std_var), hidden
        
    def updated_dist_info_sym(self, new_obs_vars, hidden, is_training=True):
        """ symbolically create RL^2 graph, for the meta-optimization, only called at the beginning of meta-training.
        """
        info = {}
        for t in range(self.num_episodes):
            cur_info, hidden = self.dist_info_sym(new_obs_vars[t], hidden, state_info_vars=dict(), is_training=is_training)
            if e == 0:
                info = cur_info
            else:
                info['mean'] = tf.concatenate(axis=1, values=[info['mean'], cur_info['mean']])
                info['log_std'] = tf.concatenate(axis=1, values=[info['log_std'], cur_info['log_std']])
        return info

    @property
    def vectorized(self):
        return True

    def reset(self, dones=None, new_trial=True):
        if dones is None:
            dones = [True]
        dones = np.asarray(dones)
        if self.prev_actions is None or len(dones) != len(self.prev_actions):
            self.prev_actions = np.zeros((len(dones), self.action_space.flat_dim))
            self.prev_reward = np.zeros((len(dones), 1))
            self.prev_done = np.zeros((len(dones), 1))

        self.prev_actions[dones] = 0.
        self.prev_reward[dones] = 0.
        self.prev_done[dones] = 0.
        if new_trial:
            self.prev_hiddens = np.zeros((len(dones), self.hidden_dim))
            self.prev_hiddens[dones] = self.h0.eval()

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation, reward, done):
        actions, agent_infos = self.get_actions([observation], reward, done)
        return actions[0], {k: v[0] for k, v in agent_infos.items()}
        
    @overrides
    def get_action_single_env(self, observation, idx=0, num_tasks=40):
        flat_obs = self.observation_space.flatten_n(observations)
        if self.state_include_action:
            assert self.prev_actions is not None
            all_input = np.concatenate([
                flat_obs,
                self.prev_actions
            ], axis=-1)
            if self.state_include_reward_done:
                all_input = np.concatenate([
                    all_input,
                    self.prev_reward,
                    self.prev_done
                ], axis=-1)
        else:
            all_input = flat_obs
        means, log_stds, hidden_vec = self._cur_f_dist([all_input for _ in range(num_tasks)], self.prev_hiddens)
        means, log_stds = means[idx], log_stds[idx]
        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_stds) + means
        prev_actions = self.prev_actions
        self.prev_actions = self.action_space.flatten_n(actions)
        self.prev_hiddens = hidden_vec
        self.prev_reward = reward
        self.prev_done = done
        agent_info = dict(mean=means, log_std=log_stds)
        if self.state_include_action:
            agent_info["prev_action"] = np.copy(prev_actions)
            if self.state_include_reward_done:
                agent_info["prev_reward"] = np.copy(prev_reward)
                agent_info["prev_done"] = np.copy(prev_done)
        return actions, agent_info

    @overrides
    def get_actions(self, observations, reward, done):
        flat_obs = self.observation_space.flatten_n(observations)
        if self.state_include_action:
            assert self.prev_actions is not None
            all_input = np.concatenate([
                flat_obs,
                self.prev_actions
            ], axis=-1)
            if self.state_include_reward_done:
                all_input = np.concatenate([
                    all_input,
                    self.prev_reward,
                    self.prev_done
                ], axis=-1)
        else:
            all_input = flat_obs
        means, log_stds, hidden_vec = self._cur_f_dist(all_input, self.prev_hiddens)
        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_stds) + means
        prev_actions = self.prev_actions
        self.prev_actions = self.action_space.flatten_n(actions)
        self.prev_hiddens = hidden_vec
        self.prev_reward = reward
        self.prev_done = done
        agent_info = dict(mean=means, log_std=log_stds)
        if self.state_include_action:
            agent_info["prev_action"] = np.copy(prev_actions)
            if self.state_include_reward_done:
                agent_info["prev_reward"] = np.copy(prev_reward)
                agent_info["prev_done"] = np.copy(prev_done)
        return actions, agent_info

    @property
    @overrides
    def recurrent(self):
        return True

    @property
    def distribution(self):
        return self.dist

    @property
    def state_info_specs(self):
        if self.state_include_action:
            return [
                ("prev_action", (self.action_dim,)),
            ]
        else:
            return []

    def log_diagnostics(self, paths):
        log_stds = np.vstack([path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular('AveragePolicyStd', np.mean(np.exp(log_stds)))

    # This makes all of the parameters.
    def create_GRU(self, name, num_units, hidden_nonlinearity,
                    hidden_init_trainable=True, weight_normalization=False):
        all_params = OrderedDict()
        self.num_units = num_units
        self.gru = tf.nn.rnn_cell.GRUCell(num_units=num_units, activation=hidden_nonlinearity)
        with tf.variable_scope(name):
            self.h0, self.hprev = make_gru_layer(num_units, name=name, trainable=hidden_init_trainable,
                                    weight_norm=weight_normalization)
            all_params['h0'] = self.h0
        return all_params
        
    def forward_GRU(self, name, all_params, state=None, horizon=1, input_tensor=None,
                    is_training=True):
        with tf.variable_scope(name, reuse=None) as gru_scope:
            if input_tensor is None:
                input_tensors = []
                l_in = make_input(shape=self.input_shape, input_var=None, name='input_0')
                input_tensors = [l_in] + [make_input(shape=self.input_shape, input_var=None, name='input_%d' % i) for i in range(1, self.num_episodes)]
            else:
                l_in = input_tensor
                input_tensors = [l_in]
            if not is_training:
                horizon = 1
            if len(self.input_shape) == 2:
                l_in = tf.expand_dims(l_in, axis=1)
            n_batches = self.input_shape[0]
            if state is None:
                state = self.h0
            state = tf.tile(
                tf.reshape(state, (1, self.num_units)),
                (n_batches, 1)
            )
            state.set_shape((None, self.num_units))
            gru_output = l_in
            gru_outputs = []
			for t in range(horizon):
				try:
					gru_output, state = self.gru(gru_output[:, t, :], state)
				except ValueError:
					gru_scope.reuse_variables()
					gru_output, state = self.gru(gru_output[:, t, :], state)
				gru_output = tf.nn.relu(gru_output)
				gru_output = tf.expand_dims(gru_output, axis=1)
				gru_outputs.append(gru_output)
			gru_output = tf.reshape(tf.concat(axis=1, values=gru_outputs), [-1, gru_output.get_shape().dims[-1].value])
			gru_output = forward_dense_layer(gru_output, all_params['W'+str(self.n_hidden)], all_params['b'+str(self.n_hidden)],
                                         batch_norm=False, nonlinearity=self.output_nonlinearity,
                                         )
            return input_tensors, gru_output, state
        
    # This makes all of the parameters.
    def create_MLP(self, name, output_dim, hidden_sizes,
                   hidden_W_init=tf_layers.xavier_initializer(dtype=dtypes.float64), hidden_b_init=tf.zeros_initializer(dtype=dtypes.float64),
                   output_W_init=tf_layers.xavier_initializer(dtype=dtypes.float64), output_b_init=tf.zeros_initializer(dtype=dtypes.float64),
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
            all_params['b'+str(len(hidden_sizes))] = b

        return all_params
        
    def get_params_internal(self, all_params=False, **tags):
        if tags.get('trainable', False):
            params = tf.trainable_variables()
        else:
            params = tf.global_variables()

        params = [p for p in params if p.name.startswith('mean_network') or p.name.startswith('output_std_param')]
        params = [p for p in params if 'Adam' not in p.name]

        return params

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

    def log_diagnostics(self, paths, prefix=''):
        log_stds = np.vstack([path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular(prefix+'AveragePolicyStd', np.mean(np.exp(log_stds)))

    #### code largely not used after here except when resuming/loading a policy. ####
    def get_reparam_action_sym(self, obs_var, action_var, old_dist_info_vars):
        """
        Given observations, old actions, and distribution of old actions, return a symbolically reparameterized
        representation of the actions in terms of the policy parameters
        :param obs_var:
        :param action_var:
        :param old_dist_info_vars:
        :return:
        """
        # Not used
        import pdb; pdb.set_trace()
        new_dist_info_vars = self.dist_info_sym(obs_var, action_var)
        new_mean_var, new_log_std_var = new_dist_info_vars["mean"], new_dist_info_vars["log_std"]
        old_mean_var, old_log_std_var = old_dist_info_vars["mean"], old_dist_info_vars["log_std"]
        epsilon_var = (action_var - old_mean_var) / (tf.exp(old_log_std_var) + 1e-8)
        new_action_var = new_mean_var + epsilon_var * tf.exp(new_log_std_var)
        return new_action_var

    def get_param_dtypes(self, all_params=False, **tags):
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_param_dtypes:
            params = self.get_params(all_params, **tags)
            param_values = tf.get_default_session().run(params)
            self._cached_param_dtypes[tag_tuple] = [val.dtype for val in param_values]
        return self._cached_param_dtypes[tag_tuple]

    def get_param_shapes(self, all_params=False, **tags):
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_param_shapes:
            params = self.get_params(all_params, **tags)
            param_values = tf.get_default_session().run(params)
            self._cached_param_shapes[tag_tuple] = [val.shape for val in param_values]
        return self._cached_param_shapes[tag_tuple]

    def set_param_values(self, flattened_params, all_params=False, **tags):
        debug = tags.pop("debug", False)
        param_values = unflatten_tensors(
            flattened_params, self.get_param_shapes(all_params, **tags))
        ops = []
        feed_dict = dict()
        for param, dtype, value in zip(
                self.get_params(all_params, **tags),
                self.get_param_dtypes(all_params, **tags),
                param_values):
            if param not in self._cached_assign_ops:
                assign_placeholder = tf.placeholder(dtype=param.dtype.base_dtype)
                assign_op = tf.assign(param, assign_placeholder)
                self._cached_assign_ops[param] = assign_op
                self._cached_assign_placeholders[param] = assign_placeholder
            ops.append(self._cached_assign_ops[param])
            feed_dict[self._cached_assign_placeholders[param]] = value.astype(dtype)
            if debug:
                print("setting value of %s" % param.name)
        tf.get_default_session().run(ops, feed_dict=feed_dict)

    def flat_to_params(self, flattened_params, all_params=False, **tags):
        return unflatten_tensors(flattened_params, self.get_param_shapes(all_params, **tags))

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        global load_params
        if load_params:
            d["params"] = self.get_param_values(all_params=True)
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        global load_params
        if load_params:
            tf.get_default_session().run(tf.variables_initializer(self.get_params(all_params=True)))
            self.set_param_values(d["params"], all_params=True)
