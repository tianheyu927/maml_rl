import numpy as np
import scipy.optimize
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc import logger
from sandbox.rocky.tf.misc import tensor_utils




class QuadDistExpertOptimizer(Serializable):
    """
    Runs Tensorflow optimization on a quadratic loss function, under the kl constraint

    """

    def __init__(
            self,
            name,
            max_opt_itr=20,
            initial_penalty=1.0,
            min_penalty=1e-2,
            max_penalty=1e6,
            increase_penalty_factor=2,
            decrease_penalty_factor=0.5,
            max_penalty_itr=10,
            adapt_penalty=True,
            adam_steps=5,
    ):
        Serializable.quick_init(self, locals())
        self._name = name
        self._max_opt_itr = max_opt_itr
        self._penalty = initial_penalty
        self._initial_penalty = initial_penalty
        self._min_penalty = min_penalty
        self._max_penalty = max_penalty
        self._increase_penalty_factor = increase_penalty_factor
        self._decrease_penalty_factor = decrease_penalty_factor
        self._max_penalty_itr = max_penalty_itr
        self._adapt_penalty = adapt_penalty

        self._opt_fun = None
        self._target = None
        self._max_constraint_val = None
        self._constraint_name = None
        self._adam_steps = adam_steps
        self._correction_term = 0


    def update_opt(self, loss, target, leq_constraint, inputs, constraint_name="constraint", *args, **kwargs):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should implement methods of the
        :class:`rllab.core.paramerized.Parameterized` class.
        :param leq_constraint: A constraint provided as a tuple (f, epsilon), of the form f(*inputs) <= epsilon.
        :param inputs: A list of symbolic variables as inputs
        :return: No return value.
        """
        constraint_term, constraint_value = leq_constraint
        with tf.variable_scope(self._name):
            penalty_var = tf.placeholder(tf.float32, tuple(), name="penalty")
        penalized_loss = loss + penalty_var * constraint_term

        self._target = target
        self._max_constraint_val = constraint_value
        self._constraint_name = constraint_name

        self._inputs = inputs
        self._loss = loss
        self._adam = tf.train.AdamOptimizer()
        self._train_step = self._adam.minimize(self._loss)

        # initialize Adam variables
        uninit_vars = []
        sess = tf.get_default_session()
        if sess is None:
            sess = tf.Session()
        for var in tf.global_variables():
            # note - this is hacky, may be better way to do this in newer TF.
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninit_vars.append(var)
        sess.run(tf.variables_initializer(uninit_vars))




        def get_opt_output():
            params = target.get_params(trainable=True)
            grads = tf.gradients(penalized_loss, params)
            for idx, (grad, param) in enumerate(zip(grads, params)):
                if grad is None:
                    grads[idx] = tf.zeros_like(param)
            flat_grad = tensor_utils.flatten_tensor_variables(grads)
            return [
                tf.cast(penalized_loss, tf.float64),
                tf.cast(flat_grad, tf.float64),
            ]

        self._opt_fun = ext.lazydict(
            f_loss=lambda: tensor_utils.compile_function(inputs, loss, log_name="f_loss"),
            f_constraint=lambda: tensor_utils.compile_function(inputs, constraint_term, log_name="f_constraint"),
            f_penalized_loss=lambda: tensor_utils.compile_function(
                inputs=inputs + [penalty_var],
                outputs=[penalized_loss, loss, constraint_term],
                log_name="f_penalized_loss",
            ),
            f_opt=lambda: tensor_utils.compile_function(
                inputs=inputs + [penalty_var],
                outputs=get_opt_output(),
            )
        )

    def loss(self, inputs):
        return self._opt_fun["f_loss"](*inputs)

    def constraint_val(self, inputs):
        return self._opt_fun["f_constraint"](*inputs)

    def optimize(self, input_vals_list, correction_term=None):
        if correction_term is None:
            correction_term = self.correction_term
        sess = tf.get_default_session()
        for _ in range(self._adam_steps):
            # sess.run(self._train_step, feed_dict=dict(list(zip(self._inputs, input_vals_list))))
            grad = self._adam.compute_gradients(self._loss)
            grad = grad + correction_term
            # self._adam.apply_gradients(grad)
            sess.run(self._adam.apply_gradients(grad), feed_dict=dict(list(zip(self._inputs, input_vals_list))))


