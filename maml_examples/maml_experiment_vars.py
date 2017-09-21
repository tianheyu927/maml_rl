import tensorflow as tf

HIDDEN_NONLINEARITY = {
    '':tf.nn.relu,
    'relu':tf.nn.relu,
    'th':tf.nn.tanh,
    'thh':tf.nn.tanh,
    'reluh':tf.nn.relu,
}
OUTPUT_NONLINEARITY = {
    '':tf.identity,
    'relu':tf.identity,
    'th':tf.identity,
    'thh':tf.nn.tanh,
    'reluh':tf.nn.tanh,
}