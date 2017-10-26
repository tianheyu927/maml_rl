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

TESTING_ITRS = [1,3,5,7,9,11,13,15,17,19] + [8*x-1 for x in range(1000)]

PLOT_ITRS = [0,2,4,6,8,10] + [100*x for x in range(80)] + [1,3,5,7,9,11,13,] + [8*x-1 for x in range(500)]

VIDEO_ITRS = [1,3] + [200*x-1 for x in range(40)]