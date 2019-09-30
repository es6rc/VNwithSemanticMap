import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v2


def fr_v2(x, output_neurons, inside_neurons, is_training, name='fr',
          wt_decay=0.0001, stride=1, updates_collections=tf.GraphKeys.UPDATE_OPS):
    """Performs fusion of information between the map and the reward map.
  Inputs
    x:   NxHxWxC1

  Outputs
    fr map:     NxHxWx(output_neurons)
  """
    if type(stride) != list:
        stride = [stride]
    with slim.arg_scope(resnet_v2.resnet_utils.resnet_arg_scope(weight_decay=wt_decay)):
        with slim.arg_scope([slim.batch_norm], updates_collections=updates_collections) as arg_sc:
            # Change the updates_collections for the conv normalizer_params to None
            for i in range(len(arg_sc.keys())):
                if 'convolution' in arg_sc.keys()[i]:
                    arg_sc.values()[i]['normalizer_params']['updates_collections'] = updates_collections
            with slim.arg_scope(arg_sc):
                bottleneck = resnet_v2.bottleneck
                blocks = []
                for i, s in enumerate(stride):
                    b = resnet_v2.resnet_utils.Block(
                        'block{:d}'.format(i + 1), bottleneck, [{
                            'depth': output_neurons,
                            'depth_bottleneck': inside_neurons,
                            'stride': stride[i]
                        }])
                    blocks.append(b)
                x, outs = resnet_v2.resnet_v2(x, blocks, num_classes=None, is_training=is_training,
                                              global_pool=False, output_stride=None, include_root_block=False,
                                              reuse=False, scope=name)
    return x, outs


def value_iteration_network(
        fr, num_iters, val_neurons, action_neurons, kernel_size, share_wts=False,
        name='vin', wt_decay=0.0001, activation_fn=None, shape_aware=False):
    """
  Constructs a Value Iteration Network, convolutions and max pooling across
  channels.
  Input:
    fr:             NxWxHxC
    val_neurons:    Number of channels for maintaining the value.
    action_neurons: Computes action_neurons * val_neurons at each iteration to
                    max pool over.
  Output:
    value image:  NxHxWx(val_neurons)
  """
    init_var = np.sqrt(2.0 / (kernel_size ** 2) / (val_neurons * action_neurons))
    vals = []
    with tf.variable_scope(name) as varscope:
        if not shape_aware:
            fr_shape = tf.unstack(tf.shape(fr))
            val_shape = tf.stack(fr_shape[:-1] + [val_neurons])
            val = tf.zeros(val_shape, name='val_init')
        else:
            val = tf.expand_dims(tf.zeros_like(fr[:, :, :, 0]), dim=-1) * \
                  tf.constant(0., dtype=tf.float32, shape=[1, 1, 1, val_neurons])
            val_shape = tf.shape(val)
        vals.append(val)
        for i in range(num_iters):
            if share_wts:
                # The first Value Iteration maybe special, so it can have its own
                # parameters.
                scope = 'conv'
                if i == 0: scope = 'conv_0'
                if i > 1: varscope.reuse_variables()
            else:
                scope = 'conv_{:d}'.format(i)
            val = slim.conv2d(tf.concat([val, fr], 3, name='concat_{:d}'.format(i)),
                              num_outputs=action_neurons * val_neurons,
                              kernel_size=kernel_size, stride=1, activation_fn=activation_fn,
                              scope=scope, normalizer_fn=None,
                              weights_regularizer=slim.l2_regularizer(wt_decay),
                              weights_initializer=tf.random_normal_initializer(stddev=init_var),
                              biases_initializer=tf.zeros_initializer())
            val = tf.reshape(val, [-1, action_neurons * val_neurons, 1, 1],
                             name='re_{:d}'.format(i))
            val = slim.max_pool2d(val, kernel_size=[action_neurons, 1],
                                  stride=[action_neurons, 1], padding='VALID',
                                  scope='val_{:d}'.format(i))
            val = tf.reshape(val, val_shape, name='unre_{:d}'.format(i))
            vals.append(val)
    return val, vals


def fc_network(x, neurons, wt_decay, name, num_pred=None, offset=0,
               batch_norm_param=None, dropout_ratio=0.0, is_training=None):
    if dropout_ratio > 0:
        assert (is_training is not None), \
            'is_training needs to be defined when trainnig with dropout.'

    repr = []
    for i, neuron in enumerate(neurons):
        init_var = np.sqrt(2.0 / neuron)
        if batch_norm_param is not None:
            x = slim.fully_connected(x, neuron, activation_fn=None,
                                     weights_initializer=tf.random_normal_initializer(stddev=init_var),
                                     weights_regularizer=slim.l2_regularizer(wt_decay),
                                     normalizer_fn=slim.batch_norm,
                                     normalizer_params=batch_norm_param,
                                     biases_initializer=tf.zeros_initializer(),
                                     scope='{:s}_{:d}'.format(name, offset + i))
        else:
            x = slim.fully_connected(x, neuron, activation_fn=tf.nn.relu,
                                     weights_initializer=tf.random_normal_initializer(stddev=init_var),
                                     weights_regularizer=slim.l2_regularizer(wt_decay),
                                     biases_initializer=tf.zeros_initializer(),
                                     scope='{:s}_{:d}'.format(name, offset + i))
        if dropout_ratio > 0:
            x = slim.dropout(x, keep_prob=1 - dropout_ratio, is_training=is_training,
                             scope='{:s}_{:d}'.format('dropout_' + name, offset + i))
        repr.append(x)

    if num_pred is not None:
        init_var = np.sqrt(2.0 / num_pred)
        x = slim.fully_connected(x, num_pred,
                                 weights_regularizer=slim.l2_regularizer(wt_decay),
                                 weights_initializer=tf.random_normal_initializer(stddev=init_var),
                                 biases_initializer=tf.zeros_initializer(),
                                 activation_fn=None,
                                 scope='{:s}_pred'.format(name))
    return x, repr
