"""The generator of WGAN."""
import tensorflow as tf
from common.model import ops
from common.model.generator_ops import block


def generator_fully_connected(zs, labels, gf_dim, num_classes, kernel=(3, 3), strides=(2, 2), dilations=(1, 1),
                              pooling='avg', scope_name='Generator', reuse=False):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        act0 = ops.linear(zs, gf_dim, 'g_h0')
        tf.summary.histogram(act0.name, act0)
        act1 = ops.linear(act0, 24 * 24, 'g_h1')
        output = tf.nn.tanh(act1)
        output = tf.reshape(output, shape=[-1, 24, 24, 1])
        tf.summary.histogram(output.name, output)
        return output


def original_generator(zs, labels, gf_dim, num_classes, kernel=(3, 3), strides=(2, 2), dilations=(1, 1),
                       pooling='avg', scope_name='Generator', reuse=False):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        dense1 = tf.layers.dense(inputs=zs,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 bias_initializer=tf.zeros_initializer(),
                                 units=6 * 6 * gf_dim,
                                 activation=tf.nn.relu,
                                 name="dense1")
        reshaped1 = tf.reshape(dense1, shape=[-1, 6, 6, gf_dim], name='reshape1')
        tf.summary.histogram(reshaped1.name, reshaped1)
        up1 = tf.keras.layers.UpSampling2D(size=(2, 2))(reshaped1)
        conv2 = tf.layers.conv2d(inputs=up1,
                                 filters=gf_dim,
                                 kernel_size=[3, 3],
                                 padding="same",
                                 activation=tf.nn.relu,
                                 name="conv2")
        tf.summary.histogram(conv2.name, conv2)
        up2 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv2)
        conv3 = tf.layers.conv2d(inputs=up2,
                                 filters=gf_dim / 2,
                                 kernel_size=[3, 3],
                                 padding="same",
                                 activation=tf.nn.relu,
                                 name="conv3")
        tf.summary.histogram(conv3.name, conv3)
        conv4 = tf.layers.conv2d(inputs=conv3,
                                 filters=1,
                                 kernel_size=[3, 3],
                                 padding="same",
                                 activation=tf.nn.tanh,
                                 name="conv4")
        tf.summary.histogram(conv4.name, conv4)

    return conv4


def generator_resnet(zs, labels, gf_dim, num_classes, kernel=(3, 3), strides=(2, 2), dilations=(1, 1),
                     pooling='avg', scope_name='Generator', reuse=False):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        act0 = ops.linear(zs, gf_dim * 3 * 3 * 8, 'g_h0')
        act0 = tf.reshape(act0, [-1, 3, 3, gf_dim * 8])
        act1 = block(act0, gf_dim * 8, 'g_block1')  # 6 * 6
        act2 = block(act1, gf_dim * 4, 'g_block2')  # 12 * 12
        tf.summary.histogram(act2.name, act2)
        # act3 = block(act2, target_class, gf_dim * 2, 'g_block3')  # 3 * 48
        # act4 = block(act3, target_class, gf_dim * 2, 'g_block4')  # 3 * 96
        act5 = block(act2, gf_dim, 'g_block5')  # 24 * 24
        bn = ops.BatchNorm(name='g_bn')

        act5 = tf.nn.relu(bn(act5))
        act6 = ops.conv2d(act5, 1, 3, 3, 1, 1, name='g_conv_last')
        out = tf.nn.tanh(act6)
        tf.summary.histogram(out.name, out)
        return out
