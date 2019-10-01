"""The discriminator of WGAN."""
import tensorflow as tf
from common.model import ops
from common.model.ops import block, leaky_relu


def discriminator_fully_connected(x, labels, df_dim, number_classes, kernel=(3, 3), strides=(2, 2), dilations=(1, 1),
                                  pooling='avg', update_collection=None, act=tf.nn.relu, scope_name='Discriminator',
                                  reuse=False):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, df_dim, name="dense_1")
        x = leaky_relu(x)
        tf.summary.histogram(x.name, x)
        output = tf.layers.dense(x, 1, name="dense_2")
        tf.summary.histogram(output.name, output)
        return output


def original_discriminator(x, labels, df_dim, number_classes, kernel=(3, 3), strides=(2, 2), dilations=(1, 1),
                           pooling='avg', update_collection=None, act=tf.nn.relu, scope_name='Discriminator',
                           reuse=False):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        conv1 = tf.layers.conv2d(
            inputs=x,
            filters=df_dim / 4,
            kernel_size=[3, 3],
            strides=(2, 2),
            padding="same",
            activation=leaky_relu,
            name="dconv1")
        tf.summary.histogram(conv1.name, conv1)
        # Convolutional Layer #2
        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=df_dim / 2,
            kernel_size=[3, 3],
            strides=(2, 2),
            padding="same",
            activation=leaky_relu,
            name="dconv2")
        tf.summary.histogram(conv2.name, conv2)
        conv3 = tf.layers.conv2d(
            inputs=conv2,
            filters=df_dim,
            kernel_size=[3, 3],
            strides=(2, 2),
            padding="same",
            activation=leaky_relu,
            name="dconv3")
        tf.summary.histogram(conv3.name, conv3)
        flat = tf.layers.flatten(conv3, name="dflat")
        output = tf.layers.dense(inputs=flat,
                                 activation=None,
                                 units=1,
                                 name="doutput")
        output = tf.reshape(output, [-1])
        tf.summary.histogram(output.name, output)
        return output


def discriminator_resnet(x, labels, df_dim, number_classes, kernel=(3, 3), strides=(2, 2), dilations=(1, 1),
                         pooling='avg', update_collection=None, act=tf.nn.relu, scope_name='Discriminator',
                         reuse=False):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        h0 = block(x, df_dim, 'd_optimized_block1', act=act)  # 12 * 12
        h1 = block(h0, df_dim * 2, 'd_block2', act=act)  # 6 * 6
        h2 = block(h1, df_dim * 4, 'd_block3', act=act)  # 3 * 3
        tf.summary.histogram(h2.name, h2)
        # h3 = block(h2, df_dim * 4, 'd_block4', update_collection, act=act)  # 8 * 8 # 3*12
        # h4 = block(h3, df_dim * 8, 'd_block5', update_collection, act=act)  # 3*6
        h5 = block(h2, df_dim * 8, 'd_block6', False, act=act)
        h5_act = act(h5)
        tf.summary.histogram(h5_act.name, h5_act)
        h6 = tf.reduce_sum(h5_act, [1, 2])
        output = ops.linear(h6, 1, scope='d_linear')
        tf.summary.histogram(output.name, output)
        return output
