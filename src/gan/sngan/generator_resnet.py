"""The generator of SNGAN."""

import tensorflow as tf
from common.model import ops
from common.model.generator_ops import get_kernel, get_dimentions_factors, sn_block
from common.model.ops import attention, leaky_relu
from gan.sngan.generator import Generator


class ResnetGenerator(Generator):
    def __init__(self, config, shape, num_classes=None, scope_name=None):
        super(ResnetGenerator, self).__init__(config, shape, num_classes, scope_name)
        self.strides = self.get_strides()
        self.number_of_layers = len(self.strides)
        self.starting_dim = self.dim * (2 ** self.number_of_layers)
        self.get_initial_shape(config)
        self.final_bn = ops.BatchNorm(name='g_bn')

    def get_strides(self):
        strides = [(1, 2), (1, 2), (1, 2), (1,2)]
        if self.length == 512:
            strides.extend([(1, 2), (1, 2)])
        return strides

    def network(self, z, labels, reuse):
        height_d, width_d = get_dimentions_factors(self.strides)
        number_of_layers = len(self.strides)
        hidden_dim = self.dim * (2 ** (number_of_layers-1))
        c_h = int(self.height / height_d)
        c_w = int((self.length / width_d))
        h = ops.snlinear(z, c_h * c_w * hidden_dim, name='noise_linear')
        h = tf.reshape(h, [-1, c_h, c_w, hidden_dim])
        print("COMPRESSED TO: ", h.shape)

        with tf.variable_scope("up", reuse=reuse):
            for layer_id in range(number_of_layers):
                print(h.shape)
                block_name = 'up_block{}'.format(number_of_layers - (layer_id + 1))
                dilation_rate = (1,1)
                h = sn_block(h, hidden_dim, block_name, get_kernel(h, self.kernel), self.strides[layer_id], dilation_rate,
                             self.act, self.pooling, 'VALID')
                tf.summary.histogram(block_name, h, family=self.scope_name)
                if layer_id == number_of_layers - 2:
                    h = attention(h, hidden_dim, sn=True, reuse=reuse)
                    tf.summary.histogram("up_attention", h, family=self.scope_name)
                hidden_dim = hidden_dim / self.strides[layer_id][1]

        bn = ops.BatchNorm(name='g_bn')
        h_act = leaky_relu(bn(h), name="h_act")
        if self.output_shape[2] == 1:
            out = tf.nn.tanh(ops.snconv2d(h_act, 1, (self.output_shape[0], 1), name='last_conv'), name="generated")
        else:
            out = tf.nn.tanh(ops.snconv2d(h_act, 21, (1, 1), name='last_conv'), name="generated")
        tf.summary.histogram("Generated_results", out, family=self.scope_name)
        print("GENERATED SHAPE", out.shape)
        return out
