import tensorflow as tf
from gan.sngan.base_model import Model
from common.model.generator_ops import get_kernel
from common.model.ops import leaky_relu, sn_block


class Discriminator(Model):
    def __init__(self, config, shape, num_classes, scope_name):
        if scope_name is None:
            scope_name = "Discriminator"
        super(Discriminator, self).__init__(config, num_classes, scope_name)
        self.dim = config.df_dim
        self.act = leaky_relu
        self.input_shape = shape
        self.length = self.input_shape[2]
        self.height = self.input_shape[1]
        self.output_shape = [config.batch_size, 1]

    def discriminate(self, data, labels, reuse=True):
        tf.logging.info(self.scope_name)
        self.validate_shape(data, self.input_shape)
        tf.summary.histogram("Discriminator_input", data, family=self.scope_name)
        with tf.variable_scope(self.scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            out = self.network(data, labels, reuse)
            self.validate_shape(out[0], self.output_shape)
            tf.summary.histogram("Discriminator_results", out[0], family=self.scope_name)
            return out

    def network(self, data, labels, reuse):
        pass

    def get_block_params(self, hidden_dim, layer):
        block_name = 'd_block{}'.format(layer)
        dilation_rate = (1, self.dilations ** max(0, layer - 2))
        strides = self.strides[layer]
        hidden_dim = hidden_dim * strides[1]
        return block_name, dilation_rate, hidden_dim, strides

    def add_sn_block(self, h, hidden_dim, block_name, dilation_rate, strides):
        kernel = get_kernel(h, self.kernel)
        h = sn_block(h, hidden_dim, block_name, kernel, strides, dilation_rate, None, self.act, self.pooling,
                     padding='VALID')
        tf.summary.histogram(block_name, h, family=self.scope_name)
        return h