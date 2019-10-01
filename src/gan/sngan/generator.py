import tensorflow as tf
from gan.sngan.base_model import Model
from common.model.generator_ops import get_dimentions_factors, get_kernel, sn_block
from common.model.ops import leaky_relu, log, attention
from tensorflow.python.training import training_util


class Generator(Model):
    def __init__(self, config, shape, num_classes, scope_name):
        if scope_name is None:
            scope_name = "Generator"
        super(Generator, self).__init__(config, num_classes, scope_name)
        self.dim = config.gf_dim
        self.act = leaky_relu
        self.output_shape = shape
        self.length = self.output_shape[2]
        self.height = self.output_shape[1]

    def generate(self, z, labels, reuse=False):
        with tf.variable_scope(self.scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            out = self.network(z, labels, reuse)
            self.validate_shape(out, self.output_shape)
            self.log("output {}".format(out.shape))
            tf.summary.histogram("Generated_results", out, family=self.scope_name)
            return out

    def network(self, z, labels, reuse):
        pass

    def get_initial_shape(self, config):
        height_d, width_d = get_dimentions_factors(self.strides)
        self.initial_shape = [config.batch_size, int(self.height / height_d), int(self.length / width_d),
                              int(self.starting_dim / 2)]

    def get_temperature(self, add_to_tensorboard=False):
        global_step = tf.cast(training_util._get_or_create_global_step_read(), tf.float32)
        e = tf.constant(0.000001)
        temperature = tf.maximum(.01, tf.constant(1.0) / (log(global_step, 3.0) + e))
        if add_to_tensorboard:
            with tf.control_dependencies([tf.print("Temperature", temperature, summarize=-1)]):
                tf.summary.histogram("Temperature", temperature, family=self.scope_name)
        return temperature

    def get_block_params(self, hidden_dim, layer_id):
        stride = self.strides[layer_id]
        block_name = 'g_block{}'.format(self.number_of_layers - layer_id)
        dilation_rate = (1, self.dilations ** max(0, (self.number_of_layers - (layer_id + 3))))
        hidden_dim = hidden_dim / stride[1]
        return block_name, dilation_rate, hidden_dim, stride

    def add_sn_block(self, x, out_dim, block_name, dilation_rate, strides):
        kernel = get_kernel(x, self.kernel)
        h = sn_block(x, out_dim, block_name, kernel, strides, dilation_rate, self.act, self.pooling, 'VALID')
        tf.summary.histogram(block_name, h, family=self.scope_name)
        return h
