"""The discriminator of SNGAN."""
import tensorflow as tf
from common.model import ops
from common.model.generator_ops import get_kernel
from common.model.ops import sn_block, attention
from gan.sngan.discriminator import Discriminator


class ResnetDiscriminator(Discriminator):
    def __init__(self, config, shape, num_classes=None, scope_name=None):
        super(ResnetDiscriminator, self).__init__(config, shape, num_classes, scope_name)
        self.act = tf.nn.relu
        self.strides = [(1, 2), (1, 2), (1, 2), (1, 2)]
        if self.length == 512:
            self.strides.extend([(1, 2), (1, 2)])

    def network(self, data, labels, reuse):
        tf.summary.histogram("Input", data, family=self.scope_name)
        h = data
        hidden_dim = self.dim
        for layer in range(len(self.strides)):
            print(h.shape)
            if layer == 1:
                h = attention(h, hidden_dim, sn=True, reuse=reuse)
                tf.summary.histogram("attention", h, family=self.scope_name)
            block_name = 'd_block{}'.format(layer)
            hidden_dim = hidden_dim * self.strides[layer][0]
            # dilation_rate = dilations[0] ** max(1, layer-2), dilations[1] ** max(1, layer-2)
            dilation_rate = (1, 1)
            h = sn_block(h, hidden_dim, block_name, get_kernel(h, self.kernel), self.strides[layer],
                         dilation_rate, None, self.act, self.pooling, padding='VALID')
            tf.summary.histogram(block_name, h, family=self.scope_name)

        end_block = self.act(h, name="after_resnet_block")
        tf.summary.histogram("after_resnet_block", end_block, family=self.scope_name)

        h_std = ops.minibatch_stddev_layer(end_block)
        tf.summary.histogram("minibatch_stddev_layer", h_std, family=self.scope_name)
        # h_std_conv_std = act(ops.snconv2d(h_std, hidden_dim, (1, 3), update_collection=update_collection,
        #                                  name='minibatch_stddev_stride', padding=None, strides=(1, 3)),
        #                     name="minibatch_stddev_stride_act")
        # tf.summary.histogram("after_mini_batch_std", h_std_conv_std, family=scope_name)
        # h_final_flattened = tf.layers.flatten(h_std_conv_std)
        h_final_flattened = tf.reduce_sum(h_std, [1, 2])
        tf.summary.histogram("h_final_flattened", h_final_flattened, family=self.scope_name)
        output = ops.snlinear(h_final_flattened, 1, name='d_sn_linear')
        tf.summary.histogram("final_output", output, family=self.scope_name)
        return output, h_final_flattened



