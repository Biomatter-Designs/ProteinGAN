import numpy as np
import tensorflow as tf
from gan.protein.helpers import ACID_EMBEDDINGS
from common.model.ops import attention, spectral_normed_weight

SEP = "#######################################"


class Model(object):
    def __init__(self, config, num_classes, scope_name):
        self.scope_name = scope_name
        print("scope_name", self.scope_name)
        self.num_classes = num_classes,
        self.kernel = (config.kernel_height, config.kernel_width)
        self.dilations = config.dilation_rate
        self.pooling = config.pooling

    def validate_shape(self, actual, expected):
        shape = actual.get_shape().as_list()
        assert shape == expected, "Output shape is {}, but should be {}".format(shape, expected)

    def log(self, msg, level=tf.logging.INFO):
        tf.logging.log(level, "{}: {}".format(self.scope_name, msg))

    def add_attention(self, h, hidden_dim, reuse):
        h = attention(h, hidden_dim, sn=True, reuse=reuse)
        tf.summary.histogram("attention", h, family=self.scope_name)
        return h

    def get_embeddings(self, shape, path=None, name=ACID_EMBEDDINGS):
        if path is None:
            initializer = tf.random_normal_initializer()
        else:
            initializer = tf.constant_initializer(np.load(path))
        embedding = tf.get_variable(name, shape=shape, initializer=initializer, trainable=True)
        embedding_normalized = spectral_normed_weight(tf.transpose(embedding))
        embedding = tf.transpose(embedding_normalized)
        return embedding


    def embedding_lookup(self, data, embedding):
        x_flattened = tf.reshape(data, [-1, embedding.get_shape().as_list()[0]])
        x_embedded = tf.matmul(x_flattened, embedding)
        x_embedded = tf.reshape(x_embedded, [-1, self.length, self.dim])
        x_embedded = tf.expand_dims(x_embedded, axis=1)
        tf.summary.histogram("x_embedded", x_embedded, family=self.scope_name)
        h = x_embedded
        return h