import os

import common.model.utils_ori as utils
import tensorflow as tf
from gan.documentation import add_image_grid


class Image(object):
    def __init__(self, flags, properties, logdir):
        self.config = flags
        self.properties = properties
        self.num_classes = properties["num_of_classes"]
        self.shape = [properties["image_height"], properties["image_width"], properties["num_channels"]]
        self.width = properties["image_width"]

    def prepare_real_data(self, real_x, labels):
        if (self.shape[0] != self.config.input_h) or (self.shape[1] != self.config.input_w):
            real_x = tf.reshape(real_x, shape=[-1, self.config.input_h, self.config.input_w, 1])
        return real_x, labels

    def display_real_data(self, real_x, labels, show_num, d_scores=None):
        images = tf.cast((real_x + 1.) * 127.5, tf.uint8)
        add_image_grid("real_image_grid", show_num, images, self.shape, utils.squarest_grid_size(show_num))

    def display_fake_data(self, fake_x, gen_sparse_class, show_num, d_scores=None, to_display=True):
        fake_for_display = fake_x
        if (self.shape[0] != self.config.input_h) or (self.shape[1] != self.config.input_w):
            fake_for_display = tf.reshape(fake_for_display, shape=[-1, self.shape[0], self.shape[1], 1])
        fake_images = tf.cast((fake_for_display + 1.) * 127.5, tf.uint8)
        tf.summary.image("generator", fake_images, max_outputs=2, family="individual")
        tf.summary.text("generator_0", tf.py_func(lambda val: str(val), [gen_sparse_class[0]], tf.string))
        tf.summary.text("generator_1", tf.py_func(lambda val: str(val), [gen_sparse_class[1]], tf.string))
        if to_display:
            add_image_grid("fake_image_grid", show_num, fake_images, self.shape,
                           utils.squarest_grid_size(show_num))

    def get_batch(self, batch_size, config):
        return utils.get_batches(utils.extract_image_and_label, os.path.join(config.data_dir, config.dataset),
                                 batch_size, cycle_length=10, shuffle_buffer_size=config.shuffle_buffer_size,
                                 running_mode=config.running_mode, args=[self.shape])
