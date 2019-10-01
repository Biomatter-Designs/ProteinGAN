"""The WGAN Model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from gan.gan_base_model import GAN
from gan.wgan.discriminator import discriminator_fully_connected, original_discriminator, discriminator_resnet
from gan.wgan.generator import generator_fully_connected, original_generator, generator_resnet

tfgan = tf.contrib.gan


class WGAN(GAN):
    """WGAN model."""

    def get_loss(self, real_x, fake_x, gen_sparse_class, discriminator_real, discriminator_fake):
        d_loss_real = tf.reduce_mean(discriminator_real)
        d_loss_fake = tf.reduce_mean(discriminator_fake)
        # Gradient Penalty
        epsilon = tf.random_uniform(shape=[self.config.batch_size, 1, 1, 1], minval=0., maxval=1.)
        x_hat = real_x + epsilon * (fake_x - real_x)
        discriminator_fn = self.get_discriminator_and_generator()
        d_x_hat = discriminator_fn(x_hat, gen_sparse_class, self.df_dim, self.num_classes,
                                   update_collection="NO_OPS", reuse=True)
        grad_d_x_hat = tf.gradients(d_x_hat, [x_hat])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_d_x_hat), reduction_indices=[1, 2, 3]))
        penalty = tf.reduce_mean((slopes - 1.) ** 2)
        tf.summary.scalar("gradient-penalty", penalty)
        d_loss = d_loss_fake - d_loss_real + 10 * tf.reduce_mean(penalty)
        g_loss = -d_loss_fake
        return d_loss_real, d_loss_fake, d_loss, g_loss

    def get_discriminator_and_generator(self):
        if self.config.architecture == 'fully_connected':
            generator_fn = generator_fully_connected
            discriminator_fn = discriminator_fully_connected
        elif self.config.architecture == 'original':
            discriminator_fn = original_discriminator
            generator_fn = original_generator
        elif self.config.architecture == 'resnet':
            discriminator_fn = discriminator_resnet
            generator_fn = generator_resnet
        else:
            raise NotImplementedError
        return discriminator_fn, generator_fn
