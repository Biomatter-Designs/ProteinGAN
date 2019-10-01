"""The SNGAN Model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from gan.gan_base_model import GAN

from gan.sngan.discriminator_resnet import ResnetDiscriminator
from gan.sngan.discriminator_gumbel import GumbelDiscriminator
from gan.sngan.generator_resnet import ResnetGenerator
from gan.sngan.generator_gumbel import GumbelGenerator

import tensorflow as tf

tfgan = tf.contrib.gan


class SNGAN(GAN):
    """SNGAN model."""

    def __init__(self, data_handler, noise):
        super(SNGAN, self).__init__(data_handler, noise)

    def minibatch_std(self, x, function=tf.reduce_mean):
        y = x
        y -= tf.reduce_mean(y, axis=0, keepdims=True)  # Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)  # Calc variance over group.
        y = tf.sqrt(y + 1e-8)  # Calc stddev over group.
        y = function(y)  # Take average of everything
        return y

    def calculate_difference(self, x, function=tf.reduce_mean):
        splitted = tf.split(x, 2)
        diff = tf.abs(splitted[0] - splitted[1])
        return function(diff)

    def get_loss(self, real_x, fake_x, gen_sparse_class, discriminator_real, discriminator_fake, r_h, f_h):
        if self.config.loss_type == 'hinge_loss_std':
            d_loss_fake, d_loss_real, g_loss_gan = self.hinge_loss(discriminator_fake, discriminator_real)
            variation_loss = self.get_variation_loss(fake_x, real_x)
            g_loss_gan = g_loss_gan + variation_loss
            print('hinge loss (+ std mean) is using')
        elif self.config.loss_type == 'hinge_loss_std_sum':
            d_loss_fake, d_loss_real, g_loss_gan = self.hinge_loss(discriminator_fake, discriminator_real)
            variation_loss = self.get_variation_loss(fake_x, real_x, tf.reduce_sum)
            g_loss_gan = g_loss_gan + variation_loss
            print('hinge loss ( + std sum ) is using')
        elif self.config.loss_type == 'hinge_loss':
            d_loss_fake, d_loss_real, g_loss_gan = self.hinge_loss(discriminator_fake, discriminator_real)
            print('hinge loss is using')
        elif self.config.loss_type == 'hinge_loss_ra':
            d_loss_fake, d_loss_real, g_loss_gan = self.hinge_ra_loss(discriminator_fake, discriminator_real)
            print('Relativistic hinge (std) loss is using')
        elif self.config.loss_type == 'hinge_loss_ra_std':
            d_loss_fake, d_loss_real, g_loss_gan = self.hinge_ra_loss(discriminator_fake, discriminator_real)
            variation_loss = self.get_variation_loss(fake_x, real_x)
            g_loss_gan = g_loss_gan + variation_loss
            print('Relativistic hinge loss is using')
        elif self.config.loss_type == 'hinge_loss_ra_std_sum':
            d_loss_fake, d_loss_real, g_loss_gan = self.hinge_ra_loss(discriminator_fake, discriminator_real)
            variation_loss = self.get_variation_loss(fake_x, real_x, tf.reduce_sum)
            g_loss_gan = g_loss_gan + variation_loss
            print('Relativistic hinge loss is using')
        elif self.config.loss_type == 'hinge_loss_diff':
            d_loss_fake, d_loss_real, g_loss_gan = self.hinge_loss(discriminator_fake, discriminator_real)
            variation_loss = self.get_variation_diff_loss(fake_x, real_x, tf.reduce_mean)
            g_loss_gan = g_loss_gan + self.config.variation_level * variation_loss
            print('Relativistic hinge loss is using')
        elif self.config.loss_type == 'kl_loss':
            d_loss_real = tf.nn.softplus(-discriminator_real)
            d_loss_fake = tf.nn.softplus(discriminator_fake)
            g_loss_gan = -discriminator_fake
            print('kl loss is using')
        elif self.config.loss_type == 'wasserstein':
            d_loss_real = discriminator_real + 0.01 * tf.reduce_mean(tf.square(discriminator_real))
            d_loss_fake = discriminator_fake + 0.01 * tf.reduce_mean(tf.square(discriminator_fake))
            g_loss_gan = -discriminator_fake
            print('wasserstein loss is using')
        elif self.config.loss_type == 'wgan-gp':
            print('WGAN_GP loss')
            d_loss_real = tf.nn.softplus(-(discriminator_real - discriminator_fake))
            gp = self.gradient_penalty(real_x, fake_x)
            drift = 0.001 * tf.reduce_mean(tf.square(discriminator_real))
            d_loss_fake = gp * 10.0  # '+ drift
            with tf.control_dependencies([tf.print("GP:", gp, "Drift", drift)]):
                tf.summary.scalar("gradient penalty", gp)
                tf.summary.scalar("Drift loss", drift)
            g_loss_gan = -discriminator_fake
        elif self.config.loss_type == 'non_saturating':
            print('Non saturating loss')
            d_loss_real = tf.nn.softplus(discriminator_fake)
            d_loss_fake = tf.nn.softplus(-discriminator_real)
            with tf.name_scope('R1Penalty'):
                real_loss = tf.reduce_sum(discriminator_real)
                real_grads = (tf.gradients(real_loss, [real_x])[0])  # real_x
                r1_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3])
                with tf.control_dependencies([tf.print("r1_penalty", tf.reduce_mean(r1_penalty))]):
                    tf.summary.scalar("r1_penalty", tf.reduce_mean(r1_penalty))
            d_loss_real += r1_penalty * (10 * 0.5)
            g_loss_gan = tf.nn.softplus(-discriminator_fake)
        elif self.config.loss_type == 'ipot':
            l = self.IPOT_distance(tf.squeeze(f_h), tf.squeeze(r_h), discriminator_fake.get_shape().as_list()[0])
            d_loss_real, d_loss_fake = -l, 0
            g_loss_gan = l
        else:
            raise NotImplementedError
        d_loss = tf.reduce_mean(tf.add(d_loss_real, d_loss_fake), name="d_loss")
        g_loss = tf.reduce_mean(g_loss_gan, name="g_loss")
        return d_loss_real, d_loss_fake, d_loss, g_loss

    def hinge_ra_loss(self, discriminator_fake, discriminator_real):
        # Reference: https://github.com/taki0112/RelativisticGAN-Tensorflow/blob/master/ops.py
        fake_logit = (discriminator_fake - tf.reduce_mean(discriminator_real))
        real_logit = (discriminator_real - tf.reduce_mean(discriminator_fake))
        d_loss_real = tf.nn.relu(1.0 - real_logit)
        d_loss_fake = tf.nn.relu(1.0 + fake_logit)
        g_loss_fake = tf.nn.relu(1.0 - fake_logit)
        g_loss_real = tf.nn.relu(1.0 + real_logit)
        g_loss_gan = g_loss_fake + g_loss_real
        return d_loss_fake, d_loss_real, g_loss_gan

    def get_variation_diff_loss(self, fake_x, real_x, function=tf.reduce_mean):
        real_variance = self.calculate_difference(real_x, function=function)
        fake_variance = self.calculate_difference(fake_x, function=function)
        minibatch_variance = real_variance - fake_variance
        variation_loss = tf.maximum(0.0, minibatch_variance)
        return variation_loss

    def get_variation_loss(self, fake_x, real_x, function=tf.reduce_mean):
        real_variance = self.minibatch_std(real_x, function=function)
        fake_variance = self.minibatch_std(fake_x, function=function)
        minibatch_variance = real_variance - fake_variance
        variation_loss = tf.maximum(0.0, minibatch_variance)
        return variation_loss

    def hinge_loss(self, discriminator_fake, discriminator_real):
        d_loss_real = tf.nn.relu(1.0 - discriminator_real)
        d_loss_fake = tf.nn.relu(1.0 + discriminator_fake)
        g_loss_gan = -discriminator_fake

        return d_loss_fake, d_loss_real, g_loss_gan

    def gradient_penalty(self, real, fake):
        def interpolate(a, b):
            shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
            alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.get_shape().as_list())
            return inter

        x = interpolate(real, fake)
        pred = self.get_discriminator_result(x, labels=None, reuse=True)
        gradients = tf.gradients(pred, x)[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=list(range(1, x.shape.ndims))))
        gp = tf.reduce_mean((slopes - 1.) ** 2)
        return gp

    def cost_matrix(self, x, y):
        "Returns the matrix of $|x_i-y_j|^p$."
        "Returns the cosine distance"
        x = tf.nn.l2_normalize(x, 1, epsilon=1e-12)
        y = tf.nn.l2_normalize(y, 1, epsilon=1e-12, )
        tmp1 = tf.matmul(x, y, transpose_b=True)
        cos_dis = 1 - tmp1
        return cos_dis

    def IPOT(self, x, y, n, beta=1):
        # pdb.set_trace()
        sigma = tf.scalar_mul(1. / n, tf.ones([n, 1]))
        T = tf.ones([n, n])
        C = self.cost_matrix(x, y)
        A = tf.exp(-C / beta)
        for t in range(50):
            Q = A * T
            for k in range(1):
                delta = 1. / (n * tf.matmul(Q, sigma))
                sigma = 1. / (n * tf.matmul(Q, delta))
            # pdb.set_trace()
            tmp = tf.matmul(tf.diag(tf.squeeze(delta)), Q)
            T = tf.matmul(tmp, tf.diag(tf.squeeze(sigma)))
        return T, C

    def IPOT_distance(self, x, y, batch_size):
        T, C = self.IPOT(tf.reshape(x, [batch_size, -1]), tf.reshape(y, [batch_size, -1]), batch_size)
        distance = tf.trace(tf.matmul(C, T, transpose_a=True))

        return distance

    def get_discriminator_and_generator(self):

        if self.config.architecture == 'resnet':
            discriminator_fn = ResnetDiscriminator(self.config, self.data_handler.shape, self.data_handler.num_classes)
            generator_fn = ResnetGenerator(self.config, self.data_handler.shape, self.data_handler.num_classes)
        elif self.config.architecture == 'gumbel':
            discriminator_fn = GumbelDiscriminator(self.config, self.data_handler.shape, self.data_handler.num_classes)
            generator_fn = GumbelGenerator(self.config, self.data_handler.shape, self.data_handler.num_classes)
        else:
            raise NotImplementedError

        return discriminator_fn, generator_fn
