import tensorflow as tf
from absl import flags
from gan.documentation import add_gan_scalars, get_gan_vars
from tensorflow.python import math_ops
from tensorflow.python.training.session_run_hook import SessionRunHook


class GAN(object):
    def __init__(self, data_handler, noise):
        self.config = self.init_param()
        self.data_handler = data_handler
        self.dataset = self.config.dataset
        self.z_dim = self.config.z_dim
        self.gf_dim = self.config.gf_dim
        self.df_dim = self.config.df_dim
        self.global_step = tf.train.create_global_step()
        self.noise = noise
        self.discriminator, self.generator = self.get_discriminator_and_generator()
        self.build_model()

    def init_param(self):
        return flags.FLAGS

    def build_model(self):
        """Builds a model."""
        config = self.config

        self.d_learning_rate, self.g_learning_rate = self.get_learning_rates()

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            self.build_model_single_gpu(batch_size=config.batch_size)
            self.d_optim, self.g_optim, self.d_learning_rate, self.g_learning_rate = self.get_optimizers()
            # Add summaries.
            with tf.variable_scope('tensorboard'):
                if self.config.running_mode == "train":
                    add_gan_scalars(self.d_learning_rate, self.g_learning_rate, self.d_loss, self.d_loss_fake,
                                    self.d_loss_real, self.g_loss, self.discriminator_real, self.discriminator_fake)
                self.add_trainable_parameters_to_tensorboard("Discriminator")
                self.add_trainable_parameters_to_tensorboard("Generator")

                self.add_gradients_to_tensorboard("Discriminator")
                self.add_gradients_to_tensorboard("Generator")

    def add_gradients_to_tensorboard(self, scope):
        [tf.summary.histogram(self.get_summary_name(x), x, family="Gradients_{}".format(scope))
         for x in tf.global_variables()
         if x not in tf.trainable_variables() and scope in x.name and ("g_opt" in x.name or "d_opt" in x.name)]

    def add_trainable_parameters_to_tensorboard(self, scope):
        [tf.summary.histogram(self.get_summary_name(x), x, family="Weights_{}".format(scope))
         for x in tf.trainable_variables() if scope in x.name and "beta" not in x.name and "gamma" not in x.name]

    def get_summary_name(self, x):
        return x.name.replace("model/", "").replace(":", "_")

    def build_model_single_gpu(self, batch_size=1):
        config = self.config
        show_num = min(config.batch_size, 16)

        self.increment_global_step = self.global_step.assign_add(1)
        # with tf.variable_scope('input'):
        batch = self.data_handler.get_batch(batch_size, config)
        real_x, labels = batch[0], batch[1]
        real_x, labels = self.data_handler.prepare_real_data(real_x, labels)
        self.fake_x = self.get_generated_data(self.noise, labels)
        # real_x_mixed, fake_x_mixed, labels_mixed = self.random_shuffle(real_x, fake_x, labels, self.config.batch_size,
        #                                                                self.config.label_noise_level)
        self.discriminator_real, r_h = self.get_discriminator_result(real_x, labels)
        self.discriminator_fake, f_h = self.get_discriminator_result(self.fake_x, labels, reuse=True)
        self.discriminator_fake = tf.identity(self.discriminator_fake, name="d_score")
        # with tf.variable_scope('display'):
        self.data_handler.display_real_data(real_x if config.already_embedded else batch[0], labels, show_num,
                                            self.discriminator_real)
        self.data_handler.display_fake_data(self.fake_x, labels, show_num, self.discriminator_fake,
                                            self.config.running_mode == "train")

        # with tf.variable_scope('loss'):
        self.d_loss_real, self.d_loss_fake, self.d_loss, self.g_loss = self.get_loss(real_x, self.fake_x, labels,
                                                                                     self.discriminator_real,
                                                                                     self.discriminator_fake, r_h,
                                                                                     f_h)
        with tf.variable_scope('stddev'):
            add_std_var_to_tensorboard([real_x, self.fake_x], ["real", "fake"])

        _, self.d_vars, self.g_vars = get_gan_vars()

    def get_optimizers(self):
        with tf.variable_scope('optimizers'):
            d_optimizer = tf.train.AdamOptimizer(learning_rate=self.d_learning_rate, name='d_opt',
                                                 beta1=self.config.beta1,
                                                 beta2=self.config.beta2)
            d_optim = d_optimizer.minimize(self.d_loss, var_list=self.d_vars)
            g_optimizer = tf.train.AdamOptimizer(learning_rate=self.g_learning_rate, name='g_opt',
                                                 beta1=self.config.beta1,
                                                 beta2=self.config.beta2)
            g_optim = g_optimizer.minimize(self.g_loss, var_list=self.g_vars)
            beta1_power, beta2_power = d_optimizer._get_beta_accumulators()
            d_lr = (d_optimizer._lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

            beta1_power, beta2_power = g_optimizer._get_beta_accumulators()
            g_lr = (g_optimizer._lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

            return d_optim, g_optim, d_lr, g_lr

    def get_learning_rates(self):
        with tf.variable_scope('learning_rate'):
            # current_step = tf.cast(self.global_step, tf.float32)
            # g_ratio = (1.0 + 2e-5 * tf.maximum((current_step - 100000.0), 0.0))
            # g_ratio = tf.minimum(g_ratio, 4.0)

            return self.config.discriminator_learning_rate, self.config.generator_learning_rate

    def get_loss(self, discriminator_fake, discriminator_real, fake_x, gen_sparse_class, real_x, r_h, f_h):
        pass

    def get_discriminator_result(self, data, labels, reuse=False):
        return self.discriminator.discriminate(data, labels, reuse=reuse)

    def get_generated_data(self, data, labels):
        return self.generator.generate(data, labels)


    def get_discriminator_and_generator(self):
        pass

    def random_shuffle(self, real, fake, labels, batch_size, label_noise_level):
        if label_noise_level > 0:
            perecentage_of_noise_data = tf.truncated_normal([1], mean=label_noise_level,
                                                            stddev=label_noise_level / 2.0)
            current_step = tf.cast(self.global_step, tf.float32)
            decay_factor = (2e-6 * tf.maximum((500000.0 - current_step), 0.0))
            num_of_not_swapped_examples = batch_size * (1 - perecentage_of_noise_data * decay_factor)
            idx = tf.random_shuffle(tf.range(int(batch_size)))
            num_real = tf.cast(tf.squeeze(num_of_not_swapped_examples), tf.int32)
            real_idx = tf.gather(idx, tf.range(num_real))
            fake_idx = tf.gather(idx, tf.range(num_real, int(batch_size)))

            real_ = tf.gather(real, real_idx)
            fake_ = tf.gather(fake, fake_idx)
            real_mix = tf.concat([real_, fake_], axis=0)

            fake_ = tf.gather(fake, real_idx)
            real_ = tf.gather(real, fake_idx)
            fake_mix = tf.concat([fake_, real_], axis=0)

            labels1 = tf.gather(labels, real_idx)
            labels2 = tf.gather(labels, fake_idx)
            labels_mix = tf.concat([labels1, labels2], axis=0)
            real_mix.set_shape([batch_size, *real_mix.get_shape().as_list()[1:]])
            fake_mix.set_shape([batch_size, *fake_mix.get_shape().as_list()[1:]])
            labels_mix.set_shape([batch_size, *labels_mix.get_shape().as_list()[1:]])
            return real_mix, fake_mix, labels_mix
        else:
            return real, fake, labels


def add_std_var_to_tensorboard(data, name):
    for i, val in enumerate(data):
        std = tf.reduce_mean(tf.keras.backend.std(val, axis=0), name=name[i])
        tf.summary.scalar(name[i], std, family="Stddev")


class VariableRestorer(SessionRunHook):
    """Hook that counts steps per second."""

    def __init__(self, model_dir, scope):
        self.model_dir = model_dir
        self.scope = scope
        self.variables_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    def after_create_session(self, session, coord):  # pylint: disable=unused-argument

        session.graph._unsafe_unfinalize()
        variable_names = [self.get_variable_name(var, self.scope) for var in self.variables_to_restore]
        print("Restoring weights {} from model {}".format(variable_names, self.model_dir))
        to_restore = dict(zip(variable_names, self.variables_to_restore))
        saver_restore = tf.train.Saver(to_restore)
        saver_restore.restore(session, self.model_dir)
        session.graph.finalize()

    def get_variable_name(self, var, scope):
        return var.name.replace(scope, "").replace(":0", "")
