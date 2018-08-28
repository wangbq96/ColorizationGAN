import time
import datetime
import logging
import os
import numpy as np
from glob import glob
from ops import *
from image_tools import save_images
from image_tools import load_images
from tools import remain_time

# input_c_dim:  Dimension of input image color.
# output_c_dim: Dimension of output image color.
INPUT_C_DIM = 1
OUTPUT_C_DIM = 3

# gf_dim: Dimension of gen filters in first conv layer.
# df_dim: Dimension of discrim filters in first conv layer.
GF_DIM = 64
DF_DIM = 64


class DataProvider(object):
    def __init__(self, config):
        self.data = glob('{}/{}/train/*.png'.format(config.dataset_dir, config.dataset_name))
        self.len = len(self.data)
        if self.len == 0:
            print(" [!] Data not found")
        else:
            print("data len: {}".format(self.len))

        self.batch_size = config.batch_size
        self.total_batch_num = self.len // self.batch_size
        self.batch_idx = 0
        self.epoch_idx = 0

    def load_data(self):
        batch_files = self.data[self.batch_idx*self.batch_size:(self.batch_idx+1)*self.batch_size]
        batch = [load_images(batch_file) for batch_file in batch_files]
        self.batch_idx += 1
        if self.batch_idx >= self.total_batch_num:
            np.random.shuffle(self.data)
            self.batch_idx = 0
            self.epoch_idx += 1
        return np.array(batch).astype(np.float32)


class ColorizationGAN(object):
    def __init__(self, sess, config):
        self.sess = sess
        self.batch_size = config.batch_size
        self.input_size = config.img_size
        self.output_size = config.img_size

        self.L1_lambda = config.L1_lambda
        self.clip_value = config.clip_value
        self.lr = config.lr
        self.beta1 = config.beta1

        self.dataset_name = config.dataset_name
        self.model_name = config.model_name

        self.dataset_dir = config.dataset_dir
        self.checkpoint_dir = config.checkpoint_dir
        self.sample_dir = config.sample_dir

        self.data_provider = DataProvider(config)
        self.epoch = config.epoch

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')

        self._init_logger()
        self._build_model()

    def _init_logger(self):
        # log
        log_format = '%(asctime)s %(name)-4s %(levelname)-8s %(message)s'
        time_format = '%m-%d %H:%M'
        now_date = datetime.datetime.now().strftime('%Y-%m-%d')
        logging.basicConfig(level=logging.INFO,
                            format=log_format,
                            datefmt=time_format,
                            filename='./log/%s-%s.log' % (self.model_name, now_date),
                            filemode='w')

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        formatter = logging.Formatter(log_format, time_format)
        console.setFormatter(formatter)

        self.root_logger = logging.getLogger('')
        self.root_logger.addHandler(console)
        self.log_template = "Epoch: [{:4d}/{:4d}] Batch: [{:4d}/{:4d}] Time: {:4.2f}s " \
                            "d_loss: {:.8f} g_loss: {:.8f} time_left: {:4.2f}min"

    def _build_model(self):
        real_data_dim = [self.batch_size, self.input_size, self.input_size, INPUT_C_DIM+OUTPUT_C_DIM]
        self.real_data = tf.placeholder(tf.float32, real_data_dim, name='real_A_and_B_images')

        self.real_A = self.real_data[:, :, :, :INPUT_C_DIM]
        self.real_B = self.real_data[:, :, :, INPUT_C_DIM: INPUT_C_DIM+OUTPUT_C_DIM]

        self.fake_B = self._generator(self.real_A)

        self.real_AB = tf.concat([self.real_A, self.real_B], 3)
        self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)
        self.D, self.D_logits = self._discriminator(self.real_AB, reuse=False)
        self.D_, self.D_logits_ = self._discriminator(self.fake_AB, reuse=True)

        self.fake_B_sample = self._sampler(self.real_A)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.fake_B_sum = tf.summary.image("fake_B", tf.image.yuv_to_rgb(self.fake_B))
        '''
        ===============================================================================================================
        '''
        self.l1_loss = tf.reduce_mean(tf.abs(self.real_B - self.fake_B))
        if self.model_name == "color-cgan":
            self.d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D))
            )
            self.d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_))
            )
            self.g_loss_without_l1 = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))
            )
            self.g_loss = self.g_loss_without_l1 + self.L1_lambda * self.l1_loss

            self.d_loss = self.d_loss_real + self.d_loss_fake

        elif self.model_name == "color-wgan":
            self.d_loss_real = -tf.reduce_mean(self.D_logits)
            self.d_loss_fake = tf.reduce_mean(self.D_logits_)
            self.g_loss_without_l1 = -tf.reduce_mean(self.D_logits_)
            self.g_loss = self.g_loss_without_l1 + self.L1_lambda * self.l1_loss
            self.d_loss = self.d_loss_real + self.d_loss_fake

        '''
        ===============================================================================================================
        '''
        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        self.l1_loss_sum = tf.summary.scalar("l1_loss", self.l1_loss)
        self.g_loss_without_l1_sum = tf.summary.scalar("g_loss_without_l1", self.g_loss_without_l1)

        self.g_sum = tf.summary.merge(
            [self.d__sum, self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum,
             self.l1_loss_sum, self.g_loss_without_l1_sum]
        )
        self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter(self.checkpoint_dir, self.sess.graph)

    def train(self):
        if self.model_name == "color-cgan":
            d_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(self.d_loss, var_list=self.d_vars)
            g_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(self.g_loss, var_list=self.g_vars)
            clip_d = None
        elif self.model_name == "color-wgan":
            d_optim = tf.train.RMSPropOptimizer(self.lr, decay=0.9).minimize(self.d_loss, var_list=self.d_vars)
            g_optim = tf.train.RMSPropOptimizer(self.lr, decay=0.9).minimize(self.g_loss, var_list=self.g_vars)
            clip_d = [var.assign(tf.clip_by_value(var, -self.clip_value, self.clip_value)) for var in self.d_vars]

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        counter = 1
        start_time = time.time()

        while self.data_provider.epoch_idx < self.epoch:
            if self.model_name == "color-cgan":
                batch_images = self.data_provider.load_data()
                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={self.real_data: batch_images})

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={self.real_data: batch_images})
                self.writer.add_summary(summary_str, counter)
                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={self.real_data: batch_images})
                self.writer.add_summary(summary_str, counter)

            elif self.model_name == "color-wgan":
                for _ in range(5):
                    # Update D network
                    batch_images = self.data_provider.load_data()
                    _, _d_loss, _g_loss, summary_str = self.sess.run(
                        [d_optim, self.d_loss, self.g_loss, self.d_sum], feed_dict={self.real_data: batch_images})
                    self.writer.add_summary(summary_str, counter)
                    self.sess.run([clip_d], feed_dict={})

                # Update G network
                batch_images = self.data_provider.load_data()
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={self.real_data: batch_images})
                self.writer.add_summary(summary_str, counter)

            err_d_fake = self.d_loss_fake.eval({self.real_data: batch_images})
            err_d_real = self.d_loss_real.eval({self.real_data: batch_images})
            err_g = self.g_loss.eval({self.real_data: batch_images})

            time_left = remain_time(
                epoch_now=self.data_provider.epoch_idx,
                epoch_total=self.epoch,
                batch_now=self.data_provider.batch_idx,
                batch_total=self.data_provider.total_batch_num,
                pass_time=time.time()-start_time
            )
            self.root_logger.info(
                self.log_template.format(
                    self.data_provider.epoch_idx,
                    self.epoch,
                    self.data_provider.batch_idx,
                    self.data_provider.total_batch_num,
                    time.time()-start_time,
                    err_d_fake+err_d_real,
                    err_g,
                    time_left/60
                )
            )

            counter += 1

            if np.mod(counter, 100) == 1:
                self._sample(self.data_provider.epoch_idx, self.data_provider.batch_idx)

            if np.mod(counter, 500) == 2:
                self.save(self.checkpoint_dir, counter)

        self._sample(self.data_provider.epoch_idx, self.data_provider.batch_idx)
        self.save(self.checkpoint_dir, counter)

    def _sample(self, epoch, idx):
        data = np.random.choice(glob('{}/{}/train/*.png'.format(self.dataset_dir, self.dataset_name)), self.batch_size)
        sample_input = [load_images(sample_file) for sample_file in data]
        sample_input = np.array(sample_input).astype(np.float32)

        sample_fake_images, d_loss, g_loss = self.sess.run(
            [self.fake_B_sample, self.d_loss, self.g_loss],
            feed_dict={self.real_data: sample_input}
        )
        sample_real_images = sample_input[:, :, :, INPUT_C_DIM: INPUT_C_DIM+OUTPUT_C_DIM]
        save_images(sample_real_images, './{}/train_{:02d}_{:04d}'.format(self.sample_dir, epoch, idx), is_real=True)
        save_images(sample_fake_images, './{}/train_{:02d}_{:04d}'.format(self.sample_dir, epoch, idx), is_real=False)
        self.root_logger.info("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

    def _discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            # image is 256 x 256 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            # df_dim = 64
            h0 = lrelu(conv2d(image, DF_DIM, name='d_h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(self.d_bn1(conv2d(h0, DF_DIM*2, name='d_h1_conv')))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(self.d_bn2(conv2d(h1, DF_DIM*4, name='d_h2_conv')))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(self.d_bn3(conv2d(h2, DF_DIM*8, d_h=1, d_w=1, name='d_h3_conv')))
            # h3 is (16 x 16 x self.df_dim*8)
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

    def _generator(self, image, y=None):
        with tf.variable_scope("generator") as scope:
            # U-net
            s = self.output_size
            s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
            s32, s64, s128 = int(s/32), int(s/64), int(s/128)

            # image is (256 x 256 x input_c_dim)
            e1 = conv2d(image, GF_DIM, name='g_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), GF_DIM*2, name='g_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), GF_DIM*4, name='g_e3_conv'))
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), GF_DIM*8, name='g_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), GF_DIM*8, name='g_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), GF_DIM*8, name='g_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), GF_DIM*8, name='g_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)
            e8 = self.g_bn_e8(conv2d(lrelu(e7), GF_DIM*8, name='g_e8_conv'))
            # e8 is (1 x 1 x self.gf_dim*8)

            self.d1, self.d1_w, self.d1_b = deconv2d(
                tf.nn.relu(e8), [self.batch_size, s128, s128, GF_DIM*8], name='g_d1', with_w=True
            )
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(
                tf.nn.relu(d1), [self.batch_size, s64, s64, GF_DIM*8], name='g_d2', with_w=True
            )
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(
                tf.nn.relu(d2), [self.batch_size, s32, s32, GF_DIM*8], name='g_d3', with_w=True
            )
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(
                tf.nn.relu(d3), [self.batch_size, s16, s16, GF_DIM*8], name='g_d4', with_w=True
            )
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(
                tf.nn.relu(d4), [self.batch_size, s8, s8, GF_DIM*4], name='g_d5', with_w=True
            )
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(
                tf.nn.relu(d5), [self.batch_size, s4, s4, GF_DIM*2], name='g_d6', with_w=True
            )
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(
                tf.nn.relu(d6), [self.batch_size, s2, s2, GF_DIM], name='g_d7', with_w=True
            )
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)

            self.d8, self.d8_w, self.d8_b = deconv2d(
                tf.nn.relu(d7), [self.batch_size, s, s, OUTPUT_C_DIM], name='g_d8', with_w=True
            )
            # d8 is (256 x 256 x output_c_dim)

            return tf.nn.tanh(self.d8)

    def _sampler(self, image, y=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            # U-net
            s = self.output_size
            s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
            s32, s64, s128 = int(s/32), int(s/64), int(s/128)

            # image is (256 x 256 x input_c_dim)
            e1 = conv2d(image, GF_DIM, name='g_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), GF_DIM*2, name='g_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), GF_DIM*4, name='g_e3_conv'))
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), GF_DIM*8, name='g_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), GF_DIM*8, name='g_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), GF_DIM*8, name='g_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), GF_DIM*8, name='g_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)
            e8 = self.g_bn_e8(conv2d(lrelu(e7), GF_DIM*8, name='g_e8_conv'))
            # e8 is (1 x 1 x self.gf_dim*8)

            self.d1, self.d1_w, self.d1_b = deconv2d(
                tf.nn.relu(e8), [self.batch_size, s128, s128, GF_DIM*8], name='g_d1', with_w=True
            )
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(
                tf.nn.relu(d1), [self.batch_size, s64, s64, GF_DIM*8], name='g_d2', with_w=True
            )
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(
                tf.nn.relu(d2), [self.batch_size, s32, s32, GF_DIM*8], name='g_d3', with_w=True
            )
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(
                tf.nn.relu(d3), [self.batch_size, s16, s16, GF_DIM*8], name='g_d4', with_w=True
            )
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(
                tf.nn.relu(d4), [self.batch_size, s8, s8, GF_DIM*4], name='g_d5', with_w=True
            )
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(
                tf.nn.relu(d5), [self.batch_size, s4, s4, GF_DIM*2], name='g_d6', with_w=True
            )
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(
                tf.nn.relu(d6), [self.batch_size, s2, s2, GF_DIM], name='g_d7', with_w=True
            )
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)

            self.d8, self.d8_w, self.d8_b = deconv2d(
                tf.nn.relu(d7), [self.batch_size, s, s, OUTPUT_C_DIM], name='g_d8', with_w=True
            )
            # d8 is (256 x 256 x output_c_dim)

            return tf.nn.tanh(self.d8)

    def save(self, checkpoint_dir, step):
        model_dir = "%s_%s_%s" % (self.model_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s_%s" % (self.model_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self, args):

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        data = glob('{}/{}/test/*.png'.format(self.dataset_dir, self.dataset_name))
        for idx in range(len(data) // args.batch_size):
            batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
            batch = [load_images(batch_file) for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)

            sample_fake_images, d_loss, g_loss = self.sess.run(
                [self.fake_B_sample, self.d_loss, self.g_loss],
                feed_dict={self.real_data: batch_images}
            )
            sample_real_images = batch_images[:, :, :, INPUT_C_DIM: INPUT_C_DIM+OUTPUT_C_DIM]
            save_images(sample_real_images, './test/{}'.format(idx), is_real=True)
            save_images(sample_fake_images, './test/{}'.format(idx), is_real=False)

