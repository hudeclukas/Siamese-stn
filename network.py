import tensorflow as tf
from tensorflow.contrib import slim as slim
import numpy as np

from siamese import siamese
from transformer_layer import spatial_transformer_layer as stl
from moe_gate import gating_network
import summary_utils as sum_uts


class mixture_of_experts:
    def __init__(self, siamese_margin: float, batch_size:int, image_size=(150,150,1)):
        self.__image_size = image_size
        self.__batch_size = batch_size

        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='main_droput_keep_prob')

        self.net_1_input = tf.placeholder(dtype=tf.float32, shape=[None, image_size[0], image_size[1], image_size[2]],
                                          name="net_1_input")
        self.net_2_input = tf.placeholder(dtype=tf.float32, shape=[None, image_size[0], image_size[1], image_size[2]],
                                          name="net_2_input")
        self.labels = tf.placeholder(tf.float32, [None, ], name="labels")

        # localization layers
        self.loc_stn = localization_stn(self.net_1_input, self.net_2_input, batch_size, image_size, self.dropout_keep_prob)
        # spatial transformer layers
        (self.stl_1, self.stl_2) = stl(self.net_1_input, self.net_2_input, self.loc_stn.loc_net, batch_size)

        # gating network for Mixture of Experts
        self.gating_network = gating_network(self.stl_1, self.stl_2, batch_size, image_size, self.dropout_keep_prob)

        self.siamese_0 = siamese(self.stl_1, self.stl_2, siamese_margin, batch_size, image_size, labels=self.labels, dropout_keep_prob=self.dropout_keep_prob, name='siamese_0')
        self.siamese_1 = siamese(self.stl_1, self.stl_2, siamese_margin, batch_size, image_size, labels=self.labels, dropout_keep_prob=self.dropout_keep_prob, name='siamese_1')
        self.siamese_2 = siamese(self.stl_1, self.stl_2, siamese_margin, batch_size, image_size, labels=self.labels, dropout_keep_prob=self.dropout_keep_prob, name='siamese_2')

        self.loss = self.mixture_of_experts_loss()

    def mixture_of_experts_loss(self):
        with tf.variable_scope('MoE_loss'):
            # simple loss
            # TODO create loss with histogram of gaussians
            losses = tf.add(tf.multiply(self.gating_network.gate[:,0], self.siamese_0.loss),
                          tf.add(tf.multiply(self.gating_network.gate[:,1], self.siamese_1.loss),
                                 tf.multiply(self.gating_network.gate[:,2], self.siamese_2.loss)))
            loss = tf.reduce_mean(losses, name='moe_loss')

            tf.summary.scalar('overall_loss_moe_min', tf.reduce_min(loss))
            tf.summary.scalar('overall_loss_moe_max', tf.reduce_max(loss))
            tf.summary.scalar('overall_loss_moe_mean',loss)
        return loss

class siamese_stn:
    def __init__(self, siamese_margin: float, batch_size: int = 40, image_size=(150, 150, 1)):
        self.net_1_input = tf.placeholder(dtype=tf.float32, shape=[None, image_size[0], image_size[1], image_size[2]],
                                          name='net_1_input')
        self.net_2_input = tf.placeholder(dtype=tf.float32, shape=[None, image_size[0], image_size[1], image_size[2]],
                                          name='net_2_input')

        self.loc_net_stn = localization_stn(self.net_1_input, self.net_2_input, batch_size, image_size)

        (self.stl_1, self.stl_2) = stl(self.net_1_input, self.net_2_input, self.loc_net_stn.loc_net, batch_size)

        self.siamese = siamese(self.stl_1, self.stl_2, siamese_margin, batch_size, image_size)


class localization_stn:

    def __init__(self, net_1_input:tf.Tensor, net_2_input:tf.Tensor, batch_size:int=40, image_size=(150,150,1), dropout_keep_prob=None):
        self.__image_size = image_size
        self.__batch_size = batch_size

        with tf.variable_scope('STN_Feature_extractor') as scope:
            self.feature_extractor_1 = self.feature_extractor(net_1_input)
            scope.reuse_variables()
            self.feature_extractor_2 = self.feature_extractor(net_2_input)
        if dropout_keep_prob is None:
            self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='Locnet_Droput_keep_prob')
        else:
            self.dropout_keep_prob = dropout_keep_prob
        self.loc_net = self.localization_network(self.feature_extractor_1, self.feature_extractor_2, 'Locnet')

    def feature_extractor(self, input:tf.Tensor):
        with tf.name_scope('summary_input_images'):
            sum_uts.image_summary(input, self.__image_size, 0, name='input_image_0')
            sum_uts.image_summary(input, self.__image_size, self.__batch_size - 1, name='input_image_n')
        conv1 = slim.conv2d(
            inputs=input,
            num_outputs=32,
            kernel_size=[5, 5],
            scope='conv_1'
        )
        pool1 = slim.max_pool2d(
            inputs=conv1,
            kernel_size=[2, 2],
            scope='pool_1'
        )
        conv2 = slim.conv2d(
            inputs=pool1,
            num_outputs=64,
            kernel_size=[5, 5],
            scope='conv_2'
        )
        pool2 = slim.max_pool2d(
            inputs=conv2,
            kernel_size=[3, 3],
            scope='pool_2'
        )
        with tf.name_scope('sum_output_images'):
            sum_uts.conv_image_summary(pool2, (self.__image_size[0] >> 2, self.__image_size[1] >> 2, 64), 0,
                                       'locnet_pool_2_out_0')
            sum_uts.conv_image_summary(pool2, (self.__image_size[0] >> 2, self.__image_size[1] >> 2, 64),
                                       self.__batch_size - 1, 'locnet_pool_2_out_n')

        flatten = slim.flatten(pool2, scope='flatten')
        return flatten

    def localization_network(self, input_1:tf.Tensor, input_2:tf.Tensor, name:str):
        with tf.variable_scope(name):

            flatten_merged = tf.concat([input_1, input_2], 1)

            fc_1 = slim.fully_connected(
                inputs=flatten_merged,
                num_outputs=256,
                scope=name+ '_fc_1'
            )
            fc_2 = slim.fully_connected(
                inputs=fc_1,
                num_outputs=256,
                scope=name+'_fc_2'
            )
            dropout = slim.dropout(
                inputs=fc_2,
                keep_prob=self.dropout_keep_prob
            )
            # identity transform
            # initial = np.array([[1., 0, 0], [0, 1., 0],[1., 0, 0], [0, 1., 0]])
            # initial = initial.astype('float32').flatten()

            # deleter = np.array([[1., 1., 0], [1., 1., 0],[1., 1., 0], [1., 1., 0]])
            # deleter = deleter.astype('float32').flatten()
            # localisation network
            with tf.variable_scope(name+'_fc_3', reuse=True):
                W_fc3 = tf.Variable(tf.zeros([dropout.shape[1].value, 2]), dtype=tf.float32, name='W_fc3')
                b_fc3 = tf.Variable(initial_value=[0,0], dtype=tf.float32, name='b_fc3')
                h_fc3 = tf.matmul(dropout, W_fc3) + b_fc3
                # tf_deleter = tf.constant(deleter, dtype=tf.float32, name='theta_deleter')
                # h_fc3 = tf.multiply(h_fc3, tf_deleter)

            return h_fc3

class Similarity():
    def __init__(self):
        with tf.name_scope('Similarity_metrics'):
            self.vec1 = tf.placeholder(tf.float32, name="placeholder_vec1")
            self.vec2 = tf.placeholder(tf.float32, name="placeholder_vec2")
            self.euclid = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(self.vec1, self.vec2), 2), 1), name="euclid_test")
            self.canberra = tf.divide(tf.reduce_sum(tf.divide(tf.abs(tf.subtract(self.vec1, self.vec2)),
                                                    tf.add(tf.add(tf.abs(self.vec1), tf.abs(self.vec2)),
                                                           tf.constant(0.0001, dtype=tf.float32))), axis=1
                                                    ), tf.constant(1000, tf.float32),
                                      name='canberra_test')