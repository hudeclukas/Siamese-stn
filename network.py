import tensorflow as tf
from tensorflow.contrib import slim as slim
import numpy as np

from siamese import siamese
from transformer_layer import spatial_transformer_layer as stl
import summary_utils as sum_uts


class siamese_stn:

    def __init__(self, siamese_margin:float, batch_size:int, image_size=(150,150,1)):
        self.__image_size = image_size
        self.__batch_size = batch_size

        self.net_1_input = tf.placeholder(dtype=tf.float32, shape=[None, image_size[0], image_size[1], image_size[2]], name="net_1_input")
        self.net_2_input = tf.placeholder(dtype=tf.float32, shape=[None, image_size[0], image_size[1], image_size[2]], name="net_2_input")

        self.loc_1 = self.localization_network(self.net_1_input, 'net_1')
        self.loc_2 = self.localization_network(self.net_2_input, 'net_2')

        self.stl_1 = stl(self.net_1_input, self.loc_1, batch_size)
        self.stl_2 = stl(self.net_2_input, self.loc_2, batch_size)

        self.siamese = siamese(self.stl_1, self.stl_2, siamese_margin, batch_size, image_size)


    def localization_network(self, input:tf.Tensor, name:str):
        with tf.variable_scope('LOC_'+name):
            sum_uts.image_summary(input, self.__image_size, 0, name='input_image_0')
            sum_uts.image_summary(input, self.__image_size, self.__batch_size-1, name='input_image_n')
            with tf.name_scope(name+'_feature_extractor'):
                conv1 = slim.conv2d(
                    inputs=input,
                    num_outputs=32,
                    kernel_size=[5,5],
                    scope=name+'_conv_1'
                )
                pool1 = slim.max_pool2d(
                    inputs=conv1,
                    kernel_size=[2,2],
                    scope=name+'_pool_1'
                )
                conv2 = slim.conv2d(
                    inputs=pool1,
                    num_outputs=64,
                    kernel_size=[5,5],
                    scope=name+'_conv_2'
                )
                pool2 = slim.max_pool2d(
                    inputs=conv2,
                    kernel_size=[3, 3],
                    scope=name + '_pool_2'
                )
            sum_uts.conv_image_summary(pool2, (self.__image_size[0] >> 2, self.__image_size[1] >> 2, 64), 0, 'locnet_pool_2_out_0')
            sum_uts.conv_image_summary(pool2, (self.__image_size[0] >> 2, self.__image_size[1] >> 2, 64), self.__batch_size-1, 'locnet_pool_2_out_n')

            flatten = slim.flatten(pool2, scope=name+'_flatten')
            fc_1 = slim.fully_connected(
                inputs=flatten,
                num_outputs=256,
                scope=name+ '_fc_1'
            )
            fc_2 = slim.fully_connected(
                inputs=fc_1,
                num_outputs=256,
                scope=name+'_fc_2'
            )
            # identity transform
            # initial = np.array([[1., 0, 0], [0, 1., 0]])
            # initial = initial.astype('float32').flatten()
            # localisation network
            with tf.variable_scope(name+'_fc_3', reuse=True):
                W_fc3 = tf.Variable(tf.zeros([fc_2.shape[1].value, 2]), name='W_fc3')
                b_fc3 = tf.Variable(initial_value=[0], name='b_fc3')
                h_fc3 = tf.matmul(fc_2, W_fc3) + b_fc3

            return h_fc3

class Similarity():
    def __init__(self):
        self.vec1 = tf.placeholder(tf.float32, name="placeholder_vec1")
        self.vec2 = tf.placeholder(tf.float32, name="placeholder_vec2")
        self.euclid = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(self.vec1, self.vec2), 2), 1), name="euclid_test")
        self.canberra = tf.divide(tf.reduce_sum(tf.divide(tf.abs(tf.subtract(self.vec1, self.vec2)),
                                                tf.add(tf.add(tf.abs(self.vec1), tf.abs(self.vec2)),
                                                       tf.constant(0.0001, dtype=tf.float32))), axis=1
                                                ), tf.constant(1000, tf.float32),
                                  name='canberra_test')