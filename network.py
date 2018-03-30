import tensorflow as tf
from tensorflow.contrib import slim as slim

from siamese import siamese
from transformer_layer import spatial_transformer_layer as stl


class siamese_stn:

    def __init__(self, siamese_margin:float, image_size=(150,150,1)):
        self.net_1_input = tf.placeholder(dtype=tf.float32, shape=[None, image_size[0], image_size[1], image_size[2]], name="net_1_input")
        self.net_2_input = tf.placeholder(dtype=tf.float32, shape=[None, image_size[0], image_size[1], image_size[2]], name="net_2_input")

        self.loc_1 = self.localization_network(self.net_1_input, 'net_1')
        self.loc_2 = self.localization_network(self.net_2_input, 'net_2')

        self.stl_1 = stl(self.net_1_input, self.loc_1)
        self.stl_2 = stl(self.net_2_input, self.loc_2)

        self.siamese = siamese(self.stl_1, self.stl_2, siamese_margin, image_size)


    def localization_network(self, input:tf.Tensor, name:str):
        with tf.variable_scope('LOC_'+name):
            conv1 = slim.conv2d(
                inputs=input,
                num_outputs=64,
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
                num_outputs=128,
                kernel_size=[5,5],
                scope=name+'_conv_2'
            )
            pool2 = slim.max_pool2d(
                inputs=conv2,
                kernel_size=[3, 3],
                scope=name + '_pool_2'
            )
            flatten = slim.flatten(pool2, scope=name+'_flatten')
            fc1 = slim.fully_connected(flatten, 3750, scope=name+'_fc_1')
            fc2 = slim.fully_connected(fc1, 2048, scope=name+'_fc_2')
            fc3 = slim.fully_connected(fc2, 6, scope=name+'_fc_3')

            return fc3


