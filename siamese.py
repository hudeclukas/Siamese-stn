import tensorflow as tf
from tensorflow.contrib import slim as slim

class siamese:
    def __init__(self, margin, image_size=(150,150,1)):
        self.__margin = margin
        self.net_1_input = tf.placeholder(dtype=tf.float32, shape=[None, image_size[0],image_size[1], image_size[2]], name="siamese_1_input")
        self.net_2_input = tf.placeholder(dtype=tf.float32, shape=[None, image_size[0], image_size[1], image_size[2]], name="siamese_2_input")
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=None)

        with tf.variable_scope("siamese") as scope:
            self.network_1 = self.__network(self.net_1_input, 'net_1')
            scope.reuse_variables()
            self.network_2 = self.__network(self.net_2_input, 'net_2')

        self.y = tf.placeholder(tf.float32, [None, ], name="labels")
        self.loss = self.__loss_contrastive()

    def __network(self, input, name:str):
        conv1 = slim.conv2d(
            inputs=input,
            num_outputs=96,
            kernel_size=[5,5],
            reuse=tf.AUTO_REUSE,
            scope=name+'_conv_1'
        )
        pool1 = slim.max_pool2d(
            inputs=conv1,
            kernel_size=[3, 3],
            stride=3,
            scope=name+'_pool_1',
        )
        conv2 = slim.conv2d(
            inputs=pool1,
            num_outputs=256,
            kernel_size=[5,5],
            reuse=tf.AUTO_REUSE,
            scope=name+'_conv_2'
        )
        pool2 = slim.max_pool2d(
            inputs=conv2,
            kernel_size=[3, 3],
            stride=2,
            scope='pool2'
        )
        conv3 = slim.conv2d(
            inputs=pool2,
            num_outputs=384,
            kernel_size=[3, 3],
            stride=1,
            reuse=tf.AUTO_REUSE,
            scope=name+'_conv_3'
        )
        conv4 = slim.conv2d(
            inputs=conv3,
            num_outputs=384,
            kernel_size=[3, 3],
            stride=1,
            reuse=tf.AUTO_REUSE,
            scope=name + '_conv_4'
        )
        conv5 = slim.conv2d(
            inputs=conv4,
            num_outputs=256,
            kernel_size=[3, 3],
            stride=1,
            reuse=tf.AUTO_REUSE,
            scope=name + '_conv_5'
        )
        fc1 = slim.conv2d(
            inputs=conv5,
            num_outputs=1024,
            kernel_size=[1, 1],
            stride=1,
            reuse=tf.AUTO_REUSE,
            scope=name + '_fc_1'
        )
        dropout = slim.dropout(
            fc1,
            scope=name+'_dropout',
            keep_prob=self.dropout_keep_prob,
        )
        fc2 = slim.conv2d(
            inputs=dropout,
            num_outputs=128,
            kernel_size=[1, 1],
            stride=1,
            reuse=tf.AUTO_REUSE,
            scope=name + '_fc_2'
        )
        flatten = slim.flatten(fc2,scope=name+'_flatten')
        return flatten

    def __distanceEuclid(self):
        with tf.variable_scope('Euclid_dw', reuse=True):
            eucd2 = tf.reduce_sum(tf.pow(tf.subtract(self.network_1, self.network_2), 2), 1, name="euclid2")
            eucd = tf.sqrt(eucd2, name="euclid")
            return eucd, eucd2

    def __loss_contrastive(self):
        # one label means similar, zero is value of dissimilarity
        with tf.variable_scope('Loss_Function'):
            y_t = tf.subtract(1.0, tf.convert_to_tensor(self.y, dtype=tf.float32, name="labels"), name="dissimilarity")
            margin = tf.constant(self.__margin, name="margin", dtype=tf.float32)

            dist, dist2 = self.__distanceEuclid()
    
            y_f = tf.subtract(1.0, y_t, name="1-y")
            half_f = tf.multiply(y_f, 0.5, name="y_f/2")
            similar = tf.multiply(half_f, dist2, name="con_l")
            half_t = tf.multiply(y_t, 0.5, name="y_t/2")
            dissimilar = tf.multiply(half_t, tf.maximum(0.0, tf.subtract(margin, dist)))

            losses = tf.add(similar, dissimilar, name="losses")
            loss = tf.reduce_mean(losses, name="loss_reduced")
            # loss=tf.reduce_mean(dist)

            return loss
