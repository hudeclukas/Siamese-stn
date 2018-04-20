import tensorflow as tf
from tensorflow.contrib import slim

import summary_utils as sum_uts

class gating_network:
    def __init__(self, net_1_input: tf.Tensor, net_2_input: tf.Tensor, batch_size: int = 40, image_size=(150, 150, 1), dropout_keep_prob=None):
        self.__image_size = image_size
        self.__batch_size = batch_size

        with tf.variable_scope('GATE_Feature_extractor') as scope:
            self.feature_extractor_1 = self.feature_extractor(net_1_input)
            scope.reuse_variables()
            self.feature_extractor_2 = self.feature_extractor(net_2_input)

        if dropout_keep_prob is None:
            self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='gating_droput_keep_prob')
        else:
            self.dropout_keep_prob = dropout_keep_prob
        self.gate = self.gate_guard(self.feature_extractor_1, self.feature_extractor_2, 'gate_guard')


    def gate_guard(self, input_1:tf.Tensor, input_2:tf.Tensor, name:str) -> tf.Tensor:
        with tf.variable_scope(name):
            flatten_merged = tf.concat([input_1, input_2], 1)
            fc_1 = slim.fully_connected(
                inputs=flatten_merged,
                num_outputs=256,
                scope=name + '_fc_1'
            )
            fc_2 = slim.fully_connected(
                inputs=fc_1,
                num_outputs=512,
                scope=name + '_fc_2'
            )
            fc_3 = slim.fully_connected(
                inputs=fc_2,
                num_outputs=512,
                scope=name + '_fc_3'
            )
            fc_4 = slim.fully_connected(
                inputs=fc_3,
                num_outputs=256,
                scope=name + '_fc_4'
            )
            dropout = slim.dropout(
                inputs=fc_4,
                keep_prob=self.dropout_keep_prob
            )
            logits = slim.fully_connected(
                inputs=dropout,
                num_outputs=3,
                scope=name + '_logits'
            )
            argmax = tf.argmax(logits, axis=1, name=name + '_expert')
            tf.summary.text(name+'gate_guard_expert',tf.as_string(argmax))
            softmax = slim.softmax(logits, name + '_softmax')
            tf.summary.text(name + 'gate_guard_softmax', tf.as_string(softmax))

            return softmax


    def feature_extractor(self, input: tf.Tensor):
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
        conv3 = slim.conv2d(
            inputs=pool2,
            num_outputs=128,
            kernel_size=[5,5],
            scope='conv_3'
        )
        conv4 = slim.conv2d(
            inputs=conv3,
            num_outputs=16,
            stride=2,
            kernel_size=[5, 5],
            scope='conv_4'
        )
        # with tf.name_scope('summary_output_images'):
        #     sum_uts.conv_image_summary(conv3, (self.__image_size[0] >> 2, self.__image_size[1] >> 2, 128), 0, 'gating_fea_conv_3_out_0')
        #     sum_uts.conv_image_summary(conv3, (self.__image_size[0] >> 2, self.__image_size[1] >> 2, 128), self.__batch_size - 1, 'gating_fea_conv_3_out_n')

        flatten = slim.flatten(conv4, scope='flatten')
        return flatten

