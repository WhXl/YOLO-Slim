#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : HL
#   File name   : common.py
#   Author      : WXing195
#   Created date: 2019-
#   Description :
#
#================================================================

import tensorflow as tf

weight_decay = 1e-4
"yolov3"

def convolutional0(input_data, filters_shape, trainable, name, threshold=None, downsample=False, bn=True, activate = True ,purning=True):

    with tf.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"


        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))
        if purning:
            weight_shape = tf.shape(weight)
            dense = weight_shape[0] * weight_shape[1] * weight_shape[2] * weight_shape[-1]
            weight_ = tf.reshape(weight, [-1, weight_shape[-1]])
            L2_norm = tf.norm(weight_, axis=0)

            threshold = tf.cast((1 - threshold), tf.float32)

            threshold_num = tf.multiply(tf.cast(weight_shape[-1], tf.float32), threshold)
            threshold_num = tf.cast(threshold_num, tf.int32)

            threshold = tf.gather(L2_norm, tf.nn.top_k(L2_norm, k=weight_shape[-1]).indices[threshold_num])
            bool_index = tf.cast(L2_norm > threshold, tf.float32)

            weight = tf.multiply(weight_, bool_index)

        tf.summary.histogram(name + "/weight",weight)
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)
        if activate == True: conv = relu(conv)

    return conv

def convolutional(input_data, filters_shape, trainable, name,  downsample=False, bn=True,
                  activate = True):

    with tf.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"

        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01),
                                 regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        # if purning == True:
        #     weight_shape = tf.shape(weight)
        #     dense = weight_shape[0] * weight_shape[1] * weight_shape[2] * weight_shape[-1]
        #     weight_ = tf.reshape(weight, [-1, weight_shape[-1]])
        #     L2_norm = tf.norm(weight_, axis=0)
        #
        #     threshold = tf.cast((1 - threshold), tf.float32)
        #
        #     threshold_num = tf.multiply(tf.cast(weight_shape[-1], tf.float32), threshold)
        #     threshold_num = tf.cast(threshold_num, tf.int32)
        #
        #     threshold = tf.gather(L2_norm, tf.nn.top_k(L2_norm, k=weight_shape[-1]).indices[threshold_num])
        #     bool_index = tf.cast(L2_norm > threshold, tf.float32)
        #
        #     weight = tf.multiply(weight_, bool_index)
        #
        # tf.summary.histogram(name + "/Weights",weight)
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)
        if activate == True: conv = tf.nn.leaky_relu(conv,alpha=0.1)

    return conv

def upsample(input_data, name, method="deconv"):
    assert method in ["resize", "deconv"]

    if method == "resize":
        with tf.variable_scope(name):
            input_shape = tf.shape(input_data)
            output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))

    if method == "deconv":
        # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT optimization
        numm_filter = input_data.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(input_data, numm_filter, kernel_size=2, padding='same',
                                            strides=(2,2), kernel_initializer=tf.random_normal_initializer())

    return output

def conv2d(input_, output_dim, k_h, k_w, d_h, d_w, stddev=0.01, name='conv2d', bias=False):
    with tf.variable_scope(name):
        input_dim = input_.get_shape().as_list()[-1]
        w = tf.get_variable('w', [k_h, k_w, input_dim, output_dim],
              regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
              initializer=tf.truncated_normal_initializer(stddev=stddev),trainable=True)

        tf.summary.histogram(name + "w",w)
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        if bias:
            biases = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        return conv


def conv_1X1(input, output_dim,name, bias=False):
    with tf.name_scope(name):
        return conv2d(input, output_dim, 1,1,1,1, stddev=0.01, name=name, bias=bias)

def relu(x,name="relu6"):
    return tf.nn.relu6(x,name)

def dwise_conv(input, k_h=3,k_w=3,channel_multiplier=1,strides=[1,1,1,1],
               padding="SAME",stddev=0.02,name="dwise_conv",bias=False):
    with tf.variable_scope(name):
        in_channel = input.get_shape().as_list()[-1]
        w = tf.get_variable(name="w", shape=[k_h,k_w,in_channel,channel_multiplier],
                            trainable=True,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            initializer=tf.random_normal_initializer(stddev=stddev))
        # w = tf.get_variable(name="w", shape=[k_h, k_w, in_channel, channel_multiplier],trainable=True,
        #                                         initializer=tf.random_normal_initializer(stddev=stddev))
        tf.summary.histogram(name + "weight",w)
        conv = tf.nn.depthwise_conv2d(input,w,strides,padding,rate=None,name=None,data_format=None)
        if bias:
            biases = tf.get_variable("bias",[in_channel*channel_multiplier],initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv,biases)
        return conv


def res_block(input_data, expansin_ratio, output_dim, stride, trainable, name, bias=False,shortcut=True,bn = True):
    with tf.variable_scope(name):
        #pw
        bottleneck_dim = round(expansin_ratio*input_data.get_shape().as_list()[-1])#输出维度
        net = conv_1X1(input_data,bottleneck_dim,name="pw",bias = bias)
        if bn:
            net = tf.layers.batch_normalization(net, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
        net = relu(net)
        #dw
        net = dwise_conv(net,strides=[1,stride,stride,1],name = "dw",bias=bias)
        if bn:
            net = tf.layers.batch_normalization(net, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
        net = relu(net)
        #pw & linear
        net = conv_1X1(net,output_dim,name = "pw_linear",bias = bias)
        if bn:
            net = tf.layers.batch_normalization(net, beta_initializer=tf.zeros_initializer(),
                                                gamma_initializer=tf.ones_initializer(),
                                                moving_mean_initializer=tf.zeros_initializer(),
                                                moving_variance_initializer=tf.ones_initializer(), training=trainable)
        #element wise eladdm, only for stride==1
        if shortcut and stride ==1:
            in_dim = int(input_data.get_shape().as_list()[-1])
            if in_dim != output_dim:
                ins = conv_1X1(input_data,output_dim,name="ex_dim")
                net = ins + net
            else:
                net = input_data + net
        return net
def pwise_block(input_data, output_dim,trainable, name, bias=False,bn =True):
    with tf.name_scope(name):
        input_data=conv_1X1(input_data, output_dim, bias=bias, name='pwb')
        if bn:
            input_data = tf.layers.batch_normalization(input_data, beta_initializer=tf.zeros_initializer(),
                                                gamma_initializer=tf.ones_initializer(),
                                                moving_mean_initializer=tf.zeros_initializer(),
                                                moving_variance_initializer=tf.ones_initializer(), training=trainable)
        input_data=relu(input_data)
        return input_data





