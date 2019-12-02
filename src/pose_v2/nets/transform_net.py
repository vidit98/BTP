import tensorflow as tf
import numpy as np
import sys
import os

from tensorflow.keras import Model

class transform_netV1(Model):
    def __init__(self, is_training):
        super(transform_netV1, self).__init__()
        
        num_point = 20
        # self.batch_size = 1

        relu = tf.keras.layers.ReLU()

        self.conv1 = tf.keras.layers.Conv2D(64, [1,3], padding="valid",  activation=relu)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(128, [1,1], padding="valid", activation=relu)
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(1024, [1,1], padding="valid", activation=relu)
        self.batchnorm3 = tf.keras.layers.BatchNormalization()

        self.maxpool = tf.keras.layers.MaxPool2D([num_point, 1], padding="valid")

        self.fc1 = tf.keras.layers.Dense(512, activation=relu)
        self.batchnorm4 = tf.keras.layers.BatchNormalization()
        self.fc2 = tf.keras.layers.Dense(256, activation=relu)
        self.batchnorm5 = tf.keras.layers.BatchNormalization()

        init_w = tf.constant_initializer(np.random.rand(0,1))
        init_b = tf.constant_initializer(np.array([1.0,0.0,0.0,0.0]))

        self.fc3 = tf.keras.layers.Dense(4, activation=relu, kernel_initializer='zeros', bias_initializer=init_b)

    def adjust_net(self, num_point):
        self.maxpool = tf.keras.layers.MaxPool2D([num_point,1], padding="valid")

    @tf.function
    def call(self, x):
        batch_size = x.get_shape()[0]
        num_point  = x.get_shape()[1]
        x = tf.expand_dims(x, -1)
        self.adjust_net(num_point)

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.batchnorm2(self.conv2(x))
        x = self.batchnorm3(self.conv3(x))

        x = self.maxpool(x)

        x = tf.reshape(x, [1, -1])

        x = self.batchnorm4(self.fc1(x))
        x = self.batchnorm5(self.fc2(x))

        x = self.fc3(x)

        return x




# def model(point_cloud, is_training, bn_decay=None):
#     """ Input (XYZ) Transform Net, input is BxNx3 gray image
#         Return:
#             quaternions size 4 """
#     s = point_cloud.get_shape()
#     batch_size = point_cloud.get_shape()[0]
#     num_point = point_cloud.get_shape()[1]
#     print("inxsixs",s,  batch_size, num_point)
#     input_image = tf.expand_dims(point_cloud, -1)

#     tf.keras.layers..Conv2d()

#     net = tf_util.conv2d(input_image, 64, [1,3],
#                          padding='VALID', stride=[1,1],
#                          bn=True, is_training=is_training,
#                          scope='tconv1', bn_decay=bn_decay)
#     net = tf_util.conv2d(net, 128, [1,1],
#                          padding='VALID', stride=[1,1],
#                          bn=True, is_training=is_training,
#                          scope='tconv2', bn_decay=bn_decay)
#     net = tf_util.conv2d(net, 1024, [1,1],
#                          padding='VALID', stride=[1,1],
#                          bn=True, is_training=is_training,
#                          scope='tconv3', bn_decay=bn_decay)
#     net = tf_util.max_pool2d(net, [num_point,1],
#                              padding='VALID', scope='tmaxpool')

#     net = tf.reshape(net, [batch_size, -1])
#     net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
#                                   scope='tfc1', bn_decay=bn_decay)
#     net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
#                                   scope='tfc2', bn_decay=bn_decay)

#     with tf.compat.v1.variable_scope('transform_XYZ') as sc:
#         init = tf.constant_initializer(np.random.rand(0,1), dtype=tf.float32)
        
#         weights = tfe.Variable(name='weights', shape=[256, 4],
#                                   initial_value=init,
#                                   dtype=tf.float32)
#         biases = tfe.Variable(name='biases', shape=[4],
#                                  initial_value=tf.compat.v1.constant_initializer(0.0),
#                                  dtype=tf.float32)
#         biases += tf.constant([0,0,0,1], dtype=tf.float32)
#         transform = tf.matmul(net, weights)
#         transform = tf.nn.bias_add(transform, biases)
#     print("BATCH1", batch_size)
#     transform = tf.reshape(transform, [batch_size, 4])
#     print("BATCH1", transform.get_shape())
#     return transform
