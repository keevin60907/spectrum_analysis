###############################
#     Copy Right @ YCC Lab    #
###############################

# Arthor: Tsung-Shan Yang
# Date  : 2018/ 09/ 20

import numpy as np
import tnesorflow as tf
import ops

class VGG16():

    def __init__(self):
        self.name = 'VGG16'
        self.reuse = False
        self._IMAGE_SIZE = 231
        self._IMAGE_CHANNELS = 3
        self._NUM_PARAMETER = 2


    def __call__(self, input):
        with tf.variable_scope(self.name, reuse= self.reuse):

            conv1_1 = ops.conv2d(input, [3, 3, 1, 64], 'conv1_1', reuse = self.reuse)
            conv1_2 = ops.conv2d(conv1_1, [3, 3, 64, 64], 'conv1_2', reuse = self.reuse)
            pooling_1 = ops.pooling(conv1_2, 'max', 'pooling_1')

            conv2_1 = ops.conv2d(pooling_1, [3, 3, 64, 128], 'conv2_1', reuse = self.reuse)
            conv2_2 = ops.conv2d(conv2_1, [3, 3, 128, 128], 'conv2_2', reuse = self.reuse)
            pooling_2 = ops.pooling(conv2_2, 'max', 'pooling_2')

            conv3_1 = ops.conv2d(pooling_2, [3, 3, 128, 256], 'conv3_1', reuse = self.reuse)
            conv3_2 = ops.conv2d(conv3_1, [3, 3, 256, 256], 'conv3_2', reuse = self.reuse)
            conv3_3 = ops.conv2d(conv3_2, [3, 3, 256, 256], 'conv3_3', reuse = self.reuse)
            pooling_3 = ops.pooling(conv3_3, 'max', 'pooling_3') 

            conv4_1 = ops.conv2d(pooling_3, [3, 3, 256, 512], 'conv4_1', reuse = self.reuse)
            conv4_2 = ops.conv2d(conv4_1, [3, 3, 512, 512], 'conv4_2', reuse = self.reuse)
            conv4_3 = ops.conv2d(conv4_2, [3, 3, 512, 512], 'conv4_3', reuse = self.reuse)
            pooling_4 = ops.pooling(conv4_3, 'max', 'pooling_4')   

            conv5_1 = ops.conv2d(pooling_4, [3, 3, 512, 512], 'conv5_1', reuse = self.reuse)
            conv5_2 = ops.conv2d(conv5_1, [3, 3, 512, 512], 'conv5_2', reuse = self.reuse)
            conv5_3 = ops.conv2d(conv5_2, [3, 3, 512, 512], 'conv5_3', reuse = self.reuse)
            pooling_5 = ops.pooling(conv5_3, 'max', 'pooling_5')

            flatten = tf.contrib.layers.flatten(pooling_5)
            fc_shape = flatten.get_shape().as_list()      

            fc_1 = ops.dense(flatten, [fc_shape[-1], 4096], 'fc_1', reuse = self.reuse)
            fc_2 = ops.dense(fc_1, [4096, 4096], 'fc_2', reuse = self.reuse)
            fc_3 = ops.dense(fc_2, [4096, 4096], 'fc_3', reuse = self.reuse)

            output = ops.dense(fc_3, [4096, 3], 'output', activation = 'linear', reuse = self.reuse)

        self.reuse = True
        return output