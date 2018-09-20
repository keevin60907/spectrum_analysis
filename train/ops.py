###############################
#     Copy Right @ YCC Lab    #
###############################

# Arthor: Tsung-Shan Yang
# Date  : 2018/ 09/ 20

import tensorflow as tf
import numpy as np

def conv2d(input, weight_shape, name, reuse, activation= 'relu', padding= 'SAME'):

    shape = np.asarray(weight_shape)
    with tf.variable_scope(name, reuse= reuse):
        weight = tf.get_variable('weight', 
                     shape= shape, 
                     initializer= tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable('bias', 
                     shape= shape[-1], 
                     initializer= tf.constant_initializer(0))
        output = tf.nn.conv2d(input, weight, [1,1,1,1], padding) + bias
        
        if activation == 'relu':
            output = tf.nn.relu(output)
        
        return output
    
def pooling(input, type, name, ksize= [1,2,2,1], stride= [1,2,2,1], padding= 'SAME'):

    if type == 'max':
        return tf.nn.max_pool(input, ksize, stride, padding, name= name)
        
    if type == 'average':
        return tf.nn.avg_pool(input, ksize, stride, padding, name= name)
    
def dense(input, weight_shape, name, reuse, activation= None):

    shape = np.asarray(weight_shape)
    with tf.variable_scope(name, reuse= reuse):
        
        flatten_weight = tf.get_variable('flatten_weight',
                         shape= np.prod(shape),
                         initializer=tf.truncated_normal_initializer(stddev=0.02))
        weight = tf.reshape(flatten_weight, shape, name= 'weight')
        bias = tf.get_variable('bias',
                     shape= shape[-1],
                     initializer= tf.constant_initializer(0))
        output = tf.matmul(input, weight) + bias
        
        if activation == 'relu':
            output = tf.nn.relu(output)
        if activation == 'sigmoid':
            output = tf.nn.sigmoid(output)
        if activation == 'softmax':
            output = tf.nn.softmax(output)
        if activation == 'tanh':
            output = tf.nn.tanh(output)
        
        return output