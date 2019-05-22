# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Build the Inception v3 network on ImageNet data set.

The Inception v3 architecture is described in http://arxiv.org/abs/1512.00567

Summary of available functions:
 inference: Compute inference on the model inputs to make a prediction
 loss: Compute the loss of the prediction with respect to the labels
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf
#from slim import slim
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

bit_packer = tf.load_op_library('/home/bsc28/bsc28687/minotauro/ann/tensorflow/bit_packer/bit_packer.so')
from tensorflow.python.framework import ops
@ops.RegisterGradient("BitUnpackGpu")
def _bitunpack_grad(op, grad):
    #pgrad = bit_packer.bit_pack_gpu(grad, op.inputs[1])
    return [grad, None, None, None]

@ops.RegisterGradient("BitPackCpu")
def _bitpack_grad(op, grad):
    #with tf.device('/cpu:0'):
    #    upgrad = bit_packer.bit_unpack_cpu(grad, op.inputs[1], tf.size(op.inputs[0]), tf.shape(op.inputs[0]))
    return [grad, None]

@ops.RegisterGradient("BitPackCpuAvx")
def _bitpack_grad(op, grad):
    #with tf.device('/cpu:0'):
    #    upgrad = bit_packer.bit_unpack_cpu(grad, op.inputs[1], tf.size(op.inputs[0]), tf.shape(op.inputs[0]))
    return [grad, None]

@ops.RegisterGradient("BitPackCpuAvxOmp")
def _bitpack_grad(op, grad):
    #with tf.device('/cpu:0'):
    #    upgrad = bit_packer.bit_unpack_cpu(grad, op.inputs[1], tf.size(op.inputs[0]), tf.shape(op.inputs[0]))
    return [grad, None]

@ops.RegisterGradient("BitUnpack")
def _bitdecompress_grad(op, grad):
    pgrad = bit_packer.bit_pack_gpu(grad, 0)
    return [pgrad, None, None]

@ops.RegisterGradient("BitPack")
def _bitdecompress_grad(op, grad):
    upgrad = bit_packer.bit_unpack_cpu(grad, 0, tf.size(op.inputs[0]))
    return [upgrad, None]


FLAGS = tf.app.flags.FLAGS

# If a model is trained using multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

# Batch normalization. Constant governing the exponential moving average of
# the 'global' mean and variance for all activations.
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999

trunc_normal = lambda stddev: init_ops.truncated_normal_initializer(0.0, stddev)

def inference_alexnet_v2_bitpack(inputs,
               num_classes=1000,
               for_training=False,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               bits_ph = [],
               scope='alexnet_v2'):

  if for_training is True:
    with tf.variable_scope('', 'alexnet_v2', [inputs]) as sc:
      w_reg_l2 = None
      # Collect outputs for conv2d, fully_connected and max_pool2d.
      end_points_collection = sc.original_name_scope + '_end_points'
      with tf.variable_scope('conv1'):
        with tf.device('/cpu:0'):
          kernel1 = tf.get_variable("weights", [11, 11, 3, 64], regularizer=w_reg_l2, dtype=tf.float32)
          ckernel1 = bit_packer.bit_pack_cpu_avx(kernel1, bits_ph[0], name="bc_conv1")
          biases1 = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        kernel1_gpu = bit_packer.bit_unpack_gpu(ckernel1, bits_ph[0], tf.size(kernel1), tf.shape(kernel1), name="gpu_conv1")
        kernel1_gpu = tf.reshape(kernel1_gpu, tf.shape(kernel1))
        conv1 = tf.nn.conv2d(inputs, kernel1_gpu, strides=[1, 4, 4, 1], padding='VALID', name='conv1')
        pre_activation1 = tf.nn.bias_add(conv1, biases1, name="pre_1")
        relu1 = tf.nn.relu(pre_activation1, name="relu1")
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        tf.add_to_collection(end_points_collection, relu1)
        tf.add_to_collection(end_points_collection, pool1)

      with tf.variable_scope('conv2'):
        with tf.device('/cpu:0'):
          kernel2 = tf.get_variable("weights", [5, 5, 64, 192], regularizer=w_reg_l2, dtype=tf.float32)
          ckernel2 = bit_packer.bit_pack_cpu_avx(kernel2, bits_ph[1], name="bc_conv2")
          biases2 = tf.get_variable('biases', [192], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        kernel2_gpu = bit_packer.bit_unpack_gpu(ckernel2, bits_ph[1], tf.size(kernel2), tf.shape(kernel2), name="gpu_conv2")
        kernel2_gpu = tf.reshape(kernel2_gpu, tf.shape(kernel2))
        conv2 = tf.nn.conv2d(pool1, kernel2_gpu, strides=[1, 1, 1, 1], padding='SAME', name='conv2')
        pre_activation2 = tf.nn.bias_add(conv2, biases2)
        relu2 = tf.nn.relu(pre_activation2, name="relu2")
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
        tf.add_to_collection(end_points_collection, relu2)
        tf.add_to_collection(end_points_collection, pool2)

      with tf.variable_scope('conv3'):
        with tf.device('/cpu:0'):
          kernel3 = tf.get_variable("weights", [3, 3, 192, 384], regularizer=w_reg_l2, dtype=tf.float32)
          ckernel3 = bit_packer.bit_pack_cpu_avx(kernel3, bits_ph[2], name="bc_conv3")
          biases3 = tf.get_variable('biases', [384], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        kernel3_gpu = bit_packer.bit_unpack_gpu(ckernel3, bits_ph[2], tf.size(kernel3), tf.shape(kernel3), name="gpu_conv3")
        kernel3_gpu = tf.reshape(kernel3_gpu, tf.shape(kernel3))
        conv3 = tf.nn.conv2d(pool2, kernel3_gpu, strides=[1, 1, 1, 1], padding='SAME', name='conv3')
        pre_activation3 = tf.nn.bias_add(conv3, biases3)
        relu3 = tf.nn.relu(pre_activation3, name="relu3")

      with tf.variable_scope('conv4'):
        with tf.device('/cpu:0'):
          kernel4 = tf.get_variable("weights", [3, 3, 384, 384], regularizer=w_reg_l2, dtype=tf.float32)
          ckernel4 = bit_packer.bit_pack_cpu_avx(kernel4, bits_ph[3], name="bc_conv4")
          biases4 = tf.get_variable('biases', [384], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        kernel4_gpu = bit_packer.bit_unpack_gpu(ckernel4, bits_ph[3], tf.size(kernel4), tf.shape(kernel4), name="gpu_conv4")
        kernel4_gpu = tf.reshape(kernel4_gpu, tf.shape(kernel4))
        conv4 = tf.nn.conv2d(relu3, kernel4_gpu, strides=[1, 1, 1, 1], padding='SAME', name='conv4')
        pre_activation4 = tf.nn.bias_add(conv4, biases4)
        relu4 = tf.nn.relu(pre_activation4, name="relu4")
        tf.add_to_collection(end_points_collection, relu4)

      with tf.variable_scope('conv5'):
        with tf.device('/cpu:0'):
          kernel5 = tf.get_variable("weights", [3, 3, 384, 256], regularizer=w_reg_l2, dtype=tf.float32)
          ckernel5 = bit_packer.bit_pack_cpu_avx(kernel5, bits_ph[4], name="bc_conv5")
          biases5 = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        kernel5_gpu = bit_packer.bit_unpack_gpu(ckernel5, bits_ph[4], tf.size(kernel5), tf.shape(kernel5), name="gpu_conv5")
        kernel5_gpu = tf.reshape(kernel5_gpu, tf.shape(kernel5))
        conv5 = tf.nn.conv2d(relu4, kernel5_gpu, strides=[1, 1, 1, 1], padding='SAME', name='conv5')
        pre_activation5 = tf.nn.bias_add(conv5, biases5)
        relu5 = tf.nn.relu(pre_activation5, name="relu5")
        pool5 = tf.nn.max_pool(relu5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
        tf.add_to_collection(end_points_collection, relu5)
        tf.add_to_collection(end_points_collection, pool5)

      with tf.variable_scope('fc6'):
        with tf.device('/cpu:0'):
          w_init = tf.truncated_normal_initializer(mean=0.0, stddev=5E-3, dtype=tf.float32)
          kernel6 = tf.get_variable("weights", [5, 5, 256, 4096], initializer=w_init, regularizer=w_reg_l2, dtype=tf.float32)
          ckernel6 = bit_packer.bit_pack_cpu_avx_omp(kernel6, bits_ph[5], name="bc_conv6")
          biases6 = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        kernel6_gpu = bit_packer.bit_unpack_gpu(ckernel6, bits_ph[5], tf.size(kernel6), tf.shape(kernel6), name="gpu_fc6")
        kernel6_gpu = tf.reshape(kernel6_gpu, tf.shape(kernel6))
        conv6 = tf.nn.conv2d(pool5, kernel6_gpu, strides=[1, 1, 1, 1], padding='VALID', name='fc6')
        fc6 = tf.nn.bias_add(conv6, biases6)
        relu6 = tf.nn.relu(fc6, name="relu6")
        fc6_dropout = tf.nn.dropout(relu6, 0.5, name="fc6_dropout")
        tf.add_to_collection(end_points_collection, relu6)

      with tf.variable_scope('fc7'):
        with tf.device('/cpu:0'):
          w_init = tf.truncated_normal_initializer(mean=0.0, stddev=5E-3, dtype=tf.float32)
          kernel7 = tf.get_variable("weights", [1, 1, 4096, 4096], initializer=w_init, regularizer=w_reg_l2, dtype=tf.float32)
          ckernel7 = bit_packer.bit_pack_cpu_avx_omp(kernel7, bits_ph[6], name="bc_conv7")
          biases7 = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        kernel7_gpu = bit_packer.bit_unpack_gpu(ckernel7, bits_ph[6], tf.size(kernel7), tf.shape(kernel7), name="gpu_fc7")
        kernel7_gpu = tf.reshape(kernel7_gpu, tf.shape(kernel7))
        conv7 = tf.nn.conv2d(fc6_dropout, kernel7_gpu, strides=[1, 1, 1, 1], padding='SAME', name='fc7')
        fc7 = tf.nn.bias_add(conv7, biases7)
        relu7 = tf.nn.relu(fc7, name="relu7")
        fc7_dropout = tf.nn.dropout(relu7, 0.5, name="fc7_dropout")
        tf.add_to_collection(end_points_collection, relu7)

      with tf.variable_scope('fc71'):
        with tf.device('/cpu:0'):
          w_init = tf.truncated_normal_initializer(mean=0.0, stddev=5E-3, dtype=tf.float32)
          kernel71 = tf.get_variable("weights", [1, 1, 4096, 4096], initializer=w_init, regularizer=w_reg_l2, dtype=tf.float32)
          ckernel71 = bit_packer.bit_pack_cpu_avx_omp(kernel71, bits_ph[7], name="bc_conv71")
          biases71 = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        kernel71_gpu = bit_packer.bit_unpack_gpu(ckernel71, bits_ph[7], tf.size(kernel71), tf.shape(kernel71), name="gpu_fc71")
        kernel71_gpu = tf.reshape(kernel71_gpu, tf.shape(kernel71))
        conv71 = tf.nn.conv2d(fc7_dropout, kernel71_gpu, strides=[1, 1, 1, 1], padding='SAME', name='fc71')
        fc71 = tf.nn.bias_add(conv71, biases71)
        relu71 = tf.nn.relu(fc71, name="relu71")
        fc71_dropout = tf.nn.dropout(relu71, 0.5, name="fc71_dropout")

      #with tf.variable_scope('fc72'):
      #  with tf.device('/cpu:0'):
      #    w_init = tf.truncated_normal_initializer(mean=0.0, stddev=5E-3, dtype=tf.float32)
      #    kernel72 = tf.get_variable("weights", [1, 1, 4096, 4096], initializer=w_init, regularizer=w_reg_l2, dtype=tf.float32)
      #    ckernel72 = bit_packer.bit_pack_cpu_avx_omp(kernel72, bits_ph[8], name="bc_conv72")
      #    biases72 = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
      #  kernel72_gpu = bit_packer.bit_unpack_gpu(ckernel72, bits_ph[8], tf.size(kernel72), tf.shape(kernel72), name="gpu_fc72")
      #  kernel72_gpu = tf.reshape(kernel72_gpu, tf.shape(kernel72))
      #  conv72 = tf.nn.conv2d(fc71_dropout, kernel72_gpu, strides=[1, 1, 1, 1], padding='SAME', name='fc72')
      #  fc72 = tf.nn.bias_add(conv72, biases72)
      #  relu72 = tf.nn.relu(fc72, name="relu72")
      #  fc72_dropout = tf.nn.dropout(relu72, 0.5, name="fc72_dropout")

      with tf.variable_scope('fc8'):
        with tf.device('/cpu:0'):
          w_init = tf.truncated_normal_initializer(mean=0.0, stddev=5E-3, dtype=tf.float32)
          kernel8 = tf.get_variable("weights", [1, 1, 4096, num_classes], initializer=w_init, regularizer=w_reg_l2, dtype=tf.float32)
          ckernel8 = bit_packer.bit_pack_cpu_avx(kernel8, bits_ph[8], name="bc_conv8")
          biases8 = tf.get_variable('biases', [num_classes], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        kernel8_gpu = bit_packer.bit_unpack_gpu(ckernel8, bits_ph[8], tf.size(kernel8), tf.shape(kernel8), name="gpu_fc8")
        kernel8_gpu = tf.reshape(kernel8_gpu, tf.shape(kernel8))
        conv8 = tf.nn.conv2d(fc71_dropout, kernel8_gpu, strides=[1, 1, 1, 1], padding='SAME', name='fc8')
        net = tf.nn.bias_add(conv8, biases8)
        print("-------out", net)
        tf.add_to_collection(end_points_collection, net)

  wd1 = tf.multiply(tf.nn.l2_loss(kernel1_gpu), FLAGS.L2_reg, name='wd1')
  wd2 = tf.multiply(tf.nn.l2_loss(kernel2_gpu), FLAGS.L2_reg, name='wd2')
  wd3 = tf.multiply(tf.nn.l2_loss(kernel3_gpu), FLAGS.L2_reg, name='wd3')
  wd4 = tf.multiply(tf.nn.l2_loss(kernel4_gpu), FLAGS.L2_reg, name='wd4')
  wd5 = tf.multiply(tf.nn.l2_loss(kernel5_gpu), FLAGS.L2_reg, name='wd5')
  wd6 = tf.multiply(tf.nn.l2_loss(kernel6_gpu), FLAGS.L2_reg, name='wd6')
  wd7 = tf.multiply(tf.nn.l2_loss(kernel7_gpu), FLAGS.L2_reg, name='wd7')
  wd71 = tf.multiply(tf.nn.l2_loss(kernel71_gpu), FLAGS.L2_reg, name='wd71')
  wd8 = tf.multiply(tf.nn.l2_loss(kernel8_gpu), FLAGS.L2_reg, name='wd8')

  tf.add_to_collection(scope+'losses', wd1)
  tf.add_to_collection(scope+'losses', wd2)
  tf.add_to_collection(scope+'losses', wd3)
  tf.add_to_collection(scope+'losses', wd4)
  tf.add_to_collection(scope+'losses', wd5)
  tf.add_to_collection(scope+'losses', wd6)
  tf.add_to_collection(scope+'losses', wd7)
  tf.add_to_collection(scope+'losses', wd71)
  tf.add_to_collection(scope+'losses', wd8)

  k1_n2 = tf.norm(kernel1_gpu, name="k1_n2")
  k2_n2 = tf.norm(kernel2_gpu, name="k2_n2")
  k3_n2 = tf.norm(kernel3_gpu, name="k3_n2")
  k4_n2 = tf.norm(kernel4_gpu, name="k4_n2")
  k5_n2 = tf.norm(kernel5_gpu, name="k5_n2")
  k6_n2 = tf.norm(kernel6_gpu, name="k6_n2")
  k7_n2 = tf.norm(kernel7_gpu, name="k7_n2")
  k71_n2 = tf.norm(kernel71_gpu, name="k71_n2")
  k8_n2 = tf.norm(kernel8_gpu, name="k8_n2")
  norms = [k1_n2, k2_n2, k3_n2, k4_n2, k5_n2, k6_n2, k7_n2, k71_n2, k8_n2] 

  # Convert end_points_collection into a end_point dict.
  end_points = utils.convert_collection_to_dict(end_points_collection)
  if spatial_squeeze:
    net = array_ops.squeeze(net, [1, 2], name='fc8/squeezed')
    end_points[sc.name + '/fc8'] = net
  return net, norms


def inference_alexnet_v2(inputs,
               num_classes=1000,
               for_training=False,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               bits_ph = [],
               scope='alexnet_v2'):

  if for_training is True:
    with tf.variable_scope('', 'alexnet_v2', [inputs]) as sc:
      w_reg_l2 = None
      #w_reg_l2 = tf.contrib.layers.l2_regularizer(FLAGS.L2_reg)
      # Collect outputs for conv2d, fully_connected and max_pool2d.
      end_points_collection = sc.original_name_scope + '_end_points'
      with tf.variable_scope('conv1'):
        with tf.device('/cpu:0'):
          kernel1 = tf.get_variable("weights", [11, 11, 3, 64], regularizer=w_reg_l2, dtype=tf.float32)
          biases1 = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        conv1 = tf.nn.conv2d(inputs, kernel1, strides=[1, 4, 4, 1], padding='VALID', name='conv1')
        pre_activation1 = tf.nn.bias_add(conv1, biases1, name="pre_1")
        relu1 = tf.nn.relu(pre_activation1, name="relu1")
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        tf.add_to_collection(end_points_collection, relu1)
        tf.add_to_collection(end_points_collection, pool1)

      with tf.variable_scope('conv2'):
        with tf.device('/cpu:0'):
          kernel2 = tf.get_variable("weights", [5, 5, 64, 192], regularizer=w_reg_l2, dtype=tf.float32)
          biases2 = tf.get_variable('biases', [192], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        conv2 = tf.nn.conv2d(pool1, kernel2, strides=[1, 1, 1, 1], padding='SAME', name='conv2')
        pre_activation2 = tf.nn.bias_add(conv2, biases2)
        relu2 = tf.nn.relu(pre_activation2, name="relu2")
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
        tf.add_to_collection(end_points_collection, relu2)
        tf.add_to_collection(end_points_collection, pool2)

      with tf.variable_scope('conv3'):
        with tf.device('/cpu:0'):
          kernel3 = tf.get_variable("weights", [3, 3, 192, 384], regularizer=w_reg_l2, dtype=tf.float32)
          biases3 = tf.get_variable('biases', [384], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        conv3 = tf.nn.conv2d(pool2, kernel3, strides=[1, 1, 1, 1], padding='SAME', name='conv3')
        pre_activation3 = tf.nn.bias_add(conv3, biases3)
        relu3 = tf.nn.relu(pre_activation3, name="relu3")
        tf.add_to_collection(end_points_collection, relu3)

      with tf.variable_scope('conv4'):
        with tf.device('/cpu:0'):
          kernel4 = tf.get_variable("weights", [3, 3, 384, 384], regularizer=w_reg_l2, dtype=tf.float32)
          biases4 = tf.get_variable('biases', [384], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        conv4 = tf.nn.conv2d(relu3, kernel4, strides=[1, 1, 1, 1], padding='SAME', name='conv4')
        pre_activation4 = tf.nn.bias_add(conv4, biases4)
        relu4 = tf.nn.relu(pre_activation4, name="relu4")
        tf.add_to_collection(end_points_collection, relu4)

      with tf.variable_scope('conv5'):
        with tf.device('/cpu:0'):
          kernel5 = tf.get_variable("weights", [3, 3, 384, 256], regularizer=w_reg_l2, dtype=tf.float32)
          biases5 = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        conv5 = tf.nn.conv2d(relu4, kernel5, strides=[1, 1, 1, 1], padding='SAME', name='conv5')
        pre_activation5 = tf.nn.bias_add(conv5, biases5)
        relu5 = tf.nn.relu(pre_activation5, name="relu5")
        pool5 = tf.nn.max_pool(relu5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
        tf.add_to_collection(end_points_collection, relu5)
        tf.add_to_collection(end_points_collection, pool5)

      with tf.variable_scope('fc6'):
        with tf.device('/cpu:0'):
         w_init = tf.truncated_normal_initializer(mean=0.0, stddev=5E-3, dtype=tf.float32)
         kernel6 = tf.get_variable("weights", [5, 5, 256, 4096], initializer=w_init, regularizer=w_reg_l2, dtype=tf.float32)
         biases6 = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        conv6 = tf.nn.conv2d(pool5, kernel6, strides=[1, 1, 1, 1], padding='VALID', name='fc6')
        fc6 = tf.nn.bias_add(conv6, biases6)
        relu6 = tf.nn.relu(fc6, name="relu6")
        fc6_dropout = tf.nn.dropout(relu6, 0.5, name="fc6_dropout")
        tf.add_to_collection(end_points_collection, relu6)

      with tf.variable_scope('fc7'):
        with tf.device('/cpu:0'):
          w_init = tf.truncated_normal_initializer(mean=0.0, stddev=5E-3, dtype=tf.float32)
          kernel7 = tf.get_variable("weights", [1, 1, 4096, 4096], initializer=w_init, regularizer=w_reg_l2, dtype=tf.float32)
          biases7 = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        conv7 = tf.nn.conv2d(fc6_dropout, kernel7, strides=[1, 1, 1, 1], padding='SAME', name='fc7')
        fc7 = tf.nn.bias_add(conv7, biases7)
        relu7 = tf.nn.relu(fc7, name="relu7")
        fc7_dropout = tf.nn.dropout(relu7, 0.5, name="fc7_dropout")
        tf.add_to_collection(end_points_collection, relu6)

      with tf.variable_scope('fc71'):
        with tf.device('/cpu:0'):
          w_init = tf.truncated_normal_initializer(mean=0.0, stddev=5E-3, dtype=tf.float32)
          kernel71 = tf.get_variable("weights", [1, 1, 4096, 4096], initializer=w_init, regularizer=w_reg_l2, dtype=tf.float32)
          biases71 = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        conv71 = tf.nn.conv2d(fc7_dropout, kernel71, strides=[1, 1, 1, 1], padding='SAME', name='fc71')
        fc71 = tf.nn.bias_add(conv71, biases71)
        relu71 = tf.nn.relu(fc71, name="relu71")
        fc71_dropout = tf.nn.dropout(relu71, 0.5, name="fc71_dropout")

      #with tf.variable_scope('fc72'):
      #  with tf.device('/cpu:0'):
      #    w_init = tf.truncated_normal_initializer(mean=0.0, stddev=5E-3, dtype=tf.float32)
      #    kernel72 = tf.get_variable("weights", [1, 1, 4096, 4096], initializer=w_init, regularizer=w_reg_l2, dtype=tf.float32)
      #    biases72 = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
      #  conv72 = tf.nn.conv2d(fc71_dropout, kernel72, strides=[1, 1, 1, 1], padding='SAME', name='fc72')
      #  fc72 = tf.nn.bias_add(conv72, biases72)
      #  relu72 = tf.nn.relu(fc72, name="relu72")
      #  fc72_dropout = tf.nn.dropout(relu72, 0.5, name="fc72_dropout")

      with tf.variable_scope('fc8'):
        with tf.device('/cpu:0'):
          w_init = tf.truncated_normal_initializer(mean=0.0, stddev=5E-3, dtype=tf.float32)
          kernel8 = tf.get_variable("weights", [1, 1, 4096, num_classes], initializer=w_init, regularizer=w_reg_l2, dtype=tf.float32)
          biases8 = tf.get_variable('biases', [num_classes], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv8 = tf.nn.conv2d(fc71_dropout, kernel8, strides=[1, 1, 1, 1], padding='SAME', name='fc8')
        net = tf.nn.bias_add(conv8, biases8)
        tf.add_to_collection(end_points_collection, net)

  elif for_training is False:
    with tf.variable_scope('', 'alexnet_v2', [inputs]) as sc:
      w_reg_l2 = None
      #w_reg_l2 = tf.contrib.layers.l2_regularizer(FLAGS.L2_reg)
      # Collect outputs for conv2d, fully_connected and max_pool2d.
      end_points_collection = sc.original_name_scope + '_end_points'
      with tf.variable_scope('conv1'):
        with tf.device('/cpu:0'):
          kernel1 = tf.get_variable("weights", [11, 11, 3, 64], regularizer=w_reg_l2, dtype=tf.float32)
          biases1 = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        conv1 = tf.nn.conv2d(inputs, kernel1, strides=[1, 4, 4, 1], padding='VALID', name='conv1')
        pre_activation1 = tf.nn.bias_add(conv1, biases1, name="pre_1")
        relu1 = tf.nn.relu(pre_activation1, name="relu1")
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        tf.add_to_collection(end_points_collection, relu1)
        tf.add_to_collection(end_points_collection, pool1)

      with tf.variable_scope('conv2'):
        with tf.device('/cpu:0'):
          kernel2 = tf.get_variable("weights", [5, 5, 64, 192], regularizer=w_reg_l2, dtype=tf.float32)
          biases2 = tf.get_variable('biases', [192], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        conv2 = tf.nn.conv2d(pool1, kernel2, strides=[1, 1, 1, 1], padding='SAME', name='conv2')
        pre_activation2 = tf.nn.bias_add(conv2, biases2)
        relu2 = tf.nn.relu(pre_activation2, name="relu2")
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
        tf.add_to_collection(end_points_collection, relu2)
        tf.add_to_collection(end_points_collection, pool2)

      with tf.variable_scope('conv3'):
        with tf.device('/cpu:0'):
          kernel3 = tf.get_variable("weights", [3, 3, 192, 384], regularizer=w_reg_l2, dtype=tf.float32)
          biases3 = tf.get_variable('biases', [384], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        conv3 = tf.nn.conv2d(pool2, kernel3, strides=[1, 1, 1, 1], padding='SAME', name='conv3')
        pre_activation3 = tf.nn.bias_add(conv3, biases3)
        relu3 = tf.nn.relu(pre_activation3, name="relu3")
        tf.add_to_collection(end_points_collection, relu3)

      with tf.variable_scope('conv4'):
        with tf.device('/cpu:0'):
          kernel4 = tf.get_variable("weights", [3, 3, 384, 384], regularizer=w_reg_l2, dtype=tf.float32)
          biases4 = tf.get_variable('biases', [384], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        conv4 = tf.nn.conv2d(relu3, kernel4, strides=[1, 1, 1, 1], padding='SAME', name='conv4')
        pre_activation4 = tf.nn.bias_add(conv4, biases4)
        relu4 = tf.nn.relu(pre_activation4, name="relu4")
        tf.add_to_collection(end_points_collection, relu4)

      with tf.variable_scope('conv5'):
        with tf.device('/cpu:0'):
          kernel5 = tf.get_variable("weights", [3, 3, 384, 256], regularizer=w_reg_l2, dtype=tf.float32)
          biases5 = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        conv5 = tf.nn.conv2d(relu4, kernel5, strides=[1, 1, 1, 1], padding='SAME', name='conv5')
        pre_activation5 = tf.nn.bias_add(conv5, biases5)
        relu5 = tf.nn.relu(pre_activation5, name="relu5")
        pool5 = tf.nn.max_pool(relu5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
        tf.add_to_collection(end_points_collection, relu5)
        tf.add_to_collection(end_points_collection, pool5)

      with tf.variable_scope('fc6'):
        with tf.device('/cpu:0'):
         w_init = tf.truncated_normal_initializer(mean=0.0, stddev=5E-3, dtype=tf.float32)
         kernel6 = tf.get_variable("weights", [5, 5, 256, 4096], initializer=w_init, regularizer=w_reg_l2, dtype=tf.float32)
         biases6 = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        conv6 = tf.nn.conv2d(pool5, kernel6, strides=[1, 1, 1, 1], padding='VALID', name='fc6')
        fc6 = tf.nn.bias_add(conv6, biases6)
        relu6 = tf.nn.relu(fc6, name="relu6")
        tf.add_to_collection(end_points_collection, relu6)

      with tf.variable_scope('fc7'):
        with tf.device('/cpu:0'):
          w_init = tf.truncated_normal_initializer(mean=0.0, stddev=5E-3, dtype=tf.float32)
          kernel7 = tf.get_variable("weights", [1, 1, 4096, 4096], initializer=w_init, regularizer=w_reg_l2, dtype=tf.float32)
          biases7 = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        conv7 = tf.nn.conv2d(relu6, kernel7, strides=[1, 1, 1, 1], padding='SAME', name='fc7')
        fc7 = tf.nn.bias_add(conv7, biases7)
        relu7 = tf.nn.relu(fc7, name="relu7")

      with tf.variable_scope('fc71'):
        with tf.device('/cpu:0'):
          w_init = tf.truncated_normal_initializer(mean=0.0, stddev=5E-3, dtype=tf.float32)
          kernel71 = tf.get_variable("weights", [1, 1, 4096, 4096], initializer=w_init, regularizer=w_reg_l2, dtype=tf.float32)
          biases71 = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        conv71 = tf.nn.conv2d(relu7, kernel71, strides=[1, 1, 1, 1], padding='SAME', name='fc71')
        fc71 = tf.nn.bias_add(conv71, biases71)
        relu71 = tf.nn.relu(fc71, name="relu71")

      #with tf.variable_scope('fc72'):
      #  with tf.device('/cpu:0'):
      #    w_init = tf.truncated_normal_initializer(mean=0.0, stddev=5E-3, dtype=tf.float32)
      #    kernel72 = tf.get_variable("weights", [1, 1, 4096, 4096], initializer=w_init, regularizer=w_reg_l2, dtype=tf.float32)
      #    biases72 = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
      #  conv72 = tf.nn.conv2d(fc71_dropout, kernel72, strides=[1, 1, 1, 1], padding='SAME', name='fc72')
      #  fc72 = tf.nn.bias_add(conv72, biases72)
      #  relu72 = tf.nn.relu(fc72, name="relu72")
      #  fc72_dropout = tf.nn.dropout(relu72, 0.5, name="fc72_dropout")

      with tf.variable_scope('fc8'):
        with tf.device('/cpu:0'):
          w_init = tf.truncated_normal_initializer(mean=0.0, stddev=5E-3, dtype=tf.float32)
          kernel8 = tf.get_variable("weights", [1, 1, 4096, num_classes], initializer=w_init, regularizer=w_reg_l2, dtype=tf.float32)
          biases8 = tf.get_variable('biases', [num_classes], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv8 = tf.nn.conv2d(relu71, kernel8, strides=[1, 1, 1, 1], padding='SAME', name='fc8')
        net = tf.nn.bias_add(conv8, biases8)
        tf.add_to_collection(end_points_collection, net)


  wd1 = tf.multiply(tf.nn.l2_loss(kernel1), FLAGS.L2_reg, name='wd1')
  wd2 = tf.multiply(tf.nn.l2_loss(kernel2), FLAGS.L2_reg, name='wd2')
  wd3 = tf.multiply(tf.nn.l2_loss(kernel3), FLAGS.L2_reg, name='wd3')
  wd4 = tf.multiply(tf.nn.l2_loss(kernel4), FLAGS.L2_reg, name='wd4')
  wd5 = tf.multiply(tf.nn.l2_loss(kernel5), FLAGS.L2_reg, name='wd5')
  wd6 = tf.multiply(tf.nn.l2_loss(kernel6), FLAGS.L2_reg, name='wd6')
  wd7 = tf.multiply(tf.nn.l2_loss(kernel7), FLAGS.L2_reg, name='wd7')
  wd71 = tf.multiply(tf.nn.l2_loss(kernel71), FLAGS.L2_reg, name='wd71')
  wd8 = tf.multiply(tf.nn.l2_loss(kernel8), FLAGS.L2_reg, name='wd8')

  tf.add_to_collection(scope+'losses', wd1)
  tf.add_to_collection(scope+'losses', wd2)
  tf.add_to_collection(scope+'losses', wd3)
  tf.add_to_collection(scope+'losses', wd4)
  tf.add_to_collection(scope+'losses', wd5)
  tf.add_to_collection(scope+'losses', wd6)
  tf.add_to_collection(scope+'losses', wd7)
  tf.add_to_collection(scope+'losses', wd71)
  tf.add_to_collection(scope+'losses', wd8)

  k1_n2 = tf.norm(kernel1, name="k1_n2")
  k2_n2 = tf.norm(kernel2, name="k2_n2")
  k3_n2 = tf.norm(kernel3, name="k3_n2")
  k4_n2 = tf.norm(kernel4, name="k4_n2")
  k5_n2 = tf.norm(kernel5, name="k5_n2")
  k6_n2 = tf.norm(kernel6, name="k6_n2")
  k7_n2 = tf.norm(kernel7, name="k7_n2")
  k71_n2 = tf.norm(kernel71, name="k71_n2")
  k8_n2 = tf.norm(kernel8, name="k8_n2")
  norms = [k1_n2, k2_n2, k3_n2, k4_n2, k5_n2, k6_n2, k7_n2, k71_n2, k8_n2] 
  # Convert end_points_collection into a end_point dict.
  end_points = utils.convert_collection_to_dict(end_points_collection)
  if spatial_squeeze:
    net = array_ops.squeeze(net, [1, 2], name='fc8/squeezed')
    end_points[sc.name + '/fc8'] = net
  return net, norms


def inference(images, num_classes, for_training=False, restore_logits=True,
              scope=None):
  """Build Inception v3 model architecture.

  See here for reference: http://arxiv.org/abs/1512.00567

  Args:
    images: Images returned from inputs() or distorted_inputs().
    num_classes: number of classes
    for_training: If set to `True`, build the inference model for training.
      Kernels that operate differently for inference during training
      e.g. dropout, are appropriately configured.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: optional prefix string identifying the ImageNet tower.

  Returns:
    Logits. 2-D float Tensor.
    Auxiliary Logits. 2-D float Tensor of side-head. Used for training only.
  """
  # Parameters for BatchNorm.
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
  }
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
    with slim.arg_scope([slim.ops.conv2d],
                        stddev=0.1,
                        activation=tf.nn.relu,
                        batch_norm_params=batch_norm_params):
      logits, endpoints = slim.inception.inception_v3(
          images,
          dropout_keep_prob=0.8,
          num_classes=num_classes,
          is_training=for_training,
          restore_logits=restore_logits,
          scope=scope)

  # Add summaries for viewing model statistics on TensorBoard.
  _activation_summaries(endpoints)

  # Grab the logits associated with the side head. Employed during training.
  auxiliary_logits = endpoints['aux_logits']
  return logits, auxiliary_logits


def loss_alex(logits, labels, batch_size=None, scope=""):
  if not batch_size:
    batch_size = FLAGS.batch_size

  # Reshape the labels into a dense Tensor of
  # shape [FLAGS.batch_size, num_classes].
  sparse_labels = tf.reshape(labels, [batch_size, 1])
  indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
  concated = tf.concat(axis=1, values=[indices, sparse_labels])
  num_classes = logits.get_shape()[-1].value
  dense_labels = tf.sparse_to_dense(concated, [batch_size, num_classes], 1.0, 0.0)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=dense_labels, logits=logits, name='xentropy')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  tf.add_to_collection(scope+'losses', cross_entropy_mean)

  return [tf.add_n(tf.get_collection(scope+'losses'), name='total_loss')]

def loss(logits, labels, batch_size=None):
  """Adds all losses for the model.

  Note the final loss is not returned. Instead, the list of losses are collected
  by slim.losses. The losses are accumulated in tower_loss() and summed to
  calculate the total loss.

  Args:
    logits: List of logits from inference(). Each entry is a 2-D float Tensor.
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
    batch_size: integer
  """
  if not batch_size:
    batch_size = FLAGS.batch_size

  # Reshape the labels into a dense Tensor of
  # shape [FLAGS.batch_size, num_classes].
  sparse_labels = tf.reshape(labels, [batch_size, 1])
  indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
  concated = tf.concat(axis=1, values=[indices, sparse_labels])
  num_classes = logits[0].get_shape()[-1].value
  dense_labels = tf.sparse_to_dense(concated,
                                    [batch_size, num_classes],
                                    1.0, 0.0)

  # Cross entropy loss for the main softmax prediction.
  slim.losses.cross_entropy_loss(logits[0],
                                 dense_labels,
                                 label_smoothing=0.1,
                                 weight=1.0)

  # Cross entropy loss for the auxiliary softmax head.
  slim.losses.cross_entropy_loss(logits[1],
                                 dense_labels,
                                 label_smoothing=0.1,
                                 weight=0.4,
                                 scope='aux_loss')



def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _activation_summaries(endpoints):
  with tf.name_scope('summaries'):
    for act in endpoints.values():
      _activation_summary(act)
