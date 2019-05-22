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

import numpy as np
from collections import namedtuple
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import moving_averages

bit_packer = tf.load_op_library('/home/bsc28/bsc28687/minotauro/ann/tensorflow/bit_packer/bit_packer.so')
from tensorflow.python.framework import ops
@ops.RegisterGradient("BitUnpackGpu")
def _bitunpack_grad(op, grad):
    return [grad, None, None, None]

@ops.RegisterGradient("BitPackCpu")
def _bitpack_grad(op, grad):
    return [grad, None]

@ops.RegisterGradient("BitPackCpuAvx")
def _bitpack_grad(op, grad):
    return [grad, None]

@ops.RegisterGradient("BitPackCpuAvxOmp")
def _bitpack_grad(op, grad):
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

def inference_resnet_bitpack_val(inputs,
               num_classes=1000,
               for_training=False,
               spatial_squeeze=True,
               bits_ph = [],
               scope='resnet'):

  var_list = []
  filters = [64, 64, 128, 256, 512]
  strides = [1, 2, 2, 2]
  units = [3, 4, 6, 3]

  resnet = ResNet(FLAGS.batch_size/FLAGS.num_gpus, 0.1, 0.0002, for_training)
  bidx = 0
  with tf.variable_scope('', 'resnet', [inputs]) as sc:
    conv_init = resnet._conv('conv_init', inputs, 7, 3, filters[0], resnet._stride_arr(2), var_list, scope)
    print(conv_init)
    conv_init = tf.nn.max_pool(conv_init, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
    print(conv_init)
    bidx += 1
  
    lunit1 = []
    lunit2 = []
    lunit3 = []
    lunit4 = []

    nlunit1 = []
    nlunit2 = []
    nlunit3 = []
    nlunit4 = []

    with tf.device("/gpu:0"):
      with tf.variable_scope('unit_1_0'):
        kernels, n = resnet._bottleneck_residual_bitpack_kernels_val(filters[0], filters[1], 
                resnet._stride_arr(strides[0]), False, var_list, scope)
        lunit1.append(kernels)
        nlunit1.append(n)
      for i in range(1, units[0]):
        with tf.variable_scope('unit_1_{}'.format(i)):
          kernels, n = resnet._bottleneck_residual_bitpack_kernels_val(filters[1], filters[1], 
                  resnet._stride_arr(1), False, var_list, scope)
          lunit1.append(kernels)
          nlunit1.append(n)
      kernels1 = tf.concat(lunit1, 0)
      #cpu_kernels1 = bit_packer.bit_pack_cpu_avx_omp(kernels1, bits_ph[0])

      with tf.variable_scope('unit_2_0'):
        kernels, n = resnet._bottleneck_residual_bitpack_kernels_val(filters[1], filters[2], 
                resnet._stride_arr(strides[1]), False, var_list, scope)
        lunit2.append(kernels)
        nlunit2.append(n)
      for i in range(1, units[1]):
        with tf.variable_scope('unit_2_{}'.format(i)):
          kernels, n = resnet._bottleneck_residual_bitpack_kernels_val(filters[2], filters[2], 
                  resnet._stride_arr(1), False, var_list, scope)
        lunit2.append(kernels)
        nlunit2.append(n)
      kernels2 = tf.concat(lunit2, 0)
      #cpu_kernels2 = bit_packer.bit_pack_cpu_avx_omp(kernels2, bits_ph[1])

      with tf.variable_scope('unit_3_0'):
        kernels, n = resnet._bottleneck_residual_bitpack_kernels_val(filters[2], filters[3], 
                resnet._stride_arr(strides[2]), False, var_list, scope)
        lunit3.append(kernels)
        nlunit3.append(n)
      for i in range(1, units[2]):
        with tf.variable_scope('unit_3_{}'.format(i)):
          kernels, n = resnet._bottleneck_residual_bitpack_kernels_val(filters[3], filters[3], 
                  resnet._stride_arr(1), False, var_list, scope)
        lunit3.append(kernels)
        nlunit3.append(n)
      kernels3 = tf.concat(lunit3, 0)
      #cpu_kernels3 = bit_packer.bit_pack_cpu_avx_omp(kernels3, bits_ph[2])

      with tf.variable_scope('unit_4_0'):
        kernels, n = resnet._bottleneck_residual_bitpack_kernels_val(filters[3], filters[4], 
                resnet._stride_arr(strides[3]), False, var_list, scope)
        lunit4.append(kernels)
        nlunit4.append(n)
      for i in range(1, units[3]):
        with tf.variable_scope('unit_4_{}'.format(i)):
          kernels, n = resnet._bottleneck_residual_bitpack_kernels_val(filters[4], filters[4], 
                resnet._stride_arr(1), False, var_list, scope)
        lunit4.append(kernels)
        nlunit4.append(n)
      kernels4 = tf.concat(lunit4, 0)
      #cpu_kernels4 = bit_packer.bit_pack_cpu_avx_omp(kernels4, bits_ph[3])

    ## GPU
    #gpu_kernels1 = bit_packer.bit_unpack_gpu(cpu_kernels1, bits_ph[0], tf.size(kernels1), tf.shape(kernels1))
    #gpu_kernels1 = tf.reshape(gpu_kernels1, tf.shape(kernels1))
    gunit1 = tf.split(kernels1, nlunit1)
    #gpu_kernels2 = bit_packer.bit_unpack_gpu(cpu_kernels2, bits_ph[1], tf.size(kernels2), tf.shape(kernels2))
    #gpu_kernels2 = tf.reshape(gpu_kernels2, tf.shape(kernels2))
    gunit2 = tf.split(kernels2, nlunit2)
    #gpu_kernels3 = bit_packer.bit_unpack_gpu(cpu_kernels3, bits_ph[2], tf.size(kernels3), tf.shape(kernels3))
    #gpu_kernels3 = tf.reshape(gpu_kernels3, tf.shape(kernels3))
    gunit3 = tf.split(kernels3, nlunit3)
    #gpu_kernels4 = bit_packer.bit_unpack_gpu(cpu_kernels4, bits_ph[3], tf.size(kernels4), tf.shape(kernels4))
    #gpu_kernels4 = tf.reshape(gpu_kernels4, tf.shape(kernels4))
    gunit4 = tf.split(kernels4, nlunit4)

    #########
    gidx = 0
    with tf.variable_scope('unit_1_0'):
      x = resnet._bottleneck_residual_bitpack_val(conv_init, gunit1[gidx], filters[0], filters[1], 
              resnet._stride_arr(strides[0]), False, var_list, scope)
      gidx += 1
    for i in range(1, units[0]):
      with tf.variable_scope('unit_1_{}'.format(i)):
        x = resnet._bottleneck_residual_bitpack_val(x, gunit1[gidx], filters[1], filters[1], 
                resnet._stride_arr(1), False, var_list, scope)
        gidx += 1
 
    print(x)
    gidx = 0
    with tf.variable_scope('unit_2_0'):
      x = resnet._bottleneck_residual_bitpack_val(x, gunit2[gidx], filters[1], filters[2], 
              resnet._stride_arr(strides[1]), False, var_list, scope)
      gidx += 1
    for i in range(1, units[1]):
      with tf.variable_scope('unit_2_{}'.format(i)):
        x = resnet._bottleneck_residual_bitpack_val(x, gunit2[gidx], filters[2], filters[2], 
                resnet._stride_arr(1), False, var_list, scope)
        gidx += 1

    print(x)
    gidx = 0
    with tf.variable_scope('unit_3_0'):
      x = resnet._bottleneck_residual_bitpack_val(x, gunit3[gidx], filters[2], filters[3], 
              resnet._stride_arr(strides[2]), False, var_list, scope)
      gidx += 1
    for i in range(1, units[2]):
      with tf.variable_scope('unit_3_{}'.format(i)):
        x = resnet._bottleneck_residual_bitpack_val(x, gunit3[gidx], filters[3], filters[3], 
                resnet._stride_arr(1), False, var_list, scope)
        gidx += 1

    print(x)
    gidx = 0
    with tf.variable_scope('unit_4_0'):
      x = resnet._bottleneck_residual_bitpack_val(x, gunit4[gidx], filters[3], filters[4], 
              resnet._stride_arr(strides[3]), False, var_list, scope)
      gidx += 1
    for i in range(1, units[3]):
      with tf.variable_scope('unit_4_{}'.format(i)):
        x = resnet._bottleneck_residual_bitpack_val(x, gunit4[gidx], filters[4], filters[4], 
              resnet._stride_arr(1), False, var_list, scope)
        gidx += 1

    print(x)
    with tf.variable_scope('unit_last'):
      x = resnet._batch_norm('final_bn', x)
      x = resnet._relu(x, resnet.relu_leakiness)
      x = resnet._global_avg_pool(x)
      #x = tf.nn.avg_pool(x, ksize=[1,7,7,1], strides=[1,1,1,1], padding='SAME')

    #with tf.variable_scope('unit_fc'):
    #    x = resnet._fully_connected(x, 4096, var_list, bits_ph[4], scope)
    #with tf.variable_scope('unit_fc0'):
    #    x = resnet._fully_connected(x, 4096, var_list, bits_ph[4], scope)

    print(x)
    with tf.variable_scope('logit'):
      net = resnet._fully_connected(x, num_classes, var_list, scope)

  norms = [tf.norm(var) for var in var_list]
  return net, norms


def inference_resnet_bitpack(inputs,
               num_classes=1000,
               for_training=False,
               spatial_squeeze=True,
               bits_ph = [],
               scope='resnet'):

  var_list = []
  filters = [64, 64, 128, 256, 512]
  strides = [1, 2, 2, 2]
  units = [3, 4, 6, 3]

  resnet = ResNet(FLAGS.batch_size/FLAGS.num_gpus, 0.1, 0.0002, for_training)
  bidx = 0
  with tf.variable_scope('', 'resnet', [inputs]) as sc:
    #conv_init = resnet._conv_bitpack('conv_init', inputs, 7, 3, filters[0], resnet._stride_arr(2), var_list, bits_ph[bidx])
    conv_init = resnet._conv('conv_init', inputs, 7, 3, filters[0], resnet._stride_arr(2), var_list, scope)
    print(conv_init)
    conv_init = tf.nn.max_pool(conv_init, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
    print(conv_init)
    bidx += 1
  
    lunit1 = []
    lunit2 = []
    lunit3 = []
    lunit4 = []

    nlunit1 = []
    nlunit2 = []
    nlunit3 = []
    nlunit4 = []

    with tf.device("/cpu:0"):
      with tf.variable_scope('unit_1_0'):
        kernels, n = resnet._bottleneck_residual_bitpack_kernels(filters[0], filters[1], 
                resnet._stride_arr(strides[0]), False, var_list, scope)
        lunit1.append(kernels)
        nlunit1.append(n)
      for i in range(1, units[0]):
        with tf.variable_scope('unit_1_{}'.format(i)):
          kernels, n = resnet._bottleneck_residual_bitpack_kernels(filters[1], filters[1], 
                  resnet._stride_arr(1), False, var_list, scope)
          lunit1.append(kernels)
          nlunit1.append(n)
      kernels1 = tf.concat(lunit1, 0)
      cpu_kernels1 = bit_packer.bit_pack_cpu_avx_omp(kernels1, bits_ph[0])

      with tf.variable_scope('unit_2_0'):
        kernels, n = resnet._bottleneck_residual_bitpack_kernels(filters[1], filters[2], 
                resnet._stride_arr(strides[1]), False, var_list, scope)
        lunit2.append(kernels)
        nlunit2.append(n)
      for i in range(1, units[1]):
        with tf.variable_scope('unit_2_{}'.format(i)):
          kernels, n = resnet._bottleneck_residual_bitpack_kernels(filters[2], filters[2], 
                  resnet._stride_arr(1), False, var_list, scope)
        lunit2.append(kernels)
        nlunit2.append(n)
      kernels2 = tf.concat(lunit2, 0)
      cpu_kernels2 = bit_packer.bit_pack_cpu_avx_omp(kernels2, bits_ph[1])

      with tf.variable_scope('unit_3_0'):
        kernels, n = resnet._bottleneck_residual_bitpack_kernels(filters[2], filters[3], 
                resnet._stride_arr(strides[2]), False, var_list, scope)
        lunit3.append(kernels)
        nlunit3.append(n)
      for i in range(1, units[2]):
        with tf.variable_scope('unit_3_{}'.format(i)):
          kernels, n = resnet._bottleneck_residual_bitpack_kernels(filters[3], filters[3], 
                  resnet._stride_arr(1), False, var_list, scope)
        lunit3.append(kernels)
        nlunit3.append(n)
      kernels3 = tf.concat(lunit3, 0)
      cpu_kernels3 = bit_packer.bit_pack_cpu_avx_omp(kernels3, bits_ph[2])

      with tf.variable_scope('unit_4_0'):
        kernels, n = resnet._bottleneck_residual_bitpack_kernels(filters[3], filters[4], 
                resnet._stride_arr(strides[3]), False, var_list, scope)
        lunit4.append(kernels)
        nlunit4.append(n)
      for i in range(1, units[3]):
        with tf.variable_scope('unit_4_{}'.format(i)):
          kernels, n = resnet._bottleneck_residual_bitpack_kernels(filters[4], filters[4], 
                resnet._stride_arr(1), False, var_list, scope)
        lunit4.append(kernels)
        nlunit4.append(n)
      kernels4 = tf.concat(lunit4, 0)
      cpu_kernels4 = bit_packer.bit_pack_cpu_avx_omp(kernels4, bits_ph[3])

    ## GPU
    gpu_kernels1 = bit_packer.bit_unpack_gpu(cpu_kernels1, bits_ph[0], tf.size(kernels1), tf.shape(kernels1))
    gpu_kernels1 = tf.reshape(gpu_kernels1, tf.shape(kernels1))
    gunit1 = tf.split(gpu_kernels1, nlunit1)
    gpu_kernels2 = bit_packer.bit_unpack_gpu(cpu_kernels2, bits_ph[1], tf.size(kernels2), tf.shape(kernels2))
    gpu_kernels2 = tf.reshape(gpu_kernels2, tf.shape(kernels2))
    gunit2 = tf.split(gpu_kernels2, nlunit2)
    gpu_kernels3 = bit_packer.bit_unpack_gpu(cpu_kernels3, bits_ph[2], tf.size(kernels3), tf.shape(kernels3))
    gpu_kernels3 = tf.reshape(gpu_kernels3, tf.shape(kernels3))
    gunit3 = tf.split(gpu_kernels3, nlunit3)
    gpu_kernels4 = bit_packer.bit_unpack_gpu(cpu_kernels4, bits_ph[3], tf.size(kernels4), tf.shape(kernels4))
    gpu_kernels4 = tf.reshape(gpu_kernels4, tf.shape(kernels4))
    gunit4 = tf.split(gpu_kernels4, nlunit4)

    #########
    gidx = 0
    with tf.variable_scope('unit_1_0'):
      x = resnet._bottleneck_residual_bitpack(conv_init, gunit1[gidx], filters[0], filters[1], 
              resnet._stride_arr(strides[0]), False, var_list, scope)
      gidx += 1
    for i in range(1, units[0]):
      with tf.variable_scope('unit_1_{}'.format(i)):
        x = resnet._bottleneck_residual_bitpack(x, gunit1[gidx], filters[1], filters[1], 
                resnet._stride_arr(1), False, var_list, scope)
        gidx += 1
 
    print(x)
    gidx = 0
    with tf.variable_scope('unit_2_0'):
      x = resnet._bottleneck_residual_bitpack(x, gunit2[gidx], filters[1], filters[2], 
              resnet._stride_arr(strides[1]), False, var_list, scope)
      gidx += 1
    for i in range(1, units[1]):
      with tf.variable_scope('unit_2_{}'.format(i)):
        x = resnet._bottleneck_residual_bitpack(x, gunit2[gidx], filters[2], filters[2], 
                resnet._stride_arr(1), False, var_list, scope)
        gidx += 1

    print(x)
    gidx = 0
    with tf.variable_scope('unit_3_0'):
      x = resnet._bottleneck_residual_bitpack(x, gunit3[gidx], filters[2], filters[3], 
              resnet._stride_arr(strides[2]), False, var_list, scope)
      gidx += 1
    for i in range(1, units[2]):
      with tf.variable_scope('unit_3_{}'.format(i)):
        x = resnet._bottleneck_residual_bitpack(x, gunit3[gidx], filters[3], filters[3], 
                resnet._stride_arr(1), False, var_list, scope)
        gidx += 1

    print(x)
    gidx = 0
    with tf.variable_scope('unit_4_0'):
      x = resnet._bottleneck_residual_bitpack(x, gunit4[gidx], filters[3], filters[4], 
              resnet._stride_arr(strides[3]), False, var_list, scope)
      gidx += 1
    for i in range(1, units[3]):
      with tf.variable_scope('unit_4_{}'.format(i)):
        x = resnet._bottleneck_residual_bitpack(x, gunit4[gidx], filters[4], filters[4], 
              resnet._stride_arr(1), False, var_list, scope)
        gidx += 1

    print(x)
    with tf.variable_scope('unit_last'):
      x = resnet._batch_norm('final_bn', x)
      x = resnet._relu(x, resnet.relu_leakiness)
      x = resnet._global_avg_pool(x)
      #x = tf.nn.avg_pool(x, ksize=[1,7,7,1], strides=[1,1,1,1], padding='SAME')

    #with tf.variable_scope('unit_fc'):
    #    x = resnet._fully_connected_bitpack(x, 4096, var_list, bits_ph[4], scope)
    #with tf.variable_scope('unit_fc0'):
    #    x = resnet._fully_connected_bitpack(x, 4096, var_list, bits_ph[4], scope)

    print(x)
    with tf.variable_scope('logit'):
      net = resnet._fully_connected_bitpack(x, num_classes, var_list, bits_ph[-1], scope)

  norms = [tf.norm(var) for var in var_list]
  return net, norms


def inference_resnet(inputs,
               num_classes=1000,
               for_training=False,
               spatial_squeeze=True,
               bits_ph = [],
               scope='resnet'):

  var_list = []
  filters = [64, 64, 128, 256, 512]
  strides = [1, 2, 2, 2]
  units = [3, 4, 6, 3]

  resnet = ResNet(FLAGS.batch_size/FLAGS.num_gpus, 0.1, 0.0001, for_training)
  with tf.variable_scope('', 'resnet', [inputs]) as sc:
    conv_init = resnet._conv('conv_init', inputs, 7, 3, filters[0], resnet._stride_arr(2), var_list, scope)
    print(conv_init)
    conv_init = tf.nn.max_pool(conv_init, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
    print(conv_init)
  
    with tf.variable_scope('unit_1_0'):
      x = resnet._bottleneck_residual(conv_init, filters[0], filters[1], 
              resnet._stride_arr(strides[0]), False, var_list, scope)
    for i in range(1, units[0]):
      with tf.variable_scope('unit_1_{}'.format(i)):
        x = resnet._bottleneck_residual(x, filters[1], filters[1], 
                resnet._stride_arr(1), False, var_list, scope)
    print(x)
 
    with tf.variable_scope('unit_2_0'):
      x = resnet._bottleneck_residual(x, filters[1], filters[2], 
              resnet._stride_arr(strides[1]), False, var_list, scope)
    for i in range(1, units[1]):
      with tf.variable_scope('unit_2_{}'.format(i)):
        x = resnet._bottleneck_residual(x, filters[2], filters[2], 
                resnet._stride_arr(1), False, var_list, scope)
    print(x)

    with tf.variable_scope('unit_3_0'):
      x = resnet._bottleneck_residual(x, filters[2], filters[3], 
              resnet._stride_arr(strides[2]), False, var_list, scope)
    for i in range(1, units[2]):
      with tf.variable_scope('unit_3_{}'.format(i)):
        x = resnet._bottleneck_residual(x, filters[3], filters[3], 
                resnet._stride_arr(1), False, var_list, scope)
    print(x)

    with tf.variable_scope('unit_4_0'):
      x = resnet._bottleneck_residual(x, filters[3], filters[4], 
              resnet._stride_arr(strides[3]), False, var_list, scope)
    for i in range(1, units[3]):
      with tf.variable_scope('unit_4_{}'.format(i)):
        x = resnet._bottleneck_residual(x, filters[4], filters[4], 
                resnet._stride_arr(1), False, var_list, scope)
    print(x)

    with tf.variable_scope('unit_last'):
      x = resnet._batch_norm('final_bn', x)
      x = resnet._relu(x, resnet.relu_leakiness)
      #x = tf.nn.avg_pool(x, ksize=[1,7,7,1], strides=[1,1,1,1], padding='SAME')
      x = resnet._global_avg_pool(x)

    #with tf.variable_scope('unit_fc'):
    #    x = resnet._fully_connected(x, 4096, var_list, scope)
    #with tf.variable_scope('unit_fc0'):
    #    x = resnet._fully_connected(x, 4096, var_list, scope)
    print(x)

    with tf.variable_scope('logit'):
      net = resnet._fully_connected(x, num_classes, var_list, scope)

  norms = [tf.norm(var) for var in var_list]
  return net, norms


def loss_resnet(logits, labels, batch_size=None, scope=""):
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
  #print(tf.get_collection(scope+'losses'))

  #costs = []
  #for var in tf.get_default_graph().get_operation_by_name('GDW'):
  #    print(var)
  #    costs.append(tf.nn.l2_loss(var))
  #for var in tf.trainable_variables():
    #if var.op.name.find(r'DW') > 0:
    #  costs.append(tf.nn.l2_loss(var))
  #decay = tf.multiply(FLAGS.L2_reg, tf.add_n(costs))
  #tf.add_to_collection(scope+'losses', decay)

  return [tf.add_n(tf.get_collection(scope+'losses'), name='total_loss')]

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

#HParams = namedtuple('HParams',
#                     'batch_size, num_classes, min_lrn_rate, lrn_rate, '
#                     'num_residual_units, use_bottleneck, weight_decay_rate, '
#                     'relu_leakiness, optimizer')

#hps = HParams(batch_size=batch_size, num_classes=num_classes, min_lrn_rate=0.0001,
#                           lrn_rate=0.1, num_residual_units=5, use_bottleneck=False,
#                           weight_decay_rate=0.0002, relu_leakiness=0.1, optimizer='mom')

class ResNet(object):
  def __init__(self, bs, relu_leakiness, decay, mode):
    self.bs = int(bs)
    self.relu_leakiness = relu_leakiness
    self.decay = decay
    self.mode = mode

    self._extra_train_ops = []

  def _batch_norm(self, name, x):
    """Batch normalization."""
    with tf.variable_scope(name):
      params_shape = [x.get_shape()[-1]]
  
      beta = tf.get_variable(
          'beta', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32))
      gamma = tf.get_variable(
          'gamma', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32))
  
      if self.mode is True:
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
  
        moving_mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)
  
        decay = 0.9
        train_mean = tf.assign(moving_mean, moving_mean * decay + mean * (1 - decay))
        train_var = tf.assign(moving_variance, moving_variance * decay + variance * (1 - decay))
        #self._extra_train_ops.append(moving_averages.assign_moving_average(
        #    moving_mean, mean, 0.9))
        #self._extra_train_ops.append(moving_averages.assign_moving_average(
        #    moving_variance, variance, 0.9))
        with tf.control_dependencies([train_mean, train_var]):
          # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
          y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
          y.set_shape(x.get_shape())
          return y
      else:
        mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)
        tf.summary.histogram(mean.op.name, mean)
        tf.summary.histogram(variance.op.name, variance)
        # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
        y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
        y.set_shape(x.get_shape())
        return y
  
  def _residual(self, x, in_filter, out_filter, stride,
                activate_before_residual=False, var_list=[]):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.relu_leakiness)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.relu_leakiness)
  
    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride, var_list)
  
    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1], var_list)
  
    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      x += orig_x
  
    tf.logging.debug('image after unit %s', x.get_shape())
    return x
  
  def _residual_bitpack(self, x, in_filter, out_filter, stride,
                activate_before_residual=False, var_list=[], bits=[]):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.relu_leakiness)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.relu_leakiness)
  
    with tf.variable_scope('sub1'):
      x = self._conv_bitpack('conv1', x, 3, in_filter, out_filter, stride, var_list)
  
    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.relu_leakiness)
      x = self._conv_bitpack('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1], var_list)
  
    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      x += orig_x
  
    tf.logging.debug('image after unit %s', x.get_shape())
    return x
 
  def _bottleneck_residual(self, x, in_filter, out_filter, stride,
                           activate_before_residual=False, var_list=[], scope=''):
    """Bottleneck residual unit with 3 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('common_bn_relu'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.relu_leakiness)
        orig_x = x
    else:
      with tf.variable_scope('residual_bn_relu'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.relu_leakiness)
  
    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride, var_list, scope)
  
    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1], var_list, scope)
  
    #with tf.variable_scope('sub3'):
    #  x = self._batch_norm('bn3', x)
    #  x = self._relu(x, self.relu_leakiness)
    #  x = self._conv('conv3', x, 1, out_filter/4, out_filter, [1, 1, 1, 1], var_list, scope)

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad( orig_x, [[0, 0], [0, 0], [0, 0], [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
        #orig_x = self._conv('project', orig_x, 1, in_filter, out_filter, stride, var_list, scope)
      x = x + orig_x
  
    tf.logging.info('image after unit %s', x.get_shape())
    return x
  
  def _bottleneck_residual_bitpack_kernels(self, in_filter, out_filter, stride,
                           activate_before_residual=False, var_list=[], scope=''):
 
   ## Get variables
    with tf.device("/cpu:0"):
      with tf.variable_scope('sub1'):
         k1, n1, s1 = self._conv_get_variable('conv1', 1, in_filter, out_filter, stride, var_list, scope)
         k1 = tf.reshape(k1, [n1])
  
      with tf.variable_scope('sub2'):
        k2, n2, s2 = self._conv_get_variable('conv2', 3, out_filter, out_filter, [1, 1, 1, 1], var_list,scope)
        k2 = tf.reshape(k2, [n2])
  
      #with tf.variable_scope('sub3'):
      #  k3, n3, s3 = self._conv_get_variable('conv3', 1, out_filter/4, out_filter, [1, 1, 1, 1], var_list, scope)
      #  k3 = tf.reshape(k3, [n3])
  
      #with tf.variable_scope('sub_add'):
        #if in_filter != out_filter:
        #  k4, n4, s4 = self._conv_get_variable('project', 1, in_filter, out_filter, stride, var_list, scope)
        #  k4 = tf.reshape(k4, [n4])
        #  cpu_kernels = tf.concat([k1, k2, k4], 0)
        #  return cpu_kernels, sum([n1, n2, n4])
        #else:
      cpu_kernels = tf.concat([k1, k2], 0)
      return cpu_kernels, sum([n1, n2])

  def _bottleneck_residual_bitpack(self, x, gpu_kernels, in_filter, out_filter, stride,
                           activate_before_residual=False, var_list=[], scope=''):
    """Bottleneck residual unit with 3 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('common_bn_relu'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.relu_leakiness)
        orig_x = x
    else:
      with tf.variable_scope('residual_bn_relu'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.relu_leakiness)
  
    ### Get variables shapes
    n1, s1 = self._conv_get_variable_attr('conv1', 1, in_filter, out_filter, stride, var_list)
    n2, s2 = self._conv_get_variable_attr('conv2', 3, out_filter, out_filter, [1, 1, 1, 1], var_list)
    #n3, s3 = self._conv_get_variable_attr('conv3', 1, out_filter/4, out_filter, [1, 1, 1, 1], var_list)
    #if in_filter != out_filter:
    #  n4, s4 = self._conv_get_variable_attr('project', 1, in_filter, out_filter, stride, var_list)

    #if in_filter != out_filter:
    #  gk1, gk2, gk4 = tf.split(gpu_kernels, [n1, n2, n4])
    #  gk1 = tf.reshape(gk1, s1, name="GDW")
    #  gk2 = tf.reshape(gk2, s2, name="GDW")
    #  #gk3 = tf.reshape(gk3, s3, name="GDW")
    #  gk4 = tf.reshape(gk4, s4, name="GDW")

    #  dw1 = tf.multiply(tf.nn.l2_loss(gk1), FLAGS.L2_reg, name='gk1')
    #  tf.add_to_collection(scope+'losses', dw1)
    #  dw2 = tf.multiply(tf.nn.l2_loss(gk2), FLAGS.L2_reg, name='gk2')
    #  tf.add_to_collection(scope+'losses', dw2)
    #  #dw3 = tf.multiply(tf.nn.l2_loss(gk3), FLAGS.L2_reg, name='gk3')
    #  #tf.add_to_collection(scope+'losses', dw3)
    #  dw4 = tf.multiply(tf.nn.l2_loss(gk4), FLAGS.L2_reg, name='gk4')
    #  tf.add_to_collection(scope+'losses', dw4)
    #else:
    gk1, gk2 = tf.split(gpu_kernels, [n1, n2])
    gk1 = tf.reshape(gk1, s1, name="GDW")
    gk2 = tf.reshape(gk2, s2, name="GDW")
    #gk3 = tf.reshape(gk3, s3, name="GDW")

    dw1 = tf.multiply(tf.nn.l2_loss(gk1), FLAGS.L2_reg, name='gk1')
    tf.add_to_collection(scope+'losses', dw1)
    dw2 = tf.multiply(tf.nn.l2_loss(gk2), FLAGS.L2_reg, name='gk2')
    tf.add_to_collection(scope+'losses', dw2)
    #dw3 = tf.multiply(tf.nn.l2_loss(gk3), FLAGS.L2_reg, name='gk3')
    #tf.add_to_collection(scope+'losses', dw3)

    with tf.variable_scope('sub1'):
      x = self._conv_bitpack('conv1', x, gk1, 1, in_filter, out_filter, stride, var_list)
  
    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.relu_leakiness)
      x = self._conv_bitpack('conv2', x, gk2, 3, out_filter, out_filter, [1, 1, 1, 1], var_list)
  
    #with tf.variable_scope('sub3'):
    #  x = self._batch_norm('bn3', x)
    #  x = self._relu(x, self.relu_leakiness)
    #  x = self._conv_bitpack('conv3', x, gk3, 1, out_filter/4, out_filter, [1, 1, 1, 1], var_list)
  
    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad( orig_x, [[0, 0], [0, 0], [0, 0], [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
        #orig_x = self._conv_bitpack('project', orig_x, gk4, 1, in_filter, out_filter, stride, var_list)
      x += orig_x
  
    tf.logging.info('image after unit %s', x.get_shape())
    return x

  def _bottleneck_residual_bitpack_kernels_val(self, in_filter, out_filter, stride,
                           activate_before_residual=False, var_list=[], scope=''):
 
   ## Get variables
    with tf.variable_scope('sub1'):
       k1, n1, s1 = self._conv_get_variable('conv1', 1, in_filter, out_filter, stride, var_list, scope)
       k1 = tf.reshape(k1, [n1])

    with tf.variable_scope('sub2'):
      k2, n2, s2 = self._conv_get_variable('conv2', 3, out_filter, out_filter, [1, 1, 1, 1], var_list,scope)
      k2 = tf.reshape(k2, [n2])

    #with tf.variable_scope('sub3'):
    #  k3, n3, s3 = self._conv_get_variable('conv3', 1, out_filter/4, out_filter, [1, 1, 1, 1], var_list, scope)
    #  k3 = tf.reshape(k3, [n3])

    #with tf.variable_scope('sub_add'):
      #if in_filter != out_filter:
      #  k4, n4, s4 = self._conv_get_variable('project', 1, in_filter, out_filter, stride, var_list, scope)
      #  k4 = tf.reshape(k4, [n4])
      #  cpu_kernels = tf.concat([k1, k2, k4], 0)
      #  return cpu_kernels, sum([n1, n2, n4])
      #else:
    cpu_kernels = tf.concat([k1, k2], 0)
    return cpu_kernels, sum([n1, n2])

  def _bottleneck_residual_bitpack_val(self, x, gpu_kernels, in_filter, out_filter, stride,
                           activate_before_residual=False, var_list=[], scope=''):
    """Bottleneck residual unit with 3 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('common_bn_relu'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.relu_leakiness)
        orig_x = x
    else:
      with tf.variable_scope('residual_bn_relu'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.relu_leakiness)
  
    ### Get variables shapes
    n1, s1 = self._conv_get_variable_attr('conv1', 1, in_filter, out_filter, stride, var_list)
    n2, s2 = self._conv_get_variable_attr('conv2', 3, out_filter, out_filter, [1, 1, 1, 1], var_list)
    #n3, s3 = self._conv_get_variable_attr('conv3', 1, out_filter/4, out_filter, [1, 1, 1, 1], var_list)
    #if in_filter != out_filter:
    #  n4, s4 = self._conv_get_variable_attr('project', 1, in_filter, out_filter, stride, var_list)

    #if in_filter != out_filter:
    #  gk1, gk2, gk4 = tf.split(gpu_kernels, [n1, n2, n4])
    #  gk1 = tf.reshape(gk1, s1, name="GDW")
    #  gk2 = tf.reshape(gk2, s2, name="GDW")
    #  #gk3 = tf.reshape(gk3, s3, name="GDW")
    #  gk4 = tf.reshape(gk4, s4, name="GDW")

    #  dw1 = tf.multiply(tf.nn.l2_loss(gk1), FLAGS.L2_reg, name='gk1')
    #  tf.add_to_collection(scope+'losses', dw1)
    #  dw2 = tf.multiply(tf.nn.l2_loss(gk2), FLAGS.L2_reg, name='gk2')
    #  tf.add_to_collection(scope+'losses', dw2)
    #  #dw3 = tf.multiply(tf.nn.l2_loss(gk3), FLAGS.L2_reg, name='gk3')
    #  #tf.add_to_collection(scope+'losses', dw3)
    #  dw4 = tf.multiply(tf.nn.l2_loss(gk4), FLAGS.L2_reg, name='gk4')
    #  tf.add_to_collection(scope+'losses', dw4)
    #else:
    gk1, gk2 = tf.split(gpu_kernels, [n1, n2])
    gk1 = tf.reshape(gk1, s1, name="GDW")
    gk2 = tf.reshape(gk2, s2, name="GDW")
    #gk3 = tf.reshape(gk3, s3, name="GDW")

    dw1 = tf.multiply(tf.nn.l2_loss(gk1), FLAGS.L2_reg, name='gk1')
    tf.add_to_collection(scope+'losses', dw1)
    dw2 = tf.multiply(tf.nn.l2_loss(gk2), FLAGS.L2_reg, name='gk2')
    tf.add_to_collection(scope+'losses', dw2)
    #dw3 = tf.multiply(tf.nn.l2_loss(gk3), FLAGS.L2_reg, name='gk3')
    #tf.add_to_collection(scope+'losses', dw3)

    with tf.variable_scope('sub1'):
      x = self._conv_bitpack('conv1', x, gk1, 1, in_filter, out_filter, stride, var_list)
  
    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.relu_leakiness)
      x = self._conv_bitpack('conv2', x, gk2, 3, out_filter, out_filter, [1, 1, 1, 1], var_list)
  
    #with tf.variable_scope('sub3'):
    #  x = self._batch_norm('bn3', x)
    #  x = self._relu(x, self.relu_leakiness)
    #  x = self._conv_bitpack('conv3', x, gk3, 1, out_filter/4, out_filter, [1, 1, 1, 1], var_list)
  
    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad( orig_x, [[0, 0], [0, 0], [0, 0], [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
        #orig_x = self._conv_bitpack('project', orig_x, gk4, 1, in_filter, out_filter, stride, var_list)
      x += orig_x
  
    tf.logging.info('image after unit %s', x.get_shape())
    return x


  def _conv_get_variable(self, name, filter_size, in_filters, out_filters, strides, var_list, scope):
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      shape = [int(filter_size), int(filter_size), int(in_filters), int(out_filters)]
      with tf.device("/cpu:0"):
        kernel = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters],
            tf.float32, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
      var_list.append(kernel)
      dw1 = tf.multiply(tf.nn.l2_loss(kernel), FLAGS.L2_reg, name='gk1')
      tf.add_to_collection(scope+'losses', dw1)
      num = int(n * in_filters)
      return kernel, num, shape

  def _conv_get_variable_attr(self, name, filter_size, in_filters, out_filters, strides, var_list):
      n = filter_size * filter_size * out_filters
      shape = [int(filter_size), int(filter_size), int(in_filters), int(out_filters)]
      num = int(n * in_filters)
      return num, shape

  def _conv_bitpack(self, name, x, kernel_gpu, filter_size, in_filters, out_filters, strides, var_list):
    """Convolution."""
    with tf.variable_scope(name):
    #  n = filter_size * filter_size * out_filters
    #  with tf.device("/cpu:0"):
    #    kernel = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters],
    #        tf.float32, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
    #    ckernel = bit_packer.bit_pack_cpu_avx(kernel, bits)
    #  kernel_gpu = bit_packer.bit_unpack_gpu(ckernel, bits, tf.size(kernel), tf.shape(kernel))
    #  kernel_gpu = tf.reshape(kernel_gpu, tf.shape(kernel))
    #  var_list.append(kernel)
      return tf.nn.conv2d(x, kernel_gpu, strides, padding='SAME')

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find(r'DW') > 0:
        costs.append(tf.nn.l2_loss(var))
    #return tf.multiply(tf.decay, tf.add_n(costs))
  
  def _conv(self, name, x, filter_size, in_filters, out_filters, strides, var_list, scope):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      with tf.device("/cpu:0"):
        kernel = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters],
            tf.float32, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
      var_list.append(kernel)
      dw1 = tf.multiply(tf.nn.l2_loss(kernel), FLAGS.L2_reg, name='ngk1')
      tf.add_to_collection(scope+'losses', dw1)
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')
  
  def _batch_norm_bitpack(self, name, x, bits):
    """Batch normalization."""
    with tf.variable_scope(name):
      params_shape = [x.get_shape()[-1]]
  
      with tf.device('/cpu:0'):
        beta = tf.get_variable( 'beta', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable( 'gamma', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32))
  
        if self.mode is True:
          mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
  
          moving_mean = tf.get_variable('moving_mean', params_shape, tf.float32, 
                  initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
          moving_variance = tf.get_variable('moving_variance', params_shape, tf.float32, 
                  initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)
  
          decay = 0.9
          train_mean = tf.assign(moving_mean, moving_mean * decay + mean * (1 - decay))
          train_var = tf.assign(moving_variance, moving_variance * decay + variance * (1 - decay))
          with tf.control_dependencies([train_mean, train_var]):
            # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
            y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())
            cy = bit_packer.bit_pack_cpu_avx(y, bits)
        else:
          mean = tf.get_variable(
              'moving_mean', params_shape, tf.float32,
              initializer=tf.constant_initializer(0.0, tf.float32),
              trainable=False)
          variance = tf.get_variable(
              'moving_variance', params_shape, tf.float32,
              initializer=tf.constant_initializer(1.0, tf.float32),
              trainable=False)
          tf.summary.histogram(mean.op.name, mean)
          tf.summary.histogram(variance.op.name, variance)

          y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
          y.set_shape(x.get_shape())
          cy = bit_packer.bit_pack_cpu_avx(y, bits)
      gy = bit_packer.bit_unpack_gpu(cy, bits, tf.size(y), tf.shape(y))
      gy = tf.reshape(gy, tf.shape(y))
      return gy

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
  
  def _fully_connected(self, x, out_dim, var_list, scope):
    """FullyConnected layer for final output."""
    x = tf.reshape(x, [self.bs, -1])
    with tf.device("/cpu:0"):
      w = tf.get_variable('DW', [x.get_shape()[1], out_dim],
          initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
      b = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer())
    var_list.append(w)
    dw = tf.multiply(tf.nn.l2_loss(w), FLAGS.L2_reg, name='DW')
    tf.add_to_collection(scope+'losses', dw)
    return tf.nn.xw_plus_b(x, w, b)
  
  def _fully_connected_bitpack(self, x, out_dim, var_list, bits, scope):
    """FullyConnected layer for final output."""
    x = tf.reshape(x, [self.bs, -1])
    with tf.device("/cpu:0"):
      w = tf.get_variable('DW', [x.get_shape()[1], out_dim],
          initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
      cw = bit_packer.bit_pack_cpu_avx_omp(w, bits)
      b = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer())
    w_gpu = bit_packer.bit_unpack_gpu(cw, bits, tf.size(w), tf.shape(w))
    w_gpu = tf.reshape(w_gpu, tf.shape(w), name="GDW")
    var_list.append(w_gpu)
    dw = tf.multiply(tf.nn.l2_loss(w_gpu), FLAGS.L2_reg, name='DW')
    tf.add_to_collection(scope+'losses', dw)
    return tf.nn.xw_plus_b(x, w_gpu, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])
  
  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

