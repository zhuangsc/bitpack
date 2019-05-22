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

def inference_vgg_19_bitpack(inputs, 
                    num_classes=1000, 
                    for_training=True, 
                    dropout_keep_prob=0.5, 
                    bits_ph=[], 
                    spatial_squeeze=True, 
                    scope='vgg_19'):

  if len(bits_ph) != 19:
    raise Exception("bits_ph != 19")
  if for_training is True:
    with tf.variable_scope('', 'vgg_19', [inputs]) as sc:
      w_reg_l2 = None
      # Collect outputs for conv2d, fully_connected and max_pool2d.
      end_points_collection = sc.original_name_scope + '_end_points'
      with tf.variable_scope('conv1'):
        with tf.device('/cpu:0'):
          kernel10 = tf.get_variable("weights0", [3, 3, 3, 64], regularizer=w_reg_l2, dtype=tf.float32)
          biases10 = tf.get_variable('biases0', [64], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel11 = tf.get_variable("weights1", [3, 3, 64, 64], regularizer=w_reg_l2, dtype=tf.float32)
          biases11 = tf.get_variable('biases1', [64], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          ckernel10 = bit_packer.bit_pack_cpu_avx(kernel10, bits_ph[0], name="bc_conv10")
          ckernel11 = bit_packer.bit_pack_cpu_avx(kernel11, bits_ph[1], name="bc_conv11")
        kernel10_gpu = bit_packer.bit_unpack_gpu(ckernel10, bits_ph[0], tf.size(kernel10), tf.shape(kernel10), name="gpu_conv10")
        kernel10_gpu = tf.reshape(kernel10_gpu, tf.shape(kernel10))
        kernel11_gpu = bit_packer.bit_unpack_gpu(ckernel11, bits_ph[1], tf.size(kernel11), tf.shape(kernel11), name="gpu_conv11")
        kernel11_gpu = tf.reshape(kernel11_gpu, tf.shape(kernel11))
        conv10 = tf.nn.conv2d(inputs, kernel10_gpu, strides=[1, 1, 1, 1], padding='SAME', name='conv10')
        pre_activation10 = tf.nn.bias_add(conv10, biases10, name="pre_10")
        relu10 = tf.nn.relu(pre_activation10, name="relu10")
        conv11 = tf.nn.conv2d(relu10, kernel11_gpu, strides=[1, 1, 1, 1], padding='SAME', name='conv11')
        pre_activation11 = tf.nn.bias_add(conv11, biases11, name="pre_11")
        relu11 = tf.nn.relu(pre_activation11, name="relu11")
        pool1 = tf.nn.max_pool(relu11, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

      with tf.variable_scope('conv2'):
        with tf.device('/cpu:0'):
          kernel20 = tf.get_variable("weights0", [3, 3, 64, 128], regularizer=w_reg_l2, dtype=tf.float32)
          biases20 = tf.get_variable('biases0', [128], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel21 = tf.get_variable("weights1", [3, 3, 128, 128], regularizer=w_reg_l2, dtype=tf.float32)
          biases21 = tf.get_variable('biases1', [128], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          ckernel20 = bit_packer.bit_pack_cpu_avx(kernel20, bits_ph[2], name="bc_conv20")
          ckernel21 = bit_packer.bit_pack_cpu_avx(kernel21, bits_ph[3], name="bc_conv21")
        kernel20_gpu = bit_packer.bit_unpack_gpu(ckernel20, bits_ph[2], tf.size(kernel20), tf.shape(kernel20), name="gpu_conv20")
        kernel20_gpu = tf.reshape(kernel20_gpu, tf.shape(kernel20))
        kernel21_gpu = bit_packer.bit_unpack_gpu(ckernel21, bits_ph[3], tf.size(kernel21), tf.shape(kernel21), name="gpu_conv21")
        kernel21_gpu = tf.reshape(kernel21_gpu, tf.shape(kernel21))
        conv20 = tf.nn.conv2d(pool1, kernel20_gpu, strides=[1, 1, 1, 1], padding='SAME', name='conv20')
        pre_activation20 = tf.nn.bias_add(conv20, biases20, name="pre_20")
        relu20 = tf.nn.relu(pre_activation20, name="relu20")
        conv21 = tf.nn.conv2d(relu20, kernel21_gpu, strides=[1, 1, 1, 1], padding='SAME', name='conv21')
        pre_activation21 = tf.nn.bias_add(conv21, biases21, name="pre_21")
        relu21 = tf.nn.relu(pre_activation21, name="relu21")
        pool2 = tf.nn.max_pool(relu21, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

      with tf.variable_scope('conv3'):
        with tf.device('/cpu:0'):
          kernel30 = tf.get_variable("weights0", [3, 3, 128, 256], regularizer=w_reg_l2, dtype=tf.float32)
          biases30 = tf.get_variable('biases0', [256], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel31 = tf.get_variable("weights1", [3, 3, 256, 256], regularizer=w_reg_l2, dtype=tf.float32)
          biases31 = tf.get_variable('biases1', [256], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel32 = tf.get_variable("weights2", [3, 3, 256, 256], regularizer=w_reg_l2, dtype=tf.float32)
          biases32 = tf.get_variable('biases2', [256], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel33 = tf.get_variable("weights3", [3, 3, 256, 256], regularizer=w_reg_l2, dtype=tf.float32)
          biases33 = tf.get_variable('biases3', [256], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          ckernel30 = bit_packer.bit_pack_cpu_avx(kernel30, bits_ph[4], name="bc_conv30")
          ckernel31 = bit_packer.bit_pack_cpu_avx(kernel31, bits_ph[5], name="bc_conv31")
          ckernel32 = bit_packer.bit_pack_cpu_avx(kernel32, bits_ph[6], name="bc_conv32")
          ckernel33 = bit_packer.bit_pack_cpu_avx(kernel33, bits_ph[7], name="bc_conv33")
        kernel30_gpu = bit_packer.bit_unpack_gpu(ckernel30, bits_ph[4], tf.size(kernel30), tf.shape(kernel30), name="gpu_conv30")
        kernel30_gpu = tf.reshape(kernel30_gpu, tf.shape(kernel30))
        kernel31_gpu = bit_packer.bit_unpack_gpu(ckernel31, bits_ph[5], tf.size(kernel31), tf.shape(kernel31), name="gpu_conv31")
        kernel31_gpu = tf.reshape(kernel31_gpu, tf.shape(kernel31))
        kernel32_gpu = bit_packer.bit_unpack_gpu(ckernel32, bits_ph[6], tf.size(kernel32), tf.shape(kernel32), name="gpu_conv32")
        kernel32_gpu = tf.reshape(kernel32_gpu, tf.shape(kernel32))
        kernel33_gpu = bit_packer.bit_unpack_gpu(ckernel33, bits_ph[7], tf.size(kernel33), tf.shape(kernel33), name="gpu_conv33")
        kernel33_gpu = tf.reshape(kernel33_gpu, tf.shape(kernel33))
        conv30 = tf.nn.conv2d(pool2, kernel30_gpu, strides=[1, 1, 1, 1], padding='SAME', name='conv30')
        pre_activation30 = tf.nn.bias_add(conv30, biases30, name="pre_30")
        relu30 = tf.nn.relu(pre_activation30, name="relu30")
        conv31 = tf.nn.conv2d(relu30, kernel31_gpu, strides=[1, 1, 1, 1], padding='SAME', name='conv31')
        pre_activation31 = tf.nn.bias_add(conv31, biases31, name="pre_31")
        relu31 = tf.nn.relu(pre_activation31, name="relu31")
        conv32 = tf.nn.conv2d(relu31, kernel32_gpu, strides=[1, 1, 1, 1], padding='SAME', name='conv32')
        pre_activation32 = tf.nn.bias_add(conv32, biases32, name="pre_32")
        relu32 = tf.nn.relu(pre_activation32, name="relu32")
        conv33 = tf.nn.conv2d(relu32, kernel33_gpu, strides=[1, 1, 1, 1], padding='SAME', name='conv33')
        pre_activation33 = tf.nn.bias_add(conv33, biases33, name="pre_33")
        relu33 = tf.nn.relu(pre_activation33, name="relu33")
        pool3 = tf.nn.max_pool(relu33, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

      with tf.variable_scope('conv4'):
        with tf.device('/cpu:0'):
          kernel40 = tf.get_variable("weights0", [3, 3, 256, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases40 = tf.get_variable('biases0', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel41 = tf.get_variable("weights1", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases41 = tf.get_variable('biases1', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel42 = tf.get_variable("weights2", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases42 = tf.get_variable('biases2', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel43 = tf.get_variable("weights3", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases43 = tf.get_variable('biases3', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          ckernel40 = bit_packer.bit_pack_cpu_avx(kernel40, bits_ph[8], name="bc_conv40")
          ckernel41 = bit_packer.bit_pack_cpu_avx(kernel41, bits_ph[9], name="bc_conv41")
          ckernel42 = bit_packer.bit_pack_cpu_avx(kernel42, bits_ph[10], name="bc_conv42")
          ckernel43 = bit_packer.bit_pack_cpu_avx(kernel43, bits_ph[11], name="bc_conv43")
        kernel40_gpu = bit_packer.bit_unpack_gpu(ckernel40, bits_ph[8], tf.size(kernel40), tf.shape(kernel40), name="gpu_conv40")
        kernel40_gpu = tf.reshape(kernel40_gpu, tf.shape(kernel40))
        kernel41_gpu = bit_packer.bit_unpack_gpu(ckernel41, bits_ph[9], tf.size(kernel41), tf.shape(kernel41), name="gpu_conv41")
        kernel41_gpu = tf.reshape(kernel41_gpu, tf.shape(kernel41))
        kernel42_gpu = bit_packer.bit_unpack_gpu(ckernel42, bits_ph[10], tf.size(kernel42), tf.shape(kernel42), name="gpu_conv42")
        kernel42_gpu = tf.reshape(kernel42_gpu, tf.shape(kernel42))
        kernel43_gpu = bit_packer.bit_unpack_gpu(ckernel43, bits_ph[11], tf.size(kernel43), tf.shape(kernel43), name="gpu_conv43")
        kernel43_gpu = tf.reshape(kernel43_gpu, tf.shape(kernel43))
        conv40 = tf.nn.conv2d(pool3, kernel40_gpu, strides=[1, 1, 1, 1], padding='SAME', name='conv40')
        pre_activation40 = tf.nn.bias_add(conv40, biases40, name="pre_40")
        relu40 = tf.nn.relu(pre_activation40, name="relu40")
        conv41 = tf.nn.conv2d(relu40, kernel41_gpu, strides=[1, 1, 1, 1], padding='SAME', name='conv41')
        pre_activation41 = tf.nn.bias_add(conv41, biases41, name="pre_41")
        relu41 = tf.nn.relu(pre_activation41, name="relu41")
        conv42 = tf.nn.conv2d(relu41, kernel42_gpu, strides=[1, 1, 1, 1], padding='SAME', name='conv42')
        pre_activation42 = tf.nn.bias_add(conv42, biases42, name="pre_42")
        relu42 = tf.nn.relu(pre_activation42, name="relu42")
        conv43 = tf.nn.conv2d(relu42, kernel43_gpu, strides=[1, 1, 1, 1], padding='SAME', name='conv43')
        pre_activation43 = tf.nn.bias_add(conv43, biases43, name="pre_43")
        relu43 = tf.nn.relu(pre_activation43, name="relu43")
        pool4 = tf.nn.max_pool(relu43, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

      with tf.variable_scope('conv5'):
        with tf.device('/cpu:0'):
          kernel50 = tf.get_variable("weights0", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases50 = tf.get_variable('biases0', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel51 = tf.get_variable("weights1", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases51 = tf.get_variable('biases1', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel52 = tf.get_variable("weights2", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases52 = tf.get_variable('biases2', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel53 = tf.get_variable("weights3", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases53 = tf.get_variable('biases3', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          ckernel50 = bit_packer.bit_pack_cpu_avx(kernel50, bits_ph[12], name="bc_conv50")
          ckernel51 = bit_packer.bit_pack_cpu_avx(kernel51, bits_ph[13], name="bc_conv51")
          ckernel52 = bit_packer.bit_pack_cpu_avx(kernel52, bits_ph[14], name="bc_conv52")
          ckernel53 = bit_packer.bit_pack_cpu_avx(kernel53, bits_ph[15], name="bc_conv53")
        kernel50_gpu = bit_packer.bit_unpack_gpu(ckernel50, bits_ph[12], tf.size(kernel50), tf.shape(kernel50), name="gpu_conv50")
        kernel50_gpu = tf.reshape(kernel50_gpu, tf.shape(kernel50))
        kernel51_gpu = bit_packer.bit_unpack_gpu(ckernel51, bits_ph[13], tf.size(kernel51), tf.shape(kernel51), name="gpu_conv51")
        kernel51_gpu = tf.reshape(kernel51_gpu, tf.shape(kernel51))
        kernel52_gpu = bit_packer.bit_unpack_gpu(ckernel52, bits_ph[14], tf.size(kernel52), tf.shape(kernel52), name="gpu_conv52")
        kernel52_gpu = tf.reshape(kernel52_gpu, tf.shape(kernel52))
        kernel53_gpu = bit_packer.bit_unpack_gpu(ckernel53, bits_ph[15], tf.size(kernel53), tf.shape(kernel53), name="gpu_conv53")
        kernel53_gpu = tf.reshape(kernel53_gpu, tf.shape(kernel53))
        conv50 = tf.nn.conv2d(pool4, kernel50_gpu, strides=[1, 1, 1, 1], padding='SAME', name='conv50')
        pre_activation50 = tf.nn.bias_add(conv50, biases50, name="pre_50")
        relu50 = tf.nn.relu(pre_activation50, name="relu50")
        conv51 = tf.nn.conv2d(relu50, kernel51_gpu, strides=[1, 1, 1, 1], padding='SAME', name='conv51')
        pre_activation51 = tf.nn.bias_add(conv51, biases51, name="pre_51")
        relu51 = tf.nn.relu(pre_activation51, name="relu51")
        conv52 = tf.nn.conv2d(relu51, kernel52_gpu, strides=[1, 1, 1, 1], padding='SAME', name='conv52')
        pre_activation52 = tf.nn.bias_add(conv52, biases52, name="pre_52")
        relu52 = tf.nn.relu(pre_activation52, name="relu52")
        conv53 = tf.nn.conv2d(relu52, kernel53_gpu, strides=[1, 1, 1, 1], padding='SAME', name='conv53')
        pre_activation53 = tf.nn.bias_add(conv53, biases53, name="pre_53")
        relu53 = tf.nn.relu(pre_activation53, name="relu53")
        pool5 = tf.nn.max_pool(relu53, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

      with tf.variable_scope('fc6'):
        with tf.device('/cpu:0'):
          kernel6 = tf.get_variable("weights", [7, 7, 512, 4096], regularizer=w_reg_l2, dtype=tf.float32)
          ckernel6 = bit_packer.bit_pack_cpu_avx_omp(kernel6, bits_ph[8], name="bc_conv6")
          biases6 = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        kernel6_gpu = bit_packer.bit_unpack_gpu(ckernel6, bits_ph[8], tf.size(kernel6), tf.shape(kernel6), name="gpu_fc6")
        kernel6_gpu = tf.reshape(kernel6_gpu, tf.shape(kernel6))
        conv6 = tf.nn.conv2d(pool5, kernel6_gpu, strides=[1, 1, 1, 1], padding='VALID', name='fc6')
        fc6 = tf.nn.bias_add(conv6, biases6)
        relu6 = tf.nn.relu(fc6, name="relu6")
        fc6_dropout = tf.nn.dropout(relu6, 0.5, name="fc6_dropout")

      with tf.variable_scope('fc7'):
        with tf.device('/cpu:0'):
          kernel7 = tf.get_variable("weights", [1, 1, 4096, 4096], regularizer=w_reg_l2, dtype=tf.float32)
          ckernel7 = bit_packer.bit_pack_cpu_avx_omp(kernel7, bits_ph[9], name="bc_conv7")
          biases7 = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        kernel7_gpu = bit_packer.bit_unpack_gpu(ckernel7, bits_ph[9], tf.size(kernel7), tf.shape(kernel7), name="gpu_fc7")
        kernel7_gpu = tf.reshape(kernel7_gpu, tf.shape(kernel7))
        conv7 = tf.nn.conv2d(fc6_dropout, kernel7_gpu, strides=[1, 1, 1, 1], padding='SAME', name='fc7')
        fc7 = tf.nn.bias_add(conv7, biases7)
        relu7 = tf.nn.relu(fc7, name="relu7")
        fc7_dropout = tf.nn.dropout(relu7, 0.5, name="fc7_dropout")

      with tf.variable_scope('fc8'):
        with tf.device('/cpu:0'):
          kernel8 = tf.get_variable("weights", [1, 1, 4096, num_classes], regularizer=w_reg_l2, dtype=tf.float32)
          ckernel8 = bit_packer.bit_pack_cpu_avx(kernel8, bits_ph[10], name="bc_conv8")
          biases8 = tf.get_variable('biases', [num_classes], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        kernel8_gpu = bit_packer.bit_unpack_gpu(ckernel8, bits_ph[10], tf.size(kernel8), tf.shape(kernel8), name="gpu_fc8")
        kernel8_gpu = tf.reshape(kernel8_gpu, tf.shape(kernel8))
        conv8 = tf.nn.conv2d(fc7_dropout, kernel8_gpu, strides=[1, 1, 1, 1], padding='SAME', name='fc8')
        net = tf.nn.bias_add(conv8, biases8)

  wd10 = tf.multiply(tf.nn.l2_loss(kernel10_gpu), FLAGS.L2_reg, name='wd10')
  wd11 = tf.multiply(tf.nn.l2_loss(kernel11_gpu), FLAGS.L2_reg, name='wd11')
  wd20 = tf.multiply(tf.nn.l2_loss(kernel20_gpu), FLAGS.L2_reg, name='wd20')
  wd21 = tf.multiply(tf.nn.l2_loss(kernel21_gpu), FLAGS.L2_reg, name='wd21')
  wd30 = tf.multiply(tf.nn.l2_loss(kernel30_gpu), FLAGS.L2_reg, name='wd30')
  wd31 = tf.multiply(tf.nn.l2_loss(kernel31_gpu), FLAGS.L2_reg, name='wd31')
  wd32 = tf.multiply(tf.nn.l2_loss(kernel32_gpu), FLAGS.L2_reg, name='wd32')
  wd33 = tf.multiply(tf.nn.l2_loss(kernel33_gpu), FLAGS.L2_reg, name='wd33')
  wd40 = tf.multiply(tf.nn.l2_loss(kernel40_gpu), FLAGS.L2_reg, name='wd40')
  wd41 = tf.multiply(tf.nn.l2_loss(kernel41_gpu), FLAGS.L2_reg, name='wd41')
  wd42 = tf.multiply(tf.nn.l2_loss(kernel42_gpu), FLAGS.L2_reg, name='wd42')
  wd43 = tf.multiply(tf.nn.l2_loss(kernel43_gpu), FLAGS.L2_reg, name='wd43')
  wd50 = tf.multiply(tf.nn.l2_loss(kernel50_gpu), FLAGS.L2_reg, name='wd50')
  wd51 = tf.multiply(tf.nn.l2_loss(kernel51_gpu), FLAGS.L2_reg, name='wd51')
  wd52 = tf.multiply(tf.nn.l2_loss(kernel52_gpu), FLAGS.L2_reg, name='wd52')
  wd53 = tf.multiply(tf.nn.l2_loss(kernel53_gpu), FLAGS.L2_reg, name='wd53')
  wd6 = tf.multiply(tf.nn.l2_loss(kernel6_gpu), FLAGS.L2_reg, name='wd6')
  wd7 = tf.multiply(tf.nn.l2_loss(kernel7_gpu), FLAGS.L2_reg, name='wd7')
  wd8 = tf.multiply(tf.nn.l2_loss(kernel8_gpu), FLAGS.L2_reg, name='wd8')

  tf.add_to_collection(scope+'losses', wd10)
  tf.add_to_collection(scope+'losses', wd11)
  tf.add_to_collection(scope+'losses', wd20)
  tf.add_to_collection(scope+'losses', wd21)
  tf.add_to_collection(scope+'losses', wd30)
  tf.add_to_collection(scope+'losses', wd31)
  tf.add_to_collection(scope+'losses', wd32)
  tf.add_to_collection(scope+'losses', wd33)
  tf.add_to_collection(scope+'losses', wd40)
  tf.add_to_collection(scope+'losses', wd41)
  tf.add_to_collection(scope+'losses', wd42)
  tf.add_to_collection(scope+'losses', wd43)
  tf.add_to_collection(scope+'losses', wd50)
  tf.add_to_collection(scope+'losses', wd51)
  tf.add_to_collection(scope+'losses', wd52)
  tf.add_to_collection(scope+'losses', wd53)
  tf.add_to_collection(scope+'losses', wd6)
  tf.add_to_collection(scope+'losses', wd7)
  tf.add_to_collection(scope+'losses', wd8)

  k10_n2 = tf.norm(kernel10_gpu, name="k10_n2")
  k11_n2 = tf.norm(kernel11_gpu, name="k11_n2")
  k20_n2 = tf.norm(kernel20_gpu, name="k20_n2")
  k21_n2 = tf.norm(kernel21_gpu, name="k21_n2")
  k30_n2 = tf.norm(kernel30_gpu, name="k30_n2")
  k31_n2 = tf.norm(kernel31_gpu, name="k31_n2")
  k32_n2 = tf.norm(kernel32_gpu, name="k32_n2")
  k33_n2 = tf.norm(kernel33_gpu, name="k33_n2")
  k40_n2 = tf.norm(kernel40_gpu, name="k40_n2")
  k41_n2 = tf.norm(kernel41_gpu, name="k41_n2")
  k42_n2 = tf.norm(kernel42_gpu, name="k42_n2")
  k43_n2 = tf.norm(kernel43_gpu, name="k43_n2")
  k50_n2 = tf.norm(kernel50_gpu, name="k50_n2")
  k51_n2 = tf.norm(kernel51_gpu, name="k51_n2")
  k52_n2 = tf.norm(kernel52_gpu, name="k52_n2")
  k53_n2 = tf.norm(kernel53_gpu, name="k53_n2")
  k6_n2 = tf.norm(kernel6_gpu, name="k6_n2")
  k7_n2 = tf.norm(kernel7_gpu, name="k7_n2")
  k8_n2 = tf.norm(kernel8_gpu, name="k8_n2")

  norms = [k10_n2, k11_n2, k20_n2, k21_n2, k30_n2, k31_n2, k32_n2, k33_n2, 
           k40_n2, k41_n2, k42_n2, k43_n2, k50_n2, k51_n2, k52_n2, k53_n2, 
           k6_n2 , k7_n2 , k8_n2] 

  # Convert end_points_collection into a end_point dict.
  end_points = utils.convert_collection_to_dict(end_points_collection)
  if spatial_squeeze:
    net = array_ops.squeeze(net, [1, 2], name='fc8/squeezed')
    end_points[sc.name + '/fc8'] = net
  return net, norms


def inference_vgg_19(inputs, 
                    num_classes=1000, 
                    for_training=True, 
                    dropout_keep_prob=0.5, 
                    bits_ph=[], 
                    spatial_squeeze=True, 
                    scope='vgg_19'):

  if for_training is True:
    with tf.variable_scope('', 'vgg_19', [inputs]) as sc:
      end_points_collection = sc.original_name_scope + '_end_points'
      w_reg_l2 = None
      with tf.variable_scope('conv1'):
        with tf.device('/cpu:0'):
          kernel10 = tf.get_variable("weights0", [3, 3, 3, 64], regularizer=w_reg_l2, dtype=tf.float32)
          biases10 = tf.get_variable('biases0', [64], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel11 = tf.get_variable("weights1", [3, 3, 64, 64], regularizer=w_reg_l2, dtype=tf.float32)
          biases11 = tf.get_variable('biases1', [64], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv10 = tf.nn.conv2d(inputs, kernel10, strides=[1, 1, 1, 1], padding='SAME', name='conv10')
        pre_activation10 = tf.nn.bias_add(conv10, biases10, name="pre_10")
        relu10 = tf.nn.relu(pre_activation10, name="relu10")
        conv11 = tf.nn.conv2d(relu10, kernel11, strides=[1, 1, 1, 1], padding='SAME', name='conv11')
        pre_activation11 = tf.nn.bias_add(conv11, biases11, name="pre_11")
        relu11 = tf.nn.relu(pre_activation11, name="relu11")
        pool1 = tf.nn.max_pool(relu11, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

      with tf.variable_scope('conv2'):
        with tf.device('/cpu:0'):
          kernel20 = tf.get_variable("weights0", [3, 3, 64, 128], regularizer=w_reg_l2, dtype=tf.float32)
          biases20 = tf.get_variable('biases0', [128], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel21 = tf.get_variable("weights1", [3, 3, 128, 128], regularizer=w_reg_l2, dtype=tf.float32)
          biases21 = tf.get_variable('biases1', [128], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv20 = tf.nn.conv2d(pool1, kernel20, strides=[1, 1, 1, 1], padding='SAME', name='conv20')
        pre_activation20 = tf.nn.bias_add(conv20, biases20, name="pre_20")
        relu20 = tf.nn.relu(pre_activation20, name="relu20")
        conv21 = tf.nn.conv2d(relu20, kernel21, strides=[1, 1, 1, 1], padding='SAME', name='conv21')
        pre_activation21 = tf.nn.bias_add(conv21, biases21, name="pre_21")
        relu21 = tf.nn.relu(pre_activation21, name="relu21")
        pool2 = tf.nn.max_pool(relu21, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

      with tf.variable_scope('conv3'):
        with tf.device('/cpu:0'):
          kernel30 = tf.get_variable("weights0", [3, 3, 128, 256], regularizer=w_reg_l2, dtype=tf.float32)
          biases30 = tf.get_variable('biases0', [256], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel31 = tf.get_variable("weights1", [3, 3, 256, 256], regularizer=w_reg_l2, dtype=tf.float32)
          biases31 = tf.get_variable('biases1', [256], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel32 = tf.get_variable("weights2", [3, 3, 256, 256], regularizer=w_reg_l2, dtype=tf.float32)
          biases32 = tf.get_variable('biases2', [256], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel33 = tf.get_variable("weights3", [3, 3, 256, 256], regularizer=w_reg_l2, dtype=tf.float32)
          biases33 = tf.get_variable('biases3', [256], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv30 = tf.nn.conv2d(pool2, kernel30, strides=[1, 1, 1, 1], padding='SAME', name='conv30')
        pre_activation30 = tf.nn.bias_add(conv30, biases30, name="pre_30")
        relu30 = tf.nn.relu(pre_activation30, name="relu30")
        conv31 = tf.nn.conv2d(relu30, kernel31, strides=[1, 1, 1, 1], padding='SAME', name='conv31')
        pre_activation31 = tf.nn.bias_add(conv31, biases31, name="pre_31")
        relu31 = tf.nn.relu(pre_activation31, name="relu31")
        conv32 = tf.nn.conv2d(relu31, kernel32, strides=[1, 1, 1, 1], padding='SAME', name='conv32')
        pre_activation32 = tf.nn.bias_add(conv32, biases32, name="pre_32")
        relu32 = tf.nn.relu(pre_activation32, name="relu32")
        conv33 = tf.nn.conv2d(relu32, kernel33, strides=[1, 1, 1, 1], padding='SAME', name='conv33')
        pre_activation33 = tf.nn.bias_add(conv33, biases33, name="pre_33")
        relu33 = tf.nn.relu(pre_activation33, name="relu33")
        pool3 = tf.nn.max_pool(relu33, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

      with tf.variable_scope('conv4'):
        with tf.device('/cpu:0'):
          kernel40 = tf.get_variable("weights0", [3, 3, 256, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases40 = tf.get_variable('biases0', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel41 = tf.get_variable("weights1", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases41 = tf.get_variable('biases1', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel42 = tf.get_variable("weights2", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases42 = tf.get_variable('biases2', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel43 = tf.get_variable("weights3", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases43 = tf.get_variable('biases3', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv40 = tf.nn.conv2d(pool3, kernel40, strides=[1, 1, 1, 1], padding='SAME', name='conv40')
        pre_activation40 = tf.nn.bias_add(conv40, biases40, name="pre_40")
        relu40 = tf.nn.relu(pre_activation40, name="relu40")
        conv41 = tf.nn.conv2d(relu40, kernel41, strides=[1, 1, 1, 1], padding='SAME', name='conv41')
        pre_activation41 = tf.nn.bias_add(conv41, biases41, name="pre_41")
        relu41 = tf.nn.relu(pre_activation41, name="relu41")
        conv42 = tf.nn.conv2d(relu41, kernel42, strides=[1, 1, 1, 1], padding='SAME', name='conv42')
        pre_activation42 = tf.nn.bias_add(conv42, biases42, name="pre_42")
        relu42 = tf.nn.relu(pre_activation42, name="relu42")
        conv43 = tf.nn.conv2d(relu42, kernel43, strides=[1, 1, 1, 1], padding='SAME', name='conv43')
        pre_activation43 = tf.nn.bias_add(conv43, biases43, name="pre_43")
        relu43 = tf.nn.relu(pre_activation43, name="relu43")
        pool4 = tf.nn.max_pool(relu43, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

      with tf.variable_scope('conv5'):
        with tf.device('/cpu:0'):
          kernel50 = tf.get_variable("weights0", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases50 = tf.get_variable('biases0', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel51 = tf.get_variable("weights1", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases51 = tf.get_variable('biases1', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel52 = tf.get_variable("weights2", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases52 = tf.get_variable('biases2', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel53 = tf.get_variable("weights3", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases53 = tf.get_variable('biases3', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv50 = tf.nn.conv2d(pool4, kernel50, strides=[1, 1, 1, 1], padding='SAME', name='conv50')
        pre_activation50 = tf.nn.bias_add(conv50, biases50, name="pre_50")
        relu50 = tf.nn.relu(pre_activation50, name="relu50")
        conv51 = tf.nn.conv2d(relu50, kernel51, strides=[1, 1, 1, 1], padding='SAME', name='conv51')
        pre_activation51 = tf.nn.bias_add(conv51, biases51, name="pre_51")
        relu51 = tf.nn.relu(pre_activation51, name="relu51")
        conv52 = tf.nn.conv2d(relu51, kernel52, strides=[1, 1, 1, 1], padding='SAME', name='conv52')
        pre_activation52 = tf.nn.bias_add(conv52, biases52, name="pre_52")
        relu52 = tf.nn.relu(pre_activation52, name="relu52")
        conv53 = tf.nn.conv2d(relu52, kernel53, strides=[1, 1, 1, 1], padding='SAME', name='conv53')
        pre_activation53 = tf.nn.bias_add(conv53, biases53, name="pre_53")
        relu53 = tf.nn.relu(pre_activation53, name="relu53")
        pool5 = tf.nn.max_pool(relu53, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

      with tf.variable_scope('fc6'):
        with tf.device('/cpu:0'):
          kernel6 = tf.get_variable("weights", [7, 7, 512, 4096], regularizer=w_reg_l2, dtype=tf.float32)
          biases6 = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv6 = tf.nn.conv2d(pool5, kernel6, strides=[1, 1, 1, 1], padding='VALID', name='fc6')
        fc6 = tf.nn.bias_add(conv6, biases6)
        relu6 = tf.nn.relu(fc6, name="relu6")
        fc6_dropout = tf.nn.dropout(relu6, 0.5, name="fc6_dropout")

      with tf.variable_scope('fc7'):
        with tf.device('/cpu:0'):
          kernel7 = tf.get_variable("weights", [1, 1, 4096, 4096], regularizer=w_reg_l2, dtype=tf.float32)
          biases7 = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv7 = tf.nn.conv2d(fc6_dropout, kernel7, strides=[1, 1, 1, 1], padding='SAME', name='fc7')
        fc7 = tf.nn.bias_add(conv7, biases7)
        relu7 = tf.nn.relu(fc7, name="relu7")
        fc7_dropout = tf.nn.dropout(relu7, 0.5, name="fc7_dropout")

      with tf.variable_scope('fc8'):
        with tf.device('/cpu:0'):
          kernel8 = tf.get_variable("weights", [1, 1, 4096, num_classes], regularizer=w_reg_l2, dtype=tf.float32)
          biases8 = tf.get_variable('biases', [num_classes], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv8 = tf.nn.conv2d(fc7_dropout, kernel8, strides=[1, 1, 1, 1], padding='SAME', name='fc8')
        net = tf.nn.bias_add(conv8, biases8)

  elif for_training is False:
    with tf.variable_scope('', 'vgg_19', [inputs]) as sc:
      end_points_collection = sc.original_name_scope + '_end_points'
      w_reg_l2 = None
      with tf.variable_scope('conv1'):
        with tf.device('/cpu:0'):
          kernel10 = tf.get_variable("weights0", [3, 3, 3, 64], regularizer=w_reg_l2, dtype=tf.float32)
          biases10 = tf.get_variable('biases0', [64], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel11 = tf.get_variable("weights1", [3, 3, 64, 64], regularizer=w_reg_l2, dtype=tf.float32)
          biases11 = tf.get_variable('biases1', [64], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv10 = tf.nn.conv2d(inputs, kernel10, strides=[1, 1, 1, 1], padding='SAME', name='conv10')
        pre_activation10 = tf.nn.bias_add(conv10, biases10, name="pre_10")
        relu10 = tf.nn.relu(pre_activation10, name="relu10")
        conv11 = tf.nn.conv2d(relu10, kernel11, strides=[1, 1, 1, 1], padding='SAME', name='conv11')
        pre_activation11 = tf.nn.bias_add(conv11, biases11, name="pre_11")
        relu11 = tf.nn.relu(pre_activation11, name="relu11")
        pool1 = tf.nn.max_pool(relu11, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

      with tf.variable_scope('conv2'):
        with tf.device('/cpu:0'):
          kernel20 = tf.get_variable("weights0", [3, 3, 64, 128], regularizer=w_reg_l2, dtype=tf.float32)
          biases20 = tf.get_variable('biases0', [128], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel21 = tf.get_variable("weights1", [3, 3, 128, 128], regularizer=w_reg_l2, dtype=tf.float32)
          biases21 = tf.get_variable('biases1', [128], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv20 = tf.nn.conv2d(pool1, kernel20, strides=[1, 1, 1, 1], padding='SAME', name='conv20')
        pre_activation20 = tf.nn.bias_add(conv20, biases20, name="pre_20")
        relu20 = tf.nn.relu(pre_activation20, name="relu20")
        conv21 = tf.nn.conv2d(relu20, kernel21, strides=[1, 1, 1, 1], padding='SAME', name='conv21')
        pre_activation21 = tf.nn.bias_add(conv21, biases21, name="pre_21")
        relu21 = tf.nn.relu(pre_activation21, name="relu21")
        pool2 = tf.nn.max_pool(relu21, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

      with tf.variable_scope('conv3'):
        with tf.device('/cpu:0'):
          kernel30 = tf.get_variable("weights0", [3, 3, 128, 256], regularizer=w_reg_l2, dtype=tf.float32)
          biases30 = tf.get_variable('biases0', [256], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel31 = tf.get_variable("weights1", [3, 3, 256, 256], regularizer=w_reg_l2, dtype=tf.float32)
          biases31 = tf.get_variable('biases1', [256], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel32 = tf.get_variable("weights2", [3, 3, 256, 256], regularizer=w_reg_l2, dtype=tf.float32)
          biases32 = tf.get_variable('biases2', [256], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel33 = tf.get_variable("weights3", [3, 3, 256, 256], regularizer=w_reg_l2, dtype=tf.float32)
          biases33 = tf.get_variable('biases3', [256], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv30 = tf.nn.conv2d(pool2, kernel30, strides=[1, 1, 1, 1], padding='SAME', name='conv30')
        pre_activation30 = tf.nn.bias_add(conv30, biases30, name="pre_30")
        relu30 = tf.nn.relu(pre_activation30, name="relu30")
        conv31 = tf.nn.conv2d(relu30, kernel31, strides=[1, 1, 1, 1], padding='SAME', name='conv31')
        pre_activation31 = tf.nn.bias_add(conv31, biases31, name="pre_31")
        relu31 = tf.nn.relu(pre_activation31, name="relu31")
        conv32 = tf.nn.conv2d(relu31, kernel32, strides=[1, 1, 1, 1], padding='SAME', name='conv32')
        pre_activation32 = tf.nn.bias_add(conv32, biases32, name="pre_32")
        relu32 = tf.nn.relu(pre_activation32, name="relu32")
        conv33 = tf.nn.conv2d(relu32, kernel33, strides=[1, 1, 1, 1], padding='SAME', name='conv33')
        pre_activation33 = tf.nn.bias_add(conv33, biases33, name="pre_33")
        relu33 = tf.nn.relu(pre_activation33, name="relu33")
        pool3 = tf.nn.max_pool(relu33, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

      with tf.variable_scope('conv4'):
        with tf.device('/cpu:0'):
          kernel40 = tf.get_variable("weights0", [3, 3, 256, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases40 = tf.get_variable('biases0', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel41 = tf.get_variable("weights1", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases41 = tf.get_variable('biases1', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel42 = tf.get_variable("weights2", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases42 = tf.get_variable('biases2', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel43 = tf.get_variable("weights3", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases43 = tf.get_variable('biases3', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv40 = tf.nn.conv2d(pool3, kernel40, strides=[1, 1, 1, 1], padding='SAME', name='conv40')
        pre_activation40 = tf.nn.bias_add(conv40, biases40, name="pre_40")
        relu40 = tf.nn.relu(pre_activation40, name="relu40")
        conv41 = tf.nn.conv2d(relu40, kernel41, strides=[1, 1, 1, 1], padding='SAME', name='conv41')
        pre_activation41 = tf.nn.bias_add(conv41, biases41, name="pre_41")
        relu41 = tf.nn.relu(pre_activation41, name="relu41")
        conv42 = tf.nn.conv2d(relu41, kernel42, strides=[1, 1, 1, 1], padding='SAME', name='conv42')
        pre_activation42 = tf.nn.bias_add(conv42, biases42, name="pre_42")
        relu42 = tf.nn.relu(pre_activation42, name="relu42")
        conv43 = tf.nn.conv2d(relu42, kernel43, strides=[1, 1, 1, 1], padding='SAME', name='conv43')
        pre_activation43 = tf.nn.bias_add(conv43, biases43, name="pre_43")
        relu43 = tf.nn.relu(pre_activation43, name="relu43")
        pool4 = tf.nn.max_pool(relu43, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

      with tf.variable_scope('conv5'):
        with tf.device('/cpu:0'):
          kernel50 = tf.get_variable("weights0", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases50 = tf.get_variable('biases0', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel51 = tf.get_variable("weights1", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases51 = tf.get_variable('biases1', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel52 = tf.get_variable("weights2", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases52 = tf.get_variable('biases2', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel53 = tf.get_variable("weights3", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases53 = tf.get_variable('biases3', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv50 = tf.nn.conv2d(pool4, kernel50, strides=[1, 1, 1, 1], padding='SAME', name='conv50')
        pre_activation50 = tf.nn.bias_add(conv50, biases50, name="pre_50")
        relu50 = tf.nn.relu(pre_activation50, name="relu50")
        conv51 = tf.nn.conv2d(relu50, kernel51, strides=[1, 1, 1, 1], padding='SAME', name='conv51')
        pre_activation51 = tf.nn.bias_add(conv51, biases51, name="pre_51")
        relu51 = tf.nn.relu(pre_activation51, name="relu51")
        conv52 = tf.nn.conv2d(relu51, kernel52, strides=[1, 1, 1, 1], padding='SAME', name='conv52')
        pre_activation52 = tf.nn.bias_add(conv52, biases52, name="pre_52")
        relu52 = tf.nn.relu(pre_activation52, name="relu52")
        conv53 = tf.nn.conv2d(relu52, kernel53, strides=[1, 1, 1, 1], padding='SAME', name='conv53')
        pre_activation53 = tf.nn.bias_add(conv53, biases53, name="pre_53")
        relu53 = tf.nn.relu(pre_activation53, name="relu53")
        pool5 = tf.nn.max_pool(relu53, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

      with tf.variable_scope('fc6'):
        with tf.device('/cpu:0'):
          kernel6 = tf.get_variable("weights", [7, 7, 512, 4096], regularizer=w_reg_l2, dtype=tf.float32)
          biases6 = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv6 = tf.nn.conv2d(pool5, kernel6, strides=[1, 1, 1, 1], padding='VALID', name='fc6')
        fc6 = tf.nn.bias_add(conv6, biases6)
        relu6 = tf.nn.relu(fc6, name="relu6")

      with tf.variable_scope('fc7'):
        with tf.device('/cpu:0'):
          kernel7 = tf.get_variable("weights", [1, 1, 4096, 4096], regularizer=w_reg_l2, dtype=tf.float32)
          biases7 = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv7 = tf.nn.conv2d(relu6, kernel7, strides=[1, 1, 1, 1], padding='SAME', name='fc7')
        fc7 = tf.nn.bias_add(conv7, biases7)
        relu7 = tf.nn.relu(fc7, name="relu7")

      with tf.variable_scope('fc8'):
        with tf.device('/cpu:0'):
          kernel8 = tf.get_variable("weights", [1, 1, 4096, num_classes], regularizer=w_reg_l2, dtype=tf.float32)
          biases8 = tf.get_variable('biases', [num_classes], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv8 = tf.nn.conv2d(relu7, kernel8, strides=[1, 1, 1, 1], padding='SAME', name='fc8')
        net = tf.nn.bias_add(conv8, biases8)

  wd10 = tf.multiply(tf.nn.l2_loss(kernel10), FLAGS.L2_reg, name='wd10')
  wd11 = tf.multiply(tf.nn.l2_loss(kernel11), FLAGS.L2_reg, name='wd11')
  wd20 = tf.multiply(tf.nn.l2_loss(kernel20), FLAGS.L2_reg, name='wd20')
  wd21 = tf.multiply(tf.nn.l2_loss(kernel21), FLAGS.L2_reg, name='wd21')
  wd30 = tf.multiply(tf.nn.l2_loss(kernel30), FLAGS.L2_reg, name='wd30')
  wd31 = tf.multiply(tf.nn.l2_loss(kernel31), FLAGS.L2_reg, name='wd31')
  wd32 = tf.multiply(tf.nn.l2_loss(kernel32), FLAGS.L2_reg, name='wd32')
  wd33 = tf.multiply(tf.nn.l2_loss(kernel33), FLAGS.L2_reg, name='wd33')
  wd40 = tf.multiply(tf.nn.l2_loss(kernel40), FLAGS.L2_reg, name='wd40')
  wd41 = tf.multiply(tf.nn.l2_loss(kernel41), FLAGS.L2_reg, name='wd41')
  wd42 = tf.multiply(tf.nn.l2_loss(kernel42), FLAGS.L2_reg, name='wd42')
  wd43 = tf.multiply(tf.nn.l2_loss(kernel43), FLAGS.L2_reg, name='wd43')
  wd50 = tf.multiply(tf.nn.l2_loss(kernel50), FLAGS.L2_reg, name='wd50')
  wd51 = tf.multiply(tf.nn.l2_loss(kernel51), FLAGS.L2_reg, name='wd51')
  wd52 = tf.multiply(tf.nn.l2_loss(kernel52), FLAGS.L2_reg, name='wd52')
  wd53 = tf.multiply(tf.nn.l2_loss(kernel53), FLAGS.L2_reg, name='wd53')
  wd6 = tf.multiply(tf.nn.l2_loss(kernel6), FLAGS.L2_reg, name='wd6')
  wd7 = tf.multiply(tf.nn.l2_loss(kernel7), FLAGS.L2_reg, name='wd7')
  wd8 = tf.multiply(tf.nn.l2_loss(kernel8), FLAGS.L2_reg, name='wd8')

  tf.add_to_collection(scope+'losses', wd10)
  tf.add_to_collection(scope+'losses', wd11)
  tf.add_to_collection(scope+'losses', wd20)
  tf.add_to_collection(scope+'losses', wd21)
  tf.add_to_collection(scope+'losses', wd30)
  tf.add_to_collection(scope+'losses', wd31)
  tf.add_to_collection(scope+'losses', wd32)
  tf.add_to_collection(scope+'losses', wd33)
  tf.add_to_collection(scope+'losses', wd40)
  tf.add_to_collection(scope+'losses', wd41)
  tf.add_to_collection(scope+'losses', wd42)
  tf.add_to_collection(scope+'losses', wd43)
  tf.add_to_collection(scope+'losses', wd50)
  tf.add_to_collection(scope+'losses', wd51)
  tf.add_to_collection(scope+'losses', wd52)
  tf.add_to_collection(scope+'losses', wd53)
  tf.add_to_collection(scope+'losses', wd6)
  tf.add_to_collection(scope+'losses', wd7)
  tf.add_to_collection(scope+'losses', wd8)

  k10_n2 = tf.norm(kernel10, name="k10_n2")
  k11_n2 = tf.norm(kernel11, name="k11_n2")
  k20_n2 = tf.norm(kernel20, name="k20_n2")
  k21_n2 = tf.norm(kernel21, name="k21_n2")
  k30_n2 = tf.norm(kernel30, name="k30_n2")
  k31_n2 = tf.norm(kernel31, name="k31_n2")
  k32_n2 = tf.norm(kernel32, name="k32_n2")
  k33_n2 = tf.norm(kernel33, name="k33_n2")
  k40_n2 = tf.norm(kernel40, name="k40_n2")
  k41_n2 = tf.norm(kernel41, name="k41_n2")
  k42_n2 = tf.norm(kernel42, name="k42_n2")
  k43_n2 = tf.norm(kernel43, name="k43_n2")
  k50_n2 = tf.norm(kernel50, name="k50_n2")
  k51_n2 = tf.norm(kernel51, name="k51_n2")
  k52_n2 = tf.norm(kernel52, name="k52_n2")
  k53_n2 = tf.norm(kernel53, name="k53_n2")
  k6_n2 = tf.norm(kernel6, name="k6_n2")
  k7_n2 = tf.norm(kernel7, name="k7_n2")
  k8_n2 = tf.norm(kernel8, name="k8_n2")

  norms = [k10_n2, k11_n2, k20_n2, k21_n2, k30_n2, k31_n2, k32_n2, k33_n2, 
           k40_n2, k41_n2, k42_n2, k43_n2, k50_n2, k51_n2, k52_n2, k53_n2, 
           k6_n2 , k7_n2 , k8_n2] 

  # Convert end_points_collection into a end_point dict.
  end_points = utils.convert_collection_to_dict(end_points_collection)
  if spatial_squeeze:
    net = array_ops.squeeze(net, [1, 2], name='fc8/squeezed')
    end_points[sc.name + '/fc8'] = net
  return net, norms


def inference_vgg_a_bitpack(inputs, 
                    num_classes=1000, 
                    for_training=True, 
                    dropout_keep_prob=0.5, 
                    bits_ph=[], 
                    spatial_squeeze=True, 
                    scope='vgg_a'):

  if len(bits_ph) != 11:
    raise Exception("bits_ph != 11")
  if for_training is True:
    with tf.variable_scope('', 'vgg_a', [inputs]) as sc:
      end_points_collection = sc.original_name_scope + '_end_points'
      w_reg_l2 = None
      with tf.variable_scope('conv1'):
        with tf.device('/cpu:0'):
          kernel1 = tf.get_variable("weights", [3, 3, 3, 64], regularizer=w_reg_l2, dtype=tf.float32)
          ckernel1 = bit_packer.bit_pack_cpu_avx(kernel1, bits_ph[0], name="bc_conv1")
          biases1 = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        kernel1_gpu = bit_packer.bit_unpack_gpu(ckernel1, bits_ph[0], tf.size(kernel1), tf.shape(kernel1), name="gpu_conv1")
        kernel1_gpu = tf.reshape(kernel1_gpu, tf.shape(kernel1))
        conv1 = tf.nn.conv2d(inputs, kernel1_gpu, strides=[1, 1, 1, 1], padding='SAME', name='conv1')
        pre_activation1 = tf.nn.bias_add(conv1, biases1, name="pre_1")
        relu1 = tf.nn.relu(pre_activation1, name="relu1")
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
      with tf.variable_scope('conv2'):
        with tf.device('/cpu:0'):
          kernel2 = tf.get_variable("weights", [3, 3, 64, 128], regularizer=w_reg_l2, dtype=tf.float32)
          ckernel2 = bit_packer.bit_pack_cpu_avx(kernel2, bits_ph[1], name="bc_conv2")
          biases2 = tf.get_variable('biases', [128], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        kernel2_gpu = bit_packer.bit_unpack_gpu(ckernel2, bits_ph[1], tf.size(kernel2), tf.shape(kernel2), name="gpu_conv2")
        kernel2_gpu = tf.reshape(kernel2_gpu, tf.shape(kernel2))
        conv2 = tf.nn.conv2d(pool1, kernel2_gpu, strides=[1, 1, 1, 1], padding='SAME', name='conv2')
        pre_activation2 = tf.nn.bias_add(conv2, biases2, name="pre_2")
        relu2 = tf.nn.relu(pre_activation2, name="relu2")
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
      with tf.variable_scope('conv3'):
        with tf.device('/cpu:0'):
          kernel30 = tf.get_variable("weights0", [3, 3, 128, 256], regularizer=w_reg_l2, dtype=tf.float32)
          biases30 = tf.get_variable('biases0', [256], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel31 = tf.get_variable("weights1", [3, 3, 256, 256], regularizer=w_reg_l2, dtype=tf.float32)
          biases31 = tf.get_variable('biases1', [256], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          ckernel30 = bit_packer.bit_pack_cpu_avx(kernel30, bits_ph[2], name="bc_conv30")
          ckernel31 = bit_packer.bit_pack_cpu_avx(kernel31, bits_ph[3], name="bc_conv31")
        kernel30_gpu = bit_packer.bit_unpack_gpu(ckernel30, bits_ph[2], tf.size(kernel30), tf.shape(kernel30), name="gpu_conv30")
        kernel30_gpu = tf.reshape(kernel30_gpu, tf.shape(kernel30))
        kernel31_gpu = bit_packer.bit_unpack_gpu(ckernel31, bits_ph[3], tf.size(kernel31), tf.shape(kernel31), name="gpu_conv31")
        kernel31_gpu = tf.reshape(kernel31_gpu, tf.shape(kernel31))
        conv30 = tf.nn.conv2d(pool2, kernel30_gpu, strides=[1, 1, 1, 1], padding='SAME', name='conv30')
        pre_activation30 = tf.nn.bias_add(conv30, biases30, name="pre_30")
        relu30 = tf.nn.relu(pre_activation30, name="relu30")
        conv31 = tf.nn.conv2d(relu30, kernel31_gpu, strides=[1, 1, 1, 1], padding='SAME', name='conv31')
        pre_activation31 = tf.nn.bias_add(conv31, biases31, name="pre_31")
        relu31 = tf.nn.relu(pre_activation31, name="relu31")
        pool3 = tf.nn.max_pool(relu31, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
      with tf.variable_scope('conv4'):
        with tf.device('/cpu:0'):
          kernel40 = tf.get_variable("weights0", [3, 3, 256, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases40 = tf.get_variable('biases0', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel41 = tf.get_variable("weights1", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases41 = tf.get_variable('biases1', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          ckernel40 = bit_packer.bit_pack_cpu_avx(kernel40, bits_ph[4], name="bc_conv40")
          ckernel41 = bit_packer.bit_pack_cpu_avx(kernel41, bits_ph[5], name="bc_conv41")
        kernel40_gpu = bit_packer.bit_unpack_gpu(ckernel40, bits_ph[4], tf.size(kernel40), tf.shape(kernel40), name="gpu_conv40")
        kernel40_gpu = tf.reshape(kernel40_gpu, tf.shape(kernel40))
        kernel41_gpu = bit_packer.bit_unpack_gpu(ckernel41, bits_ph[5], tf.size(kernel41), tf.shape(kernel41), name="gpu_conv41")
        kernel41_gpu = tf.reshape(kernel41_gpu, tf.shape(kernel41))
        conv40 = tf.nn.conv2d(pool3, kernel40_gpu, strides=[1, 1, 1, 1], padding='SAME', name='conv40')
        pre_activation40 = tf.nn.bias_add(conv40, biases40, name="pre_40")
        relu40 = tf.nn.relu(pre_activation40, name="relu40")
        conv41 = tf.nn.conv2d(relu40, kernel41_gpu, strides=[1, 1, 1, 1], padding='SAME', name='conv41')
        pre_activation41 = tf.nn.bias_add(conv41, biases41, name="pre_41")
        relu41 = tf.nn.relu(pre_activation41, name="relu41")
        pool4 = tf.nn.max_pool(relu41, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
      with tf.variable_scope('conv5'):
        with tf.device('/cpu:0'):
          kernel50 = tf.get_variable("weights0", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases50 = tf.get_variable('biases0', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel51 = tf.get_variable("weights1", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases51 = tf.get_variable('biases1', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          ckernel50 = bit_packer.bit_pack_cpu_avx(kernel50, bits_ph[6], name="bc_conv50")
          ckernel51 = bit_packer.bit_pack_cpu_avx(kernel51, bits_ph[7], name="bc_conv51")
        kernel50_gpu = bit_packer.bit_unpack_gpu(ckernel50, bits_ph[6], tf.size(kernel50), tf.shape(kernel50), name="gpu_conv50")
        kernel50_gpu = tf.reshape(kernel50_gpu, tf.shape(kernel50))
        kernel51_gpu = bit_packer.bit_unpack_gpu(ckernel51, bits_ph[7], tf.size(kernel51), tf.shape(kernel51), name="gpu_conv51")
        kernel51_gpu = tf.reshape(kernel51_gpu, tf.shape(kernel51))
        conv50 = tf.nn.conv2d(pool4, kernel50_gpu, strides=[1, 1, 1, 1], padding='SAME', name='conv50')
        pre_activation50 = tf.nn.bias_add(conv50, biases50, name="pre_50")
        relu50 = tf.nn.relu(pre_activation50, name="relu50")
        conv51 = tf.nn.conv2d(relu50, kernel51_gpu, strides=[1, 1, 1, 1], padding='SAME', name='conv51')
        pre_activation51 = tf.nn.bias_add(conv51, biases51, name="pre_51")
        relu51 = tf.nn.relu(pre_activation51, name="relu51")
        pool5 = tf.nn.max_pool(relu51, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
      with tf.variable_scope('fc6'):
        with tf.device('/cpu:0'):
          kernel6 = tf.get_variable("weights", [7, 7, 512, 4096], regularizer=w_reg_l2, dtype=tf.float32)
          ckernel6 = bit_packer.bit_pack_cpu_avx_omp(kernel6, bits_ph[8], name="bc_conv6")
          biases6 = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        kernel6_gpu = bit_packer.bit_unpack_gpu(ckernel6, bits_ph[8], tf.size(kernel6), tf.shape(kernel6), name="gpu_fc6")
        kernel6_gpu = tf.reshape(kernel6_gpu, tf.shape(kernel6))
        conv6 = tf.nn.conv2d(pool5, kernel6_gpu, strides=[1, 1, 1, 1], padding='VALID', name='fc6')
        fc6 = tf.nn.bias_add(conv6, biases6)
        relu6 = tf.nn.relu(fc6, name="relu6")
        fc6_dropout = tf.nn.dropout(relu6, 0.5, name="fc6_dropout")
      with tf.variable_scope('fc7'):
        with tf.device('/cpu:0'):
          kernel7 = tf.get_variable("weights", [1, 1, 4096, 4096], regularizer=w_reg_l2, dtype=tf.float32)
          ckernel7 = bit_packer.bit_pack_cpu_avx_omp(kernel7, bits_ph[9], name="bc_conv7")
          biases7 = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        kernel7_gpu = bit_packer.bit_unpack_gpu(ckernel7, bits_ph[9], tf.size(kernel7), tf.shape(kernel7), name="gpu_fc7")
        kernel7_gpu = tf.reshape(kernel7_gpu, tf.shape(kernel7))
        conv7 = tf.nn.conv2d(fc6_dropout, kernel7_gpu, strides=[1, 1, 1, 1], padding='SAME', name='fc7')
        fc7 = tf.nn.bias_add(conv7, biases7)
        relu7 = tf.nn.relu(fc7, name="relu7")
        fc7_dropout = tf.nn.dropout(relu7, 0.5, name="fc7_dropout")
      with tf.variable_scope('fc8'):
        with tf.device('/cpu:0'):
          kernel8 = tf.get_variable("weights", [1, 1, 4096, num_classes], regularizer=w_reg_l2, dtype=tf.float32)
          ckernel8 = bit_packer.bit_pack_cpu_avx(kernel8, bits_ph[10], name="bc_conv8")
          biases8 = tf.get_variable('biases', [num_classes], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        kernel8_gpu = bit_packer.bit_unpack_gpu(ckernel8, bits_ph[10], tf.size(kernel8), tf.shape(kernel8), name="gpu_fc8")
        kernel8_gpu = tf.reshape(kernel8_gpu, tf.shape(kernel8))
        conv8 = tf.nn.conv2d(fc7_dropout, kernel8_gpu, strides=[1, 1, 1, 1], padding='SAME', name='fc8')
        net = tf.nn.bias_add(conv8, biases8)

  wd1 = tf.multiply(tf.nn.l2_loss(kernel1_gpu), FLAGS.L2_reg, name='wd1')
  wd2 = tf.multiply(tf.nn.l2_loss(kernel2_gpu), FLAGS.L2_reg, name='wd2')
  wd30 = tf.multiply(tf.nn.l2_loss(kernel30_gpu), FLAGS.L2_reg, name='wd30')
  wd31 = tf.multiply(tf.nn.l2_loss(kernel31_gpu), FLAGS.L2_reg, name='wd31')
  wd40 = tf.multiply(tf.nn.l2_loss(kernel40_gpu), FLAGS.L2_reg, name='wd40')
  wd41 = tf.multiply(tf.nn.l2_loss(kernel41_gpu), FLAGS.L2_reg, name='wd41')
  wd50 = tf.multiply(tf.nn.l2_loss(kernel50_gpu), FLAGS.L2_reg, name='wd50')
  wd51 = tf.multiply(tf.nn.l2_loss(kernel51_gpu), FLAGS.L2_reg, name='wd51')
  wd6 = tf.multiply(tf.nn.l2_loss(kernel6_gpu), FLAGS.L2_reg, name='wd6')
  wd7 = tf.multiply(tf.nn.l2_loss(kernel7_gpu), FLAGS.L2_reg, name='wd7')
  wd8 = tf.multiply(tf.nn.l2_loss(kernel8_gpu), FLAGS.L2_reg, name='wd8')

  tf.add_to_collection(scope+'losses', wd1)
  tf.add_to_collection(scope+'losses', wd2)
  tf.add_to_collection(scope+'losses', wd30)
  tf.add_to_collection(scope+'losses', wd31)
  tf.add_to_collection(scope+'losses', wd40)
  tf.add_to_collection(scope+'losses', wd41)
  tf.add_to_collection(scope+'losses', wd50)
  tf.add_to_collection(scope+'losses', wd51)
  tf.add_to_collection(scope+'losses', wd6)
  tf.add_to_collection(scope+'losses', wd7)
  tf.add_to_collection(scope+'losses', wd8)

  k1_n2 = tf.norm(kernel1_gpu, name="k1_n2")
  k2_n2 = tf.norm(kernel2_gpu, name="k2_n2")
  k30_n2 = tf.norm(kernel30_gpu, name="k30_n2")
  k31_n2 = tf.norm(kernel31_gpu, name="k31_n2")
  k40_n2 = tf.norm(kernel40_gpu, name="k40_n2")
  k41_n2 = tf.norm(kernel41_gpu, name="k41_n2")
  k50_n2 = tf.norm(kernel50_gpu, name="k50_n2")
  k51_n2 = tf.norm(kernel51_gpu, name="k51_n2")
  k6_n2 = tf.norm(kernel6_gpu, name="k6_n2")
  k7_n2 = tf.norm(kernel7_gpu, name="k7_n2")
  k8_n2 = tf.norm(kernel8_gpu, name="k8_n2")
  norms = [k1_n2, k2_n2, k30_n2, k31_n2, k40_n2, k41_n2, k50_n2, k51_n2, k6_n2, k7_n2, k8_n2] 

  # Convert end_points_collection into a end_point dict.
  end_points = utils.convert_collection_to_dict(end_points_collection)
  if spatial_squeeze:
    net = array_ops.squeeze(net, [1, 2], name='fc8/squeezed')
    end_points[sc.name + '/fc8'] = net
  return net, norms


def inference_vgg_a(inputs, 
                    num_classes=1000, 
                    for_training=True, 
                    dropout_keep_prob=0.5, 
                    bits_ph=[], 
                    spatial_squeeze=True, 
                    scope='vgg_a'):

  if for_training is True:
    with tf.variable_scope('', 'vgg_a', [inputs]) as sc:
      end_points_collection = sc.original_name_scope + '_end_points'
      w_reg_l2 = None
      with tf.variable_scope('conv1'):
        with tf.device('/cpu:0'):
          kernel1 = tf.get_variable("weights", [3, 3, 3, 64], regularizer=w_reg_l2, dtype=tf.float32)
          biases1 = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv1 = tf.nn.conv2d(inputs, kernel1, strides=[1, 1, 1, 1], padding='SAME', name='conv1')
        pre_activation1 = tf.nn.bias_add(conv1, biases1, name="pre_1")
        relu1 = tf.nn.relu(pre_activation1, name="relu1")
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
      with tf.variable_scope('conv2'):
        with tf.device('/cpu:0'):
          kernel2 = tf.get_variable("weights", [3, 3, 64, 128], regularizer=w_reg_l2, dtype=tf.float32)
          biases2 = tf.get_variable('biases', [128], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv2 = tf.nn.conv2d(pool1, kernel2, strides=[1, 1, 1, 1], padding='SAME', name='conv2')
        pre_activation2 = tf.nn.bias_add(conv2, biases2, name="pre_2")
        relu2 = tf.nn.relu(pre_activation2, name="relu2")
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
      with tf.variable_scope('conv3'):
        with tf.device('/cpu:0'):
          kernel30 = tf.get_variable("weights0", [3, 3, 128, 256], regularizer=w_reg_l2, dtype=tf.float32)
          biases30 = tf.get_variable('biases0', [256], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel31 = tf.get_variable("weights1", [3, 3, 256, 256], regularizer=w_reg_l2, dtype=tf.float32)
          biases31 = tf.get_variable('biases1', [256], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv30 = tf.nn.conv2d(pool2, kernel30, strides=[1, 1, 1, 1], padding='SAME', name='conv30')
        pre_activation30 = tf.nn.bias_add(conv30, biases30, name="pre_30")
        relu30 = tf.nn.relu(pre_activation30, name="relu30")
        conv31 = tf.nn.conv2d(relu30, kernel31, strides=[1, 1, 1, 1], padding='SAME', name='conv31')
        pre_activation31 = tf.nn.bias_add(conv31, biases31, name="pre_31")
        relu31 = tf.nn.relu(pre_activation31, name="relu31")
        pool3 = tf.nn.max_pool(relu31, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
      with tf.variable_scope('conv4'):
        with tf.device('/cpu:0'):
          kernel40 = tf.get_variable("weights0", [3, 3, 256, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases40 = tf.get_variable('biases0', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel41 = tf.get_variable("weights1", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases41 = tf.get_variable('biases1', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv40 = tf.nn.conv2d(pool3, kernel40, strides=[1, 1, 1, 1], padding='SAME', name='conv40')
        pre_activation40 = tf.nn.bias_add(conv40, biases40, name="pre_40")
        relu40 = tf.nn.relu(pre_activation40, name="relu40")
        conv41 = tf.nn.conv2d(relu40, kernel41, strides=[1, 1, 1, 1], padding='SAME', name='conv41')
        pre_activation41 = tf.nn.bias_add(conv41, biases41, name="pre_41")
        relu41 = tf.nn.relu(pre_activation41, name="relu41")
        pool4 = tf.nn.max_pool(relu41, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
      with tf.variable_scope('conv5'):
        with tf.device('/cpu:0'):
          kernel50 = tf.get_variable("weights0", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases50 = tf.get_variable('biases0', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel51 = tf.get_variable("weights1", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases51 = tf.get_variable('biases1', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv50 = tf.nn.conv2d(pool4, kernel50, strides=[1, 1, 1, 1], padding='SAME', name='conv50')
        pre_activation50 = tf.nn.bias_add(conv50, biases50, name="pre_50")
        relu50 = tf.nn.relu(pre_activation50, name="relu50")
        conv51 = tf.nn.conv2d(relu50, kernel51, strides=[1, 1, 1, 1], padding='SAME', name='conv51')
        pre_activation51 = tf.nn.bias_add(conv51, biases51, name="pre_51")
        relu51 = tf.nn.relu(pre_activation51, name="relu51")
        pool5 = tf.nn.max_pool(relu51, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
      with tf.variable_scope('fc6'):
        with tf.device('/cpu:0'):
          kernel6 = tf.get_variable("weights", [7, 7, 512, 4096], regularizer=w_reg_l2, dtype=tf.float32)
          biases6 = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv6 = tf.nn.conv2d(pool5, kernel6, strides=[1, 1, 1, 1], padding='VALID', name='fc6')
        fc6 = tf.nn.bias_add(conv6, biases6)
        relu6 = tf.nn.relu(fc6, name="relu6")
        fc6_dropout = tf.nn.dropout(relu6, 0.5, name="fc6_dropout")
      with tf.variable_scope('fc7'):
        with tf.device('/cpu:0'):
          kernel7 = tf.get_variable("weights", [1, 1, 4096, 4096], regularizer=w_reg_l2, dtype=tf.float32)
          biases7 = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv7 = tf.nn.conv2d(fc6_dropout, kernel7, strides=[1, 1, 1, 1], padding='SAME', name='fc7')
        fc7 = tf.nn.bias_add(conv7, biases7)
        relu7 = tf.nn.relu(fc7, name="relu7")
        fc7_dropout = tf.nn.dropout(relu7, 0.5, name="fc7_dropout")
      with tf.variable_scope('fc8'):
        with tf.device('/cpu:0'):
          kernel8 = tf.get_variable("weights", [1, 1, 4096, num_classes], regularizer=w_reg_l2, dtype=tf.float32)
          biases8 = tf.get_variable('biases', [num_classes], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv8 = tf.nn.conv2d(fc7_dropout, kernel8, strides=[1, 1, 1, 1], padding='SAME', name='fc8')
        net = tf.nn.bias_add(conv8, biases8)
  elif for_training is False:
    with tf.variable_scope('', 'vgg_a', [inputs]) as sc:
      end_points_collection = sc.original_name_scope + '_end_points'
      w_reg_l2 = None
      with tf.variable_scope('conv1'):
        with tf.device('/cpu:0'):
          kernel1 = tf.get_variable("weights", [3, 3, 3, 64], regularizer=w_reg_l2, dtype=tf.float32)
          biases1 = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv1 = tf.nn.conv2d(inputs, kernel1, strides=[1, 1, 1, 1], padding='SAME', name='conv1')
        pre_activation1 = tf.nn.bias_add(conv1, biases1, name="pre_1")
        relu1 = tf.nn.relu(pre_activation1, name="relu1")
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
      with tf.variable_scope('conv2'):
        with tf.device('/cpu:0'):
          kernel2 = tf.get_variable("weights", [3, 3, 64, 128], regularizer=w_reg_l2, dtype=tf.float32)
          biases2 = tf.get_variable('biases', [128], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv2 = tf.nn.conv2d(pool1, kernel2, strides=[1, 1, 1, 1], padding='SAME', name='conv2')
        pre_activation2 = tf.nn.bias_add(conv2, biases2, name="pre_2")
        relu2 = tf.nn.relu(pre_activation2, name="relu2")
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
      with tf.variable_scope('conv3'):
        with tf.device('/cpu:0'):
          kernel30 = tf.get_variable("weights0", [3, 3, 128, 256], regularizer=w_reg_l2, dtype=tf.float32)
          biases30 = tf.get_variable('biases0', [256], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel31 = tf.get_variable("weights1", [3, 3, 256, 256], regularizer=w_reg_l2, dtype=tf.float32)
          biases31 = tf.get_variable('biases1', [256], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv30 = tf.nn.conv2d(pool2, kernel30, strides=[1, 1, 1, 1], padding='SAME', name='conv30')
        pre_activation30 = tf.nn.bias_add(conv30, biases30, name="pre_30")
        relu30 = tf.nn.relu(pre_activation30, name="relu30")
        conv31 = tf.nn.conv2d(relu30, kernel31, strides=[1, 1, 1, 1], padding='SAME', name='conv31')
        pre_activation31 = tf.nn.bias_add(conv31, biases31, name="pre_31")
        relu31 = tf.nn.relu(pre_activation31, name="relu31")
        pool3 = tf.nn.max_pool(relu31, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
      with tf.variable_scope('conv4'):
        with tf.device('/cpu:0'):
          kernel40 = tf.get_variable("weights0", [3, 3, 256, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases40 = tf.get_variable('biases0', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel41 = tf.get_variable("weights1", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases41 = tf.get_variable('biases1', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv40 = tf.nn.conv2d(pool3, kernel40, strides=[1, 1, 1, 1], padding='SAME', name='conv40')
        pre_activation40 = tf.nn.bias_add(conv40, biases40, name="pre_40")
        relu40 = tf.nn.relu(pre_activation40, name="relu40")
        conv41 = tf.nn.conv2d(relu40, kernel41, strides=[1, 1, 1, 1], padding='SAME', name='conv41')
        pre_activation41 = tf.nn.bias_add(conv41, biases41, name="pre_41")
        relu41 = tf.nn.relu(pre_activation41, name="relu41")
        pool4 = tf.nn.max_pool(relu41, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
      with tf.variable_scope('conv5'):
        with tf.device('/cpu:0'):
          kernel50 = tf.get_variable("weights0", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases50 = tf.get_variable('biases0', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
          kernel51 = tf.get_variable("weights1", [3, 3, 512, 512], regularizer=w_reg_l2, dtype=tf.float32)
          biases51 = tf.get_variable('biases1', [512], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv50 = tf.nn.conv2d(pool4, kernel50, strides=[1, 1, 1, 1], padding='SAME', name='conv50')
        pre_activation50 = tf.nn.bias_add(conv50, biases50, name="pre_50")
        relu50 = tf.nn.relu(pre_activation50, name="relu50")
        conv51 = tf.nn.conv2d(relu50, kernel51, strides=[1, 1, 1, 1], padding='SAME', name='conv51')
        pre_activation51 = tf.nn.bias_add(conv51, biases51, name="pre_51")
        relu51 = tf.nn.relu(pre_activation51, name="relu51")
        pool5 = tf.nn.max_pool(relu51, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
      with tf.variable_scope('fc6'):
        with tf.device('/cpu:0'):
          kernel6 = tf.get_variable("weights", [7, 7, 512, 4096], regularizer=w_reg_l2, dtype=tf.float32)
          biases6 = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv6 = tf.nn.conv2d(pool5, kernel6, strides=[1, 1, 1, 1], padding='VALID', name='fc6')
        fc6 = tf.nn.bias_add(conv6, biases6)
        relu6 = tf.nn.relu(fc6, name="relu6")
        #fc6_dropout = tf.nn.dropout(relu6, 0.5, name="fc6_dropout")
      with tf.variable_scope('fc7'):
        with tf.device('/cpu:0'):
          kernel7 = tf.get_variable("weights", [1, 1, 4096, 4096], regularizer=w_reg_l2, dtype=tf.float32)
          biases7 = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv7 = tf.nn.conv2d(relu6, kernel7, strides=[1, 1, 1, 1], padding='SAME', name='fc7')
        fc7 = tf.nn.bias_add(conv7, biases7)
        relu7 = tf.nn.relu(fc7, name="relu7")
      with tf.variable_scope('fc8'):
        with tf.device('/cpu:0'):
          kernel8 = tf.get_variable("weights", [1, 1, 4096, num_classes], regularizer=w_reg_l2, dtype=tf.float32)
          biases8 = tf.get_variable('biases', [num_classes], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        conv8 = tf.nn.conv2d(relu7, kernel8, strides=[1, 1, 1, 1], padding='SAME', name='fc8')
        net = tf.nn.bias_add(conv8, biases8)

  wd1 = tf.multiply(tf.nn.l2_loss(kernel1), FLAGS.L2_reg, name='wd1')
  wd2 = tf.multiply(tf.nn.l2_loss(kernel2), FLAGS.L2_reg, name='wd2')
  wd30 = tf.multiply(tf.nn.l2_loss(kernel30), FLAGS.L2_reg, name='wd30')
  wd31 = tf.multiply(tf.nn.l2_loss(kernel31), FLAGS.L2_reg, name='wd31')
  wd40 = tf.multiply(tf.nn.l2_loss(kernel40), FLAGS.L2_reg, name='wd40')
  wd41 = tf.multiply(tf.nn.l2_loss(kernel41), FLAGS.L2_reg, name='wd41')
  wd50 = tf.multiply(tf.nn.l2_loss(kernel50), FLAGS.L2_reg, name='wd50')
  wd51 = tf.multiply(tf.nn.l2_loss(kernel51), FLAGS.L2_reg, name='wd51')
  wd6 = tf.multiply(tf.nn.l2_loss(kernel6), FLAGS.L2_reg, name='wd6')
  wd7 = tf.multiply(tf.nn.l2_loss(kernel7), FLAGS.L2_reg, name='wd7')
  wd8 = tf.multiply(tf.nn.l2_loss(kernel8), FLAGS.L2_reg, name='wd8')

  tf.add_to_collection(scope+'losses', wd1)
  tf.add_to_collection(scope+'losses', wd2)
  tf.add_to_collection(scope+'losses', wd30)
  tf.add_to_collection(scope+'losses', wd31)
  tf.add_to_collection(scope+'losses', wd40)
  tf.add_to_collection(scope+'losses', wd41)
  tf.add_to_collection(scope+'losses', wd50)
  tf.add_to_collection(scope+'losses', wd51)
  tf.add_to_collection(scope+'losses', wd6)
  tf.add_to_collection(scope+'losses', wd7)
  tf.add_to_collection(scope+'losses', wd8)

  k1_n2 = tf.norm(kernel1, name="k1_n2")
  k2_n2 = tf.norm(kernel2, name="k2_n2")
  k30_n2 = tf.norm(kernel30, name="k30_n2")
  k31_n2 = tf.norm(kernel31, name="k31_n2")
  k40_n2 = tf.norm(kernel40, name="k40_n2")
  k41_n2 = tf.norm(kernel41, name="k41_n2")
  k50_n2 = tf.norm(kernel50, name="k50_n2")
  k51_n2 = tf.norm(kernel51, name="k51_n2")
  k6_n2 = tf.norm(kernel6, name="k6_n2")
  k7_n2 = tf.norm(kernel7, name="k7_n2")
  k8_n2 = tf.norm(kernel8, name="k8_n2")
  norms = [k1_n2, k2_n2, k30_n2, k31_n2, k40_n2, k41_n2, k50_n2, k51_n2, k6_n2, k7_n2, k8_n2] 

  # Convert end_points_collection into a end_point dict.
  end_points = utils.convert_collection_to_dict(end_points_collection)
  if spatial_squeeze:
    net = array_ops.squeeze(net, [1, 2], name='fc8/squeezed')
    end_points[sc.name + '/fc8'] = net
  return net, norms

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
