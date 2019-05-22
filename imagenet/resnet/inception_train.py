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
"""A library to train Inception using multiple GPUs with synchronous updates.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from datetime import datetime
import os.path
import re
import time, sys
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

import image_processing
import inception_model as inception
#from slim import slim

import norm_monitor

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('subset', 'train',
                           """Either 'train' or 'validation'.""")

# Flags governing the hardware employed for running TensorFlow.
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

# Flags governing the type of training.
tf.app.flags.DEFINE_boolean('fine_tune', False,
                            """If set, randomly initialize the final layer """
                            """of weights in order to train the network on a """
                            """new task.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")

tf.app.flags.DEFINE_string('layerinfo_file', '', """If specified, restore bits info """)

# **IMPORTANT**
# Please note that this learning rate schedule is heavily dependent on the
# hardware architecture, batch size and any changes to the model architecture
# specification. Selecting a finely tuned learning rate schedule is an
# empirical process that requires some experimentation. Please see README.md
# more guidance and discussion.
#
# With 8 Tesla K40's and a batch size = 256, the following setup achieves
# precision@1 = 73.5% after 100 hours and 100K steps (20 epochs).
# Learning rate decay factor selected from http://arxiv.org/abs/1404.5997.
tf.app.flags.DEFINE_float('num_epochs_per_decay', 30.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16,
                          """Learning rate decay factor.""")

tf.app.flags.DEFINE_float('initial_learning_rate', 0.1, """Initial learning rate.""")
tf.app.flags.DEFINE_float('L2_reg', 0.0001, """L2 regularizaion""")
tf.app.flags.DEFINE_integer('initial_bits', 8, """initial bits""")
tf.app.flags.DEFINE_boolean('use_bitpack', False, """If use bitpack""")
tf.app.flags.DEFINE_boolean('profile', False, """If produce trace""")
tf.app.flags.DEFINE_integer('digits', 6, """digits""")
tf.app.flags.DEFINE_float('rel_res', 1E-4, """If produce trace""")
tf.app.flags.DEFINE_integer('interval', 30, """If produce trace""")
tf.app.flags.DEFINE_integer('stride', 1, """If produce trace""")

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.


def _tower_loss(images, labels, num_classes, scope, reuse_variables=None, bits_ph=[]):
  """Calculate the total loss on a single tower running the ImageNet model.

  We perform 'batch splitting'. This means that we cut up a batch across
  multiple GPUs. For instance, if the batch size = 32 and num_gpus = 2,
  then each tower will operate on an batch of 16 images.

  Args:
    images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                       FLAGS.image_size, 3].
    labels: 1-D integer Tensor of [batch_size].
    num_classes: number of classes
    scope: unique prefix string identifying the ImageNet tower, e.g.
      'tower_0'.

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """
  # When fine-tuning a model, we do not restore the logits but instead we
  # randomly initialize the logits. The number of classes in the output of the
  # logit is the number of classes in specified Dataset.
  restore_logits = not FLAGS.fine_tune

  # Build inference Graph.
  with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
    if FLAGS.use_bitpack is False:
      logits, norms = inception.inference_resnet(images, num_classes, for_training=True, scope=scope, bits_ph=bits_ph)
    else:
      logits, norms = inception.inference_resnet_bitpack(images, num_classes, for_training=True, scope=scope, bits_ph=bits_ph)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  split_batch_size = images.get_shape().as_list()[0]
  losses = inception.loss_resnet(logits, labels, batch_size=split_batch_size, scope=scope)

  # Assemble all of the losses for the current tower only.
  #losses = tf.get_collection(slim.losses.LOSSES_COLLECTION, scope)

  # Calculate the total loss for the current tower.
  regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  print("reg_losses {}".format(regularization_losses))
  total_loss = tf.add_n(losses + regularization_losses, name='total_loss')

  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summmary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on TensorBoard.
    loss_name = re.sub('%s_[0-9]*/' % inception.TOWER_NAME, '', l.op.name)
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(loss_name +' (raw)', l)
    tf.summary.scalar(loss_name, loss_averages.average(l))

  with tf.control_dependencies([loss_averages_op]):
    total_loss = tf.identity(total_loss)
  return total_loss, norms, logits


def _average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train(dataset):
  #sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
  """Train on dataset for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    tf.set_random_seed(time.time())
    tf.set_random_seed(198918)
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    bits_ph = []
    for i in range(31):
        bits_ph.append(tf.placeholder(tf.int32))

    nm = norm_monitor.norm_monitor(FLAGS.digits, len(bits_ph), FLAGS.rel_res, FLAGS.interval, FLAGS.stride)
    if FLAGS.layerinfo_file:
      assert tf.gfile.Exists(FLAGS.layerinfo_file)
      tmp = pickle.load(open(FLAGS.layerinfo_file,'rb'))
      nm.set_layerinfo(tmp[-1])
      print("Restore layerinfo")
      print(nm.get_layerinfo())
    #print(nm.get_layerinfo())

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (dataset.num_examples_per_epoch() / FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)
    print("num_batches_per_epoch: {}".format(num_batches_per_epoch))
    print("use bitpack: {}".format(FLAGS.use_bitpack))
    print("learning rate: {}".format(FLAGS.initial_learning_rate))
    print("produce trace: {}".format(FLAGS.profile))
    print("digits: {}".format(FLAGS.digits))
    print("rel_res: {}".format(FLAGS.rel_res))
    print("interval: {}".format(FLAGS.interval))
    print("stride: {}".format(FLAGS.stride))

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    FLAGS.learning_rate_decay_factor,
                                    staircase=True)

    # Create an optimizer that performs gradient descent.
    opt = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY, momentum=RMSPROP_MOMENTUM, epsilon=RMSPROP_EPSILON)

    # Get images and labels for ImageNet and split the batch across GPUs.
    assert FLAGS.batch_size % FLAGS.num_gpus == 0, (
        'Batch size must be divisible by number of GPUs')
    split_batch_size = int(FLAGS.batch_size / FLAGS.num_gpus)

    # Override the number of preprocessing threads to account for the increased
    # number of GPU towers.
    num_preprocess_threads = FLAGS.num_preprocess_threads * FLAGS.num_gpus
    images, labels = image_processing.distorted_inputs(
        dataset,
        num_preprocess_threads=num_preprocess_threads)

    input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))

    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.
    num_classes = dataset.num_classes() + 1

     # Split the batch of images and labels for towers.
    images_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=images)
    labels_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=labels)

    # Calculate the gradients for each model tower.
    tower_norms  = []
    tower_grads  = []
    tower_preds_1  = []
    tower_preds_5  = []
    tower_losses = []

    reuse_variables = None
    for i in range(FLAGS.num_gpus):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('%s_%d' % (inception.TOWER_NAME, i)) as scope:
          # Force all Variables to reside on the CPU.
          # Calculate the loss for one tower of the ImageNet model. This
          # function constructs the entire ImageNet model but shares the
          # variables across all towers.
          #print(images_splits[i])
          #print(labels_splits[i])
          loss, norms, logits_split = _tower_loss(images_splits[i], labels_splits[i], num_classes, scope, reuse_variables, bits_ph)
          top_1_correct = tf.nn.in_top_k(logits_split, labels_splits[i], 1)
          top_5_correct = tf.nn.in_top_k(logits_split, labels_splits[i], 5)
          # Reuse variables for the next tower.
          reuse_variables = True

          # Retain the summaries from the final tower.
          summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

          # Retain the Batch Normalization updates operations only from the
          # final tower. Ideally, we should grab the updates from all towers
          # but these stats accumulate extremely fast so we can ignore the
          # other stats from the other towers without significant detriment.
          #batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION, scope)
          batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

          # Calculate the gradients for the batch of data on this ImageNet
          # tower.
          grads = opt.compute_gradients(loss)

          # Keep track of the gradients across all towers.
          tower_grads.append(grads)
          tower_norms.append(norms)
          tower_preds_1.append(tf.reduce_sum(tf.cast(top_1_correct, tf.int32)))
          tower_preds_5.append(tf.reduce_sum(tf.cast(top_5_correct, tf.int32)))
          tower_losses.append(loss)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = _average_gradients(tower_grads)

    top_1_sum = tf.add_n(tower_preds_1)
    top_5_sum = tf.add_n(tower_preds_5)
    losses_sum = tf.add_n(tower_losses)
    # Add a summaries for the input processing and global_step.
    summaries.extend(input_summaries)

    # Add a summary to track the learning rate.
    summaries.append(tf.summary.scalar('learning_rate', lr))

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(
            tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(var.op.name, var))

    # Track the moving averages of all trainable variables.
    # Note that we maintain a "double-average" of the BatchNormalization
    # global statistics. This is more complicated then need be but we employ
    # this for backward-compatibility with our previous models.
    variable_averages = tf.train.ExponentialMovingAverage(
        inception.MOVING_AVERAGE_DECAY, global_step)

    # Another possibility is to use tf.slim.get_variables().
    variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
    variables_averages_op = variable_averages.apply(variables_to_average)

    # Group all updates to into a single train op.
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, variables_averages_op, batchnorm_updates_op)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    if FLAGS.pretrained_model_checkpoint_path:
      assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
      #variables_to_restore = tf.get_collection(slim.variables.VARIABLES_TO_RESTORE)
      restorer = tf.train.Saver(tf.global_variables(), max_to_keep=100)
      restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)
      print('%s: Pre-trained model restored from %s' %
            (datetime.now(), FLAGS.pretrained_model_checkpoint_path))
    for v in tf.all_variables():
      print("%s %s %s %s" % (v.name, v.get_shape(), v.dtype, v.device))
    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(
        FLAGS.train_dir,
        graph=sess.graph)

    bits_dict = dict()
    #run_metadata = tf.RunMetadata()
    elapse = []

    #gweights = []
    glayerinfo = []
    #wnp_name = 'weights_norm_{}_{}_{}_{}_{}_{}_{}.dat'.format(9, 2048, 0, FLAGS.digits, FLAGS.stride, FLAGS.interval, FLAGS.use_bitpack)
    lip_name = 'layerinfo_{}_{}_{}_{}_{}_{}_{}.dat'.format(9, 4096, 0, FLAGS.digits, FLAGS.stride, FLAGS.interval, FLAGS.use_bitpack)

    for step in range(FLAGS.max_steps):
      run_metadata = tf.RunMetadata()
      start_time = time.time()
      info = nm.get_layerinfo()
      for i, bits in enumerate(bits_ph):
        bits_dict[bits] = info[i][0]
      if FLAGS.profile is False:
        _, loss_value, norms, top_1, top_5 = sess.run([train_op, losses_sum, tower_norms, top_1_sum, top_5_sum], feed_dict=bits_dict)
      else:
        _, loss_value, norms = sess.run([train_op, loss, tower_norms], 
                                 feed_dict=bits_dict, 
                                 options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), 
                                 run_metadata=run_metadata)
        top_1 = 5
        top_5 = 25

      nm.adjust_digits(norms)
      duration = time.time() - start_time
      #gweights.append(norms)
      #glayerinfo.append(copy.deepcopy(nm.get_layerinfo()))
      elapse.append(duration)

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        glayerinfo.append(copy.deepcopy(nm.get_layerinfo()))
        # Print layerinfo
        print(info)
        examples_per_sec = FLAGS.batch_size / float(duration)
        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f ' 'sec/batch) elapse %.5f s top_1 %.5f top_5 %.5f')
        pred_1 = top_1 / (FLAGS.batch_size*FLAGS.num_gpus)
        pred_5 = top_5 / (FLAGS.batch_size*FLAGS.num_gpus)
        print(format_str % (datetime.now(), step, loss_value, examples_per_sec, duration, sum(elapse), pred_1, pred_5))
        sys.stdout.flush()
        tl = timeline.Timeline(run_metadata.step_stats)
        if FLAGS.profile is True:
          if FLAGS.use_bitpack is False:
            trace_file = tf.gfile.Open(name='timeline%03d.json' % step, mode='w')
          else:
            trace_file = tf.gfile.Open(name='bitpack_timeline%03d.json' % step, mode='w')
          trace_file.write(tl.generate_chrome_trace_format(show_memory=True))

      if step % 100 == 0:
        summary_str = sess.run(summary_op, feed_dict=bits_dict)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 4000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

  glayerinfo.append(copy.deepcopy(nm.get_layerinfo()))
  #pickle.dump(gweights, open(wnp_name,'wb'))
  pickle.dump(glayerinfo, open(lip_name,'wb'))
