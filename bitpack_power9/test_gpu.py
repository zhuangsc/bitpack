#! /usr/bin/env python3

import time
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

KEEPING_BITS = 8
SIZE = 5
SIZE = 20 * 64 * 5 * 5 
SHAPE = [20, 64, 5, 5]
REP = 100

rseed = int(time.time())
np.random.seed(rseed)
tf.set_random_seed(rseed)

bit_packer = tf.load_op_library('/home/bsc28/bsc28687/minotauro/ann/tensorflow/bit_packer/bit_packer.so')

pre_a = np.random.normal(loc=0.0, scale=1E-1, size=SIZE)

with tf.device("/cpu:0"):
    #a = tf.constant(pre_a, dtype=tf.float32, name="a")
    a = tf.random_normal(SHAPE, mean=0.1, stddev=0.5, dtype=tf.float32, name="a_init")
    #b = bit_packer.bit_pack(a, KEEPING_BITS, name="s-packer")
    b = bit_packer.bit_pack_cpu(a, KEEPING_BITS, name="m-packer")
    #zeros = tf.zeros([SIZE], dtype=tf.float32, name="zeros")

with tf.device("/gpu:0"):
    c = bit_packer.bit_unpack_gpu(b, KEEPING_BITS, SIZE, SHAPE, name="unpacker")
    #d = tf.add(c, zeros, name="zero_add")

#with tf.device("/cpu:0"):
#    #a = tf.constant(pre_a, dtype=tf.float32, name="a")
#    a = tf.random_normal([SIZE], mean=0.1, stddev=0.5, dtype=tf.float32, name="a_init")
#    b = tf.get_variable('b', [SIZE/2], initializer=tf.constant_initializer(0), dtype=tf.float32)
#    #b = bit_packer.bit_pack(a, KEEPING_BITS, name="s-packer")
#    bit_packer.bit_pack_cpu_ip(a, KEEPING_BITS, b, name="m-packer")
#    #zeros = tf.zeros([SIZE], dtype=tf.float32, name="zeros")
#
#with tf.device("/gpu:0"):
#    c = tf.get_variable('c', [SIZE], initializer=tf.constant_initializer(0), dtype=tf.float32)
#    c = bit_packer.bit_unpack_gpu(b, KEEPING_BITS, SIZE, name="unpacker")
#    #d = tf.add(c, zeros, name="zero_add")


summary = tf.summary.merge_all()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
#summary_writer = tf.summary.FileWriter("a_sum", sess.graph)
init = tf.global_variables_initializer()
sess.run(init)

res = []
elapse = np.ndarray(shape=(REP,), dtype=np.float32)
for i in range(REP):
    run_metadata = tf.RunMetadata()
    start = time.time()
    e, f = sess.run([a, c])
    #e = sess.run(d, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
    stop = time.time()
    elapse[i] = stop-start
    #res.append(1)
    ne = np.linalg.norm(e)
    nf = np.linalg.norm(f)
    res.append(abs((ne-nf)/nf))
sess.close()

print("Avg elapse time: {} s".format(np.mean(elapse[1:])))
#tl = timeline.Timeline(run_metadata.step_stats)
#trace_file = tf.gfile.Open(name='test_gpu.json', mode='w')
#trace_file.write(tl.generate_chrome_trace_format(show_memory=True))
#summary_writer.flush()

print(e.shape)
print(f.shape)

print("Size: {}, Rep: {}".format(SIZE, REP))
print("Keeping {} bits, sizeof array {}".format(KEEPING_BITS, SIZE))
tot_float_bits = SIZE * 32
tot_float_out_bits = SIZE * KEEPING_BITS
print("tot_float_bits: {}".format(tot_float_bits))
print("tot_float_out_bits: {}".format(tot_float_out_bits))
print("tot_float_out: {}".format(int((tot_float_out_bits+31)/32)))

print("Relative Residual")
res_avg = np.average(res)
res_std = np.std(res)
print("AVG: {}, STD: {}".format(res_avg, res_std))
