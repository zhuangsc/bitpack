#! /usr/bin/env python3

import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

rseed = int(time.time())
#rseed= 198918
np.random.seed(rseed)
tf.set_random_seed(rseed)

bit_packer = tf.load_op_library('/home/bsc28/bsc28687/power9/ann/tensorflow/bit_packer/bit_packer.so')

def elapse(mode, KEEPING_BITS, REP, scale, gpus):

    SIZE = 20 * 64 * 5 * 5 * scale
    SHAPE = [20*scale, 64, 5, 5]

    #SIZE = 8 * scale
    #SHAPE = [8*scale]

    with tf.device("/cpu:0"):
        a = tf.Variable(tf.random_normal(SHAPE, mean=0.1, stddev=0.5, dtype=tf.float32, name="a_init"), name="a")
        if mode == 1:
            b = bit_packer.bit_pack_cpu(a, KEEPING_BITS, name="m-packer")
        elif mode == 2:
            b = bit_packer.bit_pack_cpu_avx(a, KEEPING_BITS, name="m-packer")
        elif mode == 3:
            b = bit_packer.bit_pack_cpu_avx_omp(a, KEEPING_BITS, name="m-packer")


    gg = []
    for i in range(gpus):
        with tf.device('/gpu:%d' % i):
            if mode == 0:
                g = tf.add(a, 0, name="add")
            else:
                g = bit_packer.bit_unpack_gpu(b, KEEPING_BITS, SIZE, SHAPE, name="unpacker")
            gg.append(g)

    grad = tf.reduce_mean(gg, 0)
    summary = tf.summary.merge_all()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    summary_writer = tf.summary.FileWriter("time_elapse", sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)

    res = []
    elapse = np.ndarray(shape=(REP,), dtype=np.float32)
    for i in range(REP):
        run_metadata = tf.RunMetadata()
        start = time.time()
        e, f  = sess.run([grad, a])
        #e = sess.run(grad, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
        stop = time.time()
        elapse[i] = (stop-start) * 1000000
        #res.append(1)
        ne = np.linalg.norm(e)
        nf = np.linalg.norm(f)
        res.append(abs((ne-nf)/nf))
    sess.close()

    print("avg: {} us".format(np.mean(elapse[1:])))
    #for ela in elapse[1:]:
    #    print(ela)
    tl = timeline.Timeline(run_metadata.step_stats)
    trace_file = tf.gfile.Open(name='timeline.json', mode='w')
    trace_file.write(tl.generate_chrome_trace_format(show_memory=True))
    summary_writer.flush()


    #print("Size: {}, Rep: {}".format(SIZE, REP))
    #print("Keeping {} bits, sizeof array {}".format(KEEPING_BITS, SIZE))
    #tot_float_bits = SIZE * 32
    #tot_float_out_bits = SIZE * KEEPING_BITS
    #print("tot_float_bits: {}".format(tot_float_bits))
    #print("tot_float_out_bits: {}".format(tot_float_out_bits))
    #print("tot_float_out: {}".format(int((tot_float_out_bits+31)/32)))

    print("Relative Residual")
    res_avg = np.average(res)
    res_std = np.std(res)
    print("AVG: {}, STD: {}".format(res_avg, res_std))

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("USAGE: {} [mode] [bits] [rep] [scale] [gpus]".format(sys.argv[0]))
        sys.exit(1)
    mode = int(sys.argv[1])
    bits = int(sys.argv[2])
    rep = int(sys.argv[3])
    scale = int(sys.argv[4])
    gpus = int(sys.argv[5])
    elapse(mode, bits, rep, scale, gpus)
