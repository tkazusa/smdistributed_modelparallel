#!/usr/bin/env python3

# Third Party
import tensorflow as tf

# First Party
import smdistributed.modelparallel.tensorflow.v1 as smp

smp.init({"partitions": 4})

DTYPES = [tf.float16, tf.float32, tf.float64, tf.int8, tf.uint8, tf.int16, tf.uint16]
DTYPES += [tf.int32, tf.uint32, tf.int64, tf.uint64, tf.complex64, tf.complex128, tf.bool]

res = []
expected = []

for dtype in DTYPES:

    @smp.step()
    def func():
        input = tf.constant([1.0, 1.0]) * smp.rank()
        input = tf.cast(input, dtype=dtype)
        return smp.allgather_tensor_ppgroup(input)

    expected.append(
        tf.cast(tf.stack([tf.constant([1.0, 1.0]) * r for r in range(smp.size())]), dtype=dtype)
    )
    res.append(func().outputs[0])

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(smp.local_rank())

with tf.train.MonitoredTrainingSession(checkpoint_dir=None, config=config) as sess:
    outputs, expected = sess.run([res, expected])
    for out, exp in zip(outputs, expected):
        assert (out == exp).all(), f"Mismatched output: {(out, exp)}"


smp.barrier()
if smp.rank() == 0:
    print("DType test passed!")
