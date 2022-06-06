#!/usr/bin/env python3

# Third Party
import tensorflow as tf

# First Party
import smdistributed.modelparallel.tensorflow.v1 as smp
from smdistributed.modelparallel.tensorflow import state

smp.init({"partitions": 2, "microbatches": 4, "horovod": True})
if smp.pp_rank() == 0:
    x = tf.constant(1.0)
else:
    x = tf.constant(2.0)

allgather_output1 = smp.allgather_tensor_ppgroup(x)
output1 = tf.reduce_sum(allgather_output1)
allgather_output2 = smp.allgather_tensor_ppgroup(output1)
output2 = tf.reduce_sum(allgather_output2)

assert state.allgather_idx == 2, state.allgather_idx


@smp.step()
def func():
    input = tf.constant([1.0, 1.0])
    allgather_output3 = smp.allgather_tensor_ppgroup(input)
    output3 = tf.reduce_sum(allgather_output3)
    allgather_output4 = smp.allgather_tensor_ppgroup(output3)
    output4 = tf.reduce_sum(allgather_output4)
    return output4


output4 = func()
assert state.allgather_idx == 4, state.allgather_idx

output4 = output4.reduce_mean() + output2
allgather_output5 = smp.allgather_tensor_ppgroup(output4)
output5 = tf.reduce_sum(allgather_output5)
allgather_output6 = smp.allgather_tensor_ppgroup(output5)
output6 = tf.reduce_sum(allgather_output6)

assert state.allgather_idx == 6, state.allgather_idx


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(smp.local_rank())

with tf.train.MonitoredTrainingSession(checkpoint_dir=None, config=config) as sess:
    output = sess.run(output6)
    assert output == 56, output

smp.barrier()
if smp.rank() == 0:
    print("Allgather test passed!")
