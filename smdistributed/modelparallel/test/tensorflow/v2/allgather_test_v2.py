#!/usr/bin/env python3

# Third Party
import tensorflow as tf

# First Party
import smdistributed.modelparallel.tensorflow as smp
from smdistributed.modelparallel.tensorflow import state


def print_tensor(tensor, name):
    print_op = tf.print(name, tensor)
    with tf.control_dependencies([print_op]):
        return tf.identity(tensor)


smp.init({"partitions": 2, "microbatches": 2, "horovod": True})
if smp.pp_rank() == 0:
    x = tf.constant(1.0)
else:
    x = tf.constant(2.0)


class MyModel(smp.DistributedModel):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = tf.keras.layers.Dense(2, activation="relu")
        self.d2 = tf.keras.layers.Dense(2, activation="relu")

    def call(self, x):
        x = self.d1(x)
        return self.d2(x)


model = MyModel()


@tf.function
def test(dummy, x):
    allgather_output1 = smp.allgather_tensor_ppgroup(x)
    output1 = tf.reduce_sum(allgather_output1)
    allgather_output2 = smp.allgather_tensor_ppgroup(output1)
    output2 = tf.reduce_sum(allgather_output2)

    assert state.allgather_idx == 2, state.allgather_idx

    @smp.step()
    def func(dummy):
        preds = model(dummy, training=True)
        input = tf.constant([1.0, 1.0])
        allgather_output3 = smp.allgather_tensor_ppgroup(input)
        output3 = tf.reduce_sum(allgather_output3)
        allgather_output4 = smp.allgather_tensor_ppgroup(output3)
        output4 = tf.reduce_sum(allgather_output4)
        return output4, preds

    output4, _ = func(dummy)
    assert state.allgather_idx == 6, state.allgather_idx

    output4 = output4.reduce_mean() + output2
    allgather_output5 = smp.allgather_tensor_ppgroup(output4)
    output5 = tf.reduce_sum(allgather_output5)
    allgather_output6 = smp.allgather_tensor_ppgroup(output5)
    output6 = tf.reduce_sum(allgather_output6)

    assert state.allgather_idx == 8, state.allgather_idx
    return output6


dummy = tf.ones([2, 2])
output = test(dummy, x)
assert output == 56, output

smp.barrier()
if smp.rank() == 0:
    print("Allgather test passed!")
