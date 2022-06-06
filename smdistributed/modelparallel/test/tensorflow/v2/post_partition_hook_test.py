#!/usr/bin/env python3

# Third Party
import tensorflow as tf

# First Party
import smdistributed.modelparallel.tensorflow as smp

smp.init({"partitions": 2, "microbatches": 1})


class MyModel(smp.DistributedModel):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = tf.keras.layers.Dense(2, activation="relu")
        self.d2 = tf.keras.layers.Dense(2, activation="relu")

    def call(self, x):
        x = self.d1(x)
        return self.d2(x)


model = MyModel()
loss_object = tf.keras.losses.MSE


@smp.step
def get_grads(inputs, labels):
    predictions = model(inputs, training=True)
    return tf.reduce_mean(loss_object(labels, predictions))


@tf.function
def train_smp_step(inputs, labels):
    loss = get_grads(inputs, labels)
    return loss.reduce_mean()


@smp.register_post_partition_hook
def test_eager():
    tf.print("Entered hook through eager context")


@smp.register_post_partition_hook
def test_sample_hook_fn(x, y, sample_hook_result):
    res = x ** 3 + y ** 3
    tf.print(f"Test hook function result is {res}")
    sample_hook_result.append(res)


test_eager()
sample_hook_result = []
test_sample_hook_fn(2, 2, sample_hook_result)

assert (
    sample_hook_result == []
), f"Array passed to hook must be empty here since hook is not executed."

inputs = tf.ones([2, 2])
for step in range(5):
    smp_loss = train_smp_step(inputs, inputs)
    if step == 0:
        assert (
            sample_hook_result[0] == 16
        ), f"Sample hook result should not be empty after execution of first step."
    if smp.rank() == 0:
        print(f"Step: {step+1} SMP loss: {smp_loss.numpy()}")

assert (
    sample_hook_result[0] == 16
), f"Sample hook result should not be empty after execution of first step."
assert len(sample_hook_result) == 1, f"Sample hook result should not contain more than 1 value."


smp.barrier()
if smp.rank() == 0:
    print("Test Complete")
