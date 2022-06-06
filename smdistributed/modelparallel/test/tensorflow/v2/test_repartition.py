#!/usr/bin/env python3

# Third Party
import tensorflow as tf

# First Party
import smdistributed.modelparallel.tensorflow as smp

smp.init({"partitions": 2, "microbatches": 2, "optimize": "speed"})


class MyModel(smp.DistributedModel):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(2, (1, 1), padding="same", input_shape=(2, 2, 1))
        self.act1 = tf.keras.layers.Activation("relu")
        self.d1 = tf.keras.layers.Dense(2)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.flatten = tf.keras.layers.Flatten()
        self.act2 = tf.keras.layers.Activation("softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.act2(x)


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


@smp.step
def val_smp_step(inputs, labels):
    predictions = model(inputs, training=False)
    return tf.reduce_mean(loss_object(labels, predictions))


inputs = tf.ones([2, 2, 2, 1])
for step in range(3):
    train_smp_loss = train_smp_step(inputs, inputs)
    val_smp_loss = val_smp_step(inputs, inputs)
    val_smp_loss = val_smp_loss.reduce_mean()
    if smp.rank() == 0:
        print(
            f"Step: {step+1} SMP train loss: {train_smp_loss.numpy()} SMP validation loss: {val_smp_loss.numpy()}"
        )


smp.barrier()
if smp.rank() == 0:
    print("Test Complete")
