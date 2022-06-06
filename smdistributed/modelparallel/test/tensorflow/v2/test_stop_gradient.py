# Standard Library

# Third Party
import tensorflow as tf

# First Party
import smdistributed.modelparallel.tensorflow as smp

tf.config.optimizer.set_jit(False)

cfg = {
    "microbatches": 2,
    "partitions": 2,
    "placement_strategy": "spread",
    "pipeline": "interleaved",
    "optimize": "memory",
}
smp.init(cfg)


class MyModel(smp.DistributedModel):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = tf.keras.layers.Dense(2, activation="relu")
        self.d2 = tf.keras.layers.Dense(2, activation="relu")
        self.add = tf.keras.layers.add

    def first(self, x):
        with tf.name_scope("first") as scope:
            x = self.d1(x)
            x_ = tf.cast(x, dtype=tf.int32)
            return x, x_

    def second(self, x, x_):
        with tf.name_scope("second") as scope:
            x_ = tf.cast(x_, dtype=tf.float32)
            x = x + x_
            ret = self.d2(x)
            return ret

    def call(self, x):
        x, id_x = self.first(x)
        x = self.second(x, id_x)
        return x


# Create an instance of the model
model = MyModel()
optimizer = tf.keras.optimizers.SGD()
loss_obj = tf.keras.losses.MeanSquaredError()


@smp.step
def get_grads(x, y):
    preds = model(x, training=True)
    loss = loss_obj(y, preds)
    grads = optimizer.get_gradients(loss, model.trainable_variables)
    return grads, loss


@tf.function
def train_step(x, y):
    gradients, loss = get_grads(x, y)
    gradients = [g.reduce_mean() for g in gradients]
    return gradients, loss.reduce_mean()


grads_list, loss_list = [], []
x = tf.ones([2, 2])
y = tf.constant([1.0, 0.0])
for step in range(2):
    grads, loss = train_step(x, y)
    grads_list.append(grads)
    loss_list.append(loss)

assert len(loss_list) == 2
assert len(grads_list) == 2
print("OK")
