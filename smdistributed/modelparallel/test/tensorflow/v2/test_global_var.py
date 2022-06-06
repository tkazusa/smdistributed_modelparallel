# Standard Library

# Third Party
import tensorflow as tf

# First Party
import smdistributed.modelparallel.tensorflow as smp
from smdistributed.modelparallel.tensorflow.utils import GraphBuildError

tf.config.optimizer.set_jit(False)

cfg = {
    "microbatches": 2,
    "partitions": 1,
    "placement_strategy": "spread",
    "pipeline": "interleaved",
    "optimize": "memory",
}
smp.init(cfg)

test_var = tf.Variable(1.0, name="test_var")


class MyModel(smp.DistributedModel):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense = tf.keras.layers.Dense(2, activation="relu")

    def call(self, x):
        global test_var
        x = x * test_var
        return self.dense(x)


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
try:
    grads, loss = train_step(x, y)
except GraphBuildError:
    print("OK")
except Exception as e:
    print(f"Fail to detect with wrong exception {e}")
    raise
else:
    print(f"Fail to detect the variable outside the smp.DistributedModel")
    raise
