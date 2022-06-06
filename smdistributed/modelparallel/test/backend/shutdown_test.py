# Standard Library
import argparse
import os
import random
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    import smdistributed.modelparallel.tensorflow as smp
except ImportError:
    import smdistributed.modelparallel.torch as smp

parser = argparse.ArgumentParser(description="Shutdown test")
parser.add_argument("--prob", type=float, default=0.5, help="probability to fail")
parser.add_argument("--index", type=int, default=0, help="test index")

args = parser.parse_args()

cfg = {"partitions": 8}
smp.init(cfg)

random.seed(123 * args.index + smp.rank())
time.sleep(random.random() * 10)
if random.random() < args.prob:
    raise ValueError(f"rank {smp.rank()} error")
