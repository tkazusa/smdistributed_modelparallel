# Standard Library
import os

conda_default_env = os.environ["CONDA_DEFAULT_ENV"]

collect_ignore_glob = [
    "v1/allgather_test_v1.py",
    "v2/allgather_test_v2.py",
    "dtype_test.py",
    "v2/test_stop_gradient.py",
    "v2/test_repartition.py",
    "v2/post_partition_hook_test.py",
]

if conda_default_env in ["tensorflow2_p36"]:
    collect_ignore_glob.append("v1")
elif conda_default_env in ["tensorflow_p36"]:
    collect_ignore_glob.append("v2")
