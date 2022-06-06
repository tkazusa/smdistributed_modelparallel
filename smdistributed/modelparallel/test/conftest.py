# Standard Library
import os

conda_default_env = os.environ["CONDA_DEFAULT_ENV"]

collect_ignore_glob = []

if conda_default_env in ["tensorflow2_p36", "tensorflow_p36"]:
    collect_ignore_glob.append("torch")
elif conda_default_env in ["pytorch_p36", "pytorch_latest_p36"]:
    collect_ignore_glob.append("tensorflow")
    collect_ignore_glob.append("tensorflow2")
