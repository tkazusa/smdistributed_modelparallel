# Standard Library
import atexit
import ctypes
import os
import sys

# First Party
from smdistributed.modelparallel.backend.exceptions import SMPUnsupportedError
from smdistributed.modelparallel.backend.logger import get_logger


def init_internal(lib_path, config, core, state, true_if_pt_else_tf, start_pipeline_threads=True):
    from smdistributed.modelparallel.backend.utils import get_ext_suffix
    from smdistributed.modelparallel.backend.config import ModelParallelConfig

    if isinstance(config, dict):
        cfg = ModelParallelConfig(config)
    elif isinstance(config, ModelParallelConfig):
        cfg = config
    else:
        raise SMPUnsupportedError

    path = os.path.join(lib_path, "smplib" + get_ext_suffix())

    core.initialize(path, cfg, true_if_pt_else_tf, start_pipeline_threads)
    state.initialize(cfg, core)
    os.environ["LOCAL_RANK"] = str(core.local_rank())
    os.environ["RANK"] = str(core.rank())
    os.environ["WORLD_SIZE"] = str(core.size())
    if core.rank() == 0:
        cfg.display_config()
