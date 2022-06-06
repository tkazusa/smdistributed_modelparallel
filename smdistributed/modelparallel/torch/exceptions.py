# First Party
from smdistributed.modelparallel.torch.core import pp_rank

_skip_suggestion = (
    "If you think the detected scenario is not the case with your code,"
    "please report this issue and set the environment variable SMP_SKIP_GRAPH_VALIDATION=1"
    "to bypass this check. If you choose to skip this validation, "
    "please make sure to verify that grads for the whole model are being computed as expected."
)


class InvalidExecutor(RuntimeError):
    def __init__(self, module_name, partition):
        self.module_name = module_name
        self.partition = partition

    def __str__(self):
        return (
            f"{self.module_name} is assigned the partition {self.partition}, but was being executed on "
            f"the partition {pp_rank()}. This module should only be executed by parent and actual executor."
        )


class MissingPathFromComputationToModuleOutput(RuntimeError):
    def __init__(self, parent_module_name, module_name):
        self.parent_module_name = parent_module_name
        self.module_name = module_name

    def __str__(self):
        return (
            f"Unsupported usecase found during execution of {self.parent_module_name}. "
            f"The output obtained from module: {self.module_name}, doesn't have a path to "
            f"outputs of its ancestor module: {self.parent_module_name}, on which backward is called."
            f"This breaks backward flow with pipeline parallelism. "
            f"Please either remove such computation from the module, or execute it in a torch.no_grad() context, or detach it within the module {self.module_name} before returning its output. "
            f"{_skip_suggestion}"
        )


class MissingPathFromModuleInputToModuleOutput(RuntimeError):
    def __init__(self, module_name, idx):
        self.module_name = module_name
        self.idx = idx

    def __str__(self):
        return (
            f"Unsupported usecase found during module execution of: {self.module_name}. "
            f"The input with index {self.idx} to the module doesn't have a path to the outputs on which backward is called."
            f" Please remove the unused input, or detach it before passing to the module. {_skip_suggestion}"
        )


class NumParametersNotMatch(RuntimeError):
    def __init__(self, origin_params, dist_params):
        self.origin_params = origin_params
        self.dist_params = dist_params

    def __str__(self):
        return (
            f"Origin module got {self.origin_params} parameters but it's dist counterpart got {self.dist_params}"
            f" Please check whether module hyperparameters were fed correctly."
            " If you think the detected scenario is not the case with your code,"
            " please report this issue and set the environment variable SMP_SKIP_PARAMS_CHECKING=1"
        )
