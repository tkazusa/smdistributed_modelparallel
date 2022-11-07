# Standard Library
from distutils.version import LooseVersion

# First Party
from smdistributed.modelparallel.backend.exceptions import (
    SMPInvalidArgumentError,
    SMPRuntimeError,
    SMPUnsupportedError,
    SMPValidationError,
    TensorSplitError,
)
from smdistributed.modelparallel.torch.core import pp_rank, pp_size, rdp_size, tp_size

_skip_suggestion = (
    "If you think the detected scenario is not the case with your code,"
    "please report this issue and set the environment variable SMP_SKIP_GRAPH_VALIDATION=1"
    "to bypass this check. If you choose to skip this validation, "
    "please make sure to verify that grads for the whole model are being computed as expected."
)

# Validation errors


class DDPNotEnabledError(SMPValidationError):
    def __str__(self):
        return "Torch.dist is not initialized. Please enable DDP in config to use torch.dist"


class InvalidCommGroupError(SMPValidationError):
    def __init__(self, group):
        self.group = group

    def __str__(self):
        return f"Invalid group {group} passed, it needs to be one of CommGroup.WORLD, CommGroup.PP_GROUP, CommGroup.DP_GROUP"


class DDPConfigError(SMPValidationError):
    pass


class HorovodConfigError(SMPValidationError):
    pass


class CheckpointingConfigError(SMPValidationError):
    pass


class DistEmbeddingConfigError(SMPValidationError):
    pass


class DistTransformerConfigError(SMPValidationError):
    pass


class HFBertConfigError(SMPValidationError):
    pass


class HFGPT2ConfigError(SMPValidationError):
    pass


class HFGPTJConfigError(SMPValidationError):
    pass


class HFGPTNeoConfigError(SMPValidationError):
    pass

class HFGPTNeoxConfigError(SMPValidationError):
    pass

class HFRobertaConfigError(SMPValidationError):
    pass


class HFT5ConfigError(SMPValidationError):
    pass

class HFViTConfigError(SMPValidationError):
    pass

class ModelTooSmallError(SMPValidationError):
    def __init__(self, num_partitions):
        self.num_partitions = num_partitions

    def __str__(self):
        return f"Model is too small to split into {self.num_partitions} partitions!"


class InvalidSeqLenPrescaledBatchError(SMPValidationError):
    def __str__(self):
        return "Sequence length must be greater than and divisible by the tensor parallelism degree when prescaled_batch is True."


class MemoryWeightError(SMPValidationError):
    def __str__(self):
        return "Memory weight must between 0.0 and 1.0."


class CostListError(SMPValidationError):
    def __str__(self):
        return "Cost list cannot be larger than the number of partitions."


class MultipleDistributedModelError(SMPValidationError):
    def __str__(self):
        "Using DistributedModel wrapper more than once in a process is not supported yet"


class BwdExecutionNotInStepFnError(SMPValidationError):
    def __str__(self):
        return "smp.DistributedModel's backward can only be run inside smp.step annotated function"


class FwdExecutionNotInStepFnError(SMPValidationError):
    def __str__(self):
        return "SMP DistributedModel forward can only be run from inside a smp.step annotated function when pipeline parallelism degree is more than 1."


class ModelNotPartitionedError(SMPValidationError):
    def __str__(self):
        return "Model has not been partitioned yet. You can call this method after first step when using autopartitioning."


class HiddenDimError(SMPValidationError):
    def __init__(self, dim):
        self.dim = dim

    def __str__(self):
        return f"Input tensor rank should be one of [4, 5], but is: {self.dim}"


class SplitShapeLenError(SMPValidationError):
    def __init__(self, split_shapes):
        self.split_shapes = split_shapes

    def __str__(self):
        return f"Split shapes {self.split_shapes} must have same length of tp_size {tp_size()}"


class SplitShapeMismatchError(SMPValidationError):
    def __init__(self, bs_size, split_shapes):
        self.bs_size = bs_size
        self.split_shapes = split_shapes

    def __str__(self):
        return f"Dimension size {self.bs_size} can not be splitted by {self.split_shapes}"


class ShiftValueError(SMPValidationError):
    def __init__(self, shift, local_size):
        self.shift = shift
        self.local_size = local_size

    def __str__(self):
        return f"Shift {self.shift} must be less than or equal to the local size {self.local_size}."


class PaddingSizeError(SMPValidationError):
    def __init__(self, orig_size, pad_size):
        self.orig_size = orig_size
        self.pad_size = pad_size

    def __str__(self):
        return f"When padding the tensor, orig_size {self.orig_size} must be smaller or equal to pad_size {self.pad_size}"


class DistributedModelNotWrappedError(SMPValidationError):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return f"The model needs to be wrapped with smp.DistributedModel before {self.msg}"


class DistributedModelWrappedError(SMPValidationError):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return f"The model needs to be wrapped with smp.DistributedModel after {self.msg}"


class StepFunctionCalledError(SMPValidationError):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return f"The smp.step-decorated function needs to be called after {self.msg}"


class InvalidPartitionIDError(SMPValidationError):
    def __str__(self):
        "Partition ID must be non-negative, and less than the pipeline parallel degree."


class HFNotAvailableError(SMPValidationError):
    def __str__(self):
        return "HuggingFace transformers library not found in the environment."


# Checkpoint errors


class SMPCheckpointError(SMPValidationError):
    pass


class MissingCheckpointFilesError(SMPCheckpointError):
    def __init__(self, prefix):
        self.prefix = prefix

    def __str__(self):
        return f"Could not find partial checkpoint files with prefix {self.prefix}"


class IncompatibleCheckpointRankFoundError(SMPCheckpointError):
    def __init__(self, rank_type, rank, num_parts, prefix):
        self.rank_type = rank_type
        self.rank = rank
        self.num_parts = num_parts
        self.prefix = prefix

    def __str__(self):
        return f"The {self.rank_type} of a checkpoint file ({self.rank}) exceeds the total number of unique {self.rank_type} checkpoint files ({self.num_parts}) matching given prefix ({self.prefix}). Please ensure all checkpoint parts are available for loading at the given location."


class IncompatibleCheckpointFoundError(SMPCheckpointError):
    def __init__(self, num_parts, matching_files):
        self.num_parts = num_parts
        self.matching_files = matching_files

    def __str__(self):
        msg = f"{pp_size()} (pipeline) partitions, {tp_size()} (tensor parallel) partitions, and {rdp_size()} reduced-data-parallel ranks"
        return f"Can not load a {self.num_parts} parts checkpoint to a model with {msg}. Please save the full model and load that instead when changing the number of partitions across runs. Matching files: {self.matching_files}"


class MissingKeysInCheckpointError(SMPCheckpointError):
    def __init__(self, missing_keys):
        self.missing_keys = missing_keys

    def __str__(self):
        return (
            f"len of missing keys greater than 0, {self.missing_keys} when loading."
            f"state_dict on {tp_rank()}, please check if keys are missing from the state_dict."
            f"If this is intended, set strict to False when calling load_state_dict on the DistributedModel"
        )


class RemoteBufferShouldNotExistError(SMPCheckpointError):
    def __init__(self, buffer_name):
        self.buffer_name = buffer_name

    def __str__(self):
        return f"remote variable/buffer {self.buffer_name} should not exist in local dict!"


class RemoteBufferShouldExistError(SMPCheckpointError):
    def __init__(self, buffer_name):
        self.buffer_name = buffer_name

    def __str__(self):
        return f"full state dict should contain the remote variable/buffer {self.buffer_name}!"


class InvalidReturnTypeFromCheckpointedModuleError(SMPCheckpointError):
    def __init__(self, s):
        self.s = s

    def __str__(self):
        return self.s


# Unsupported Errors


class FusedLAMBError(SMPUnsupportedError):
    pass


class CheckpointingError(SMPUnsupportedError):
    pass


class DelayedParamDeviceError(SMPUnsupportedError):
    def __str__(self):
        return "Parameter must be on CPU for delayed initialization."


class UnsupportedTorchVersionError(SMPUnsupportedError):
    def __str__(self):
        return f"Unsupported Torch version {LooseVersion(torch.__version__)}"


class UnsupportedReducerTypeError(SMPUnsupportedError):
    def __init__(self, reducer_type):
        self.reducer_type = reducer_type

    def __str__(self):
        return f"reducer type can only be default or scaled_batch, but {self.reducer_type} given."


class UnsupportedCommunicationVolumeUnitError(SMPUnsupportedError):
    def __init__(self, unit):
        self.unit = unit

    def __str__(self):
        return f"Unit can be MB or GB, getting {self.unit}."


class UnsupportedTPModuleError(SMPUnsupportedError):
    def __init__(self, module_type):
        self.module_type = module_type

    def __str__(self):
        return f"Modules of type {self.module_type} are not supported for tensor parallelism."


class TPModuleRegisterError(SMPUnsupportedError):
    def __init__(self, module_name):
        self.module_name = module_name

    def __str__(self):
        f"Can only register a sub-class of smp.nn.DistributedModule. Found {self.module_name}."


class MultipleDtypeOptShardingError(SMPUnsupportedError):
    def __init__(self, tensor_name1, dtype1, tensor_name2, dtype2):
        self.tensor_name1 = tensor_name1
        self.dtype1 = dtype1
        self.tensor_name2 = tensor_name2
        self.dtype2 = dtype2

    def __str__(self):
        return f"Currently shard_optimizer_state is supported only when all parameters in the model have the same dtype. Found {self.tensor_name1} with {self.dtype1}, and {self.tensor_name2} with {self.dtype2}"


class SequentialBackwardBrokenError(SMPUnsupportedError):
    def __init__(self, module_name, grad_enabled, input):
        self.module_name = module_name
        self.grad_enabled = grad_enabled
        self.input = input

    def __str__(self):
        return (
            f"Backward is broken in between sequential because there was no tensor which required gradients "
            f"in one of the layer's {self.module_name} outputs. {self.grad_enabled, self.input}"
        )


class NotSupportedByFastModeError(SMPUnsupportedError):
    def __init__(self, graph_change=False):
        self.graph_change = graph_change

    def __str__(self):
        if not self.graph_change:
            return "Model is not supported by fast mode, please set 'fast_mode' to False."
        else:
            return "A change in model graph is detected. Graph changes are not supported in fast mode, please set 'fast_mode' to False."


class RecursionDepthExceededDuringSerializationError(SMPUnsupportedError):
    def __init__(self, obj_type):
        self.obj_type = obj_type

    def __str__(self):
        return f"Recursion depth exceeded while serializing object of type {self.obj_type}. This is probably caused by a large, non-Tensor object being passed as an argument to a module, or being returned from a module or smp.step."


class UnsupportedMessageError(SMPUnsupportedError):
    def __init__(self, msg_type):
        self.msg_type = msg_type

    def __str__(self):
        return f"Unsupported message type {self.msg_type}."


# Runtime errors


class InvalidHandleError(SMPRuntimeError):
    pass


class UnsupportedShardedConfigError(RuntimeError):
    def __init__(self, s):
        self.s = s

    def __str__(self):
        return self.s


class ScaledBatchBufNotInDistModuleError:
    def __str__(self):
        return "Scaled batch buffer has to be in distributed module."


class MissingOutputForModuleError(SMPRuntimeError):
    def __init__(self, mb, parent_module_name, module_name, position):
        self.mb = mb
        self.parent_module_name = parent_module_name
        self.module_name = module_name
        self.position = position

    def __str__(self):
        return f"Could not fetch output for mb: {self.mb}, parent: {self.parent_module_name}, module: {self.module_name} with position {self.position} as it was not saved."


class MissingParentModuleError(SMPRuntimeError):
    def __init__(self, execution_stack, module_name):
        self.execution_stack = execution_stack
        self.module_name = module_name

    def __str__(self):
        return f"Exec stack was {self.execution_stack}; could not find parent_module of {self.module_name}"


class MissingModuleError(SMPRuntimeError):
    def __init__(self, module_name):
        self.module_name = module_name

    def __str__(self):
        return f"Cannot find the name for the module {self.module_name}. This is commonly caused by module instantiation inside a forward() method. Please create all child modules inside the __init__ method of its parent module."


class ParentNodeExistingError(SMPRuntimeError):
    def __init__(self, node, parent):
        self.node = node
        self.parent = parent

    def __str__(self):
        return f"Node {self.node} already has parent {self.parent}."


class ChildNodeExistingError(SMPRuntimeError):
    def __init__(self, node):
        self.node = node

    def __str__(self):
        return f"Child {self.node} already exists."


class UnrecognizedHFKeyError(SMPRuntimeError):
    def __init__(self, key, options=None):
        self.key = key
        self.options = options

    def __str__(self):
        msg = f"Unrecognized HuggingFace key {self.key}."
        if self.options != None:
            msg += f" Must be one of {[k for k in self.options]}."
        return msg


class CustomSoftmaxKernelDtypeError(SMPRuntimeError):
    def __str__(self):
        return "Fused softmax kernel can only be used with float16 or bfloat16 dtypes."


class MissingGradientError(SMPRuntimeError):
    def __init__(self, param_name):
        self.param_name = param_name

    def __str__(self):
        return rmsg(
            f"Param {self.param_name} has None grad although it required grad, can not allreduce it"
        )


class GradRequireGradError(SMPRuntimeError):
    def __str__(self):
        return "grad should not require grad"


class SMPAMPError(SMPRuntimeError):
    pass


class PipelineParallelBWDError(SMPRuntimeError):
    pass


class InvalidRequestError(SMPRuntimeError):
    def __init__(self, req):
        self.req = req

    def __str__(self):
        return f"request {self.req} does not exist."


class InvalidExecutorError(SMPRuntimeError):
    def __init__(self, module_name, partition):
        self.module_name = module_name
        self.partition = partition

    def __str__(self):
        return (
            f"{self.module_name} is assigned the partition {self.partition}, but was being executed on "
            f"the partition {pp_rank()}. This module should only be executed by parent and actual executor."
        )


class InvalidBwdCountError(SMPRuntimeError):
    def __str__(self):
        return "Pending smpinput bwd count is less than 0"


class NonDummyTensorError(SMPRuntimeError):
    def __init__(self, op):
        self.op = op

    def __str__(self):
        return f"{self.op} operation should be done on dummy tensor"


class InvalidParentModuleError(SMPRuntimeError):
    def __init__(self, parent_name):
        self.parent_name = parent_name

    def __str__(self):
        return f"actual_parent should be different than module_execution_stack parent only for torch.nn.ModuleList. Parent: {self.parent_name}"


class MissingPathFromComputationToModuleOutputError(SMPRuntimeError):
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


class MissingPathFromModuleInputToModuleOutputError(SMPRuntimeError):
    def __init__(self, module_name, idx):
        self.module_name = module_name
        self.idx = idx

    def __str__(self):
        return (
            f"Unsupported usecase found during module execution of: {self.module_name}. "
            f"The input with index {self.idx} to the module doesn't have a path to the outputs on which backward is called."
            f" Please remove the unused input, or detach it before passing to the module. {_skip_suggestion}"
        )


class NumParametersNotMatchError(SMPRuntimeError):
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
