# First Party
# Standard Library
from typing import List, Set, Tuple, Union

# Third Party
import torch
from torch.nn.parameter import Parameter

from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.torch.exceptions import SMPUnsupportedError

logger = get_logger()


def log_backward_graph(var_grad_fn, level=0):  # pragma: no cover
    """
    Takes a grad function for a torch tensor
    Helper function to print the sequence of ops executed in the backward pass
    Useful to debug the flow of gradients
    """
    if var_grad_fn is None:
        return
    prefix = "*" * level
    logger.debug(f"{prefix} {var_grad_fn}")
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], "variable")
                logger.debug(f"{prefix} - {n[0]}")
                logger.debug(f"{prefix} - Tensor with grad found: {tensor.shape}")
            except AttributeError as e:
                log_backward_graph(n[0], level + 1)


def _add_nodes_to_explore(nodes, stack, upto_inputs):
    for node in nodes:
        for fn, _ in node.next_functions:
            if fn and fn not in upto_inputs:
                stack.append(fn)


def get_ancestors(
    outputs: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
    checkpoint_nodes_cache: "CheckpointNodesCache" = None,
    upto_inputs: List[torch.Tensor] = None,
) -> Tuple[
    Set["Parameter"],
    Set[Union["SMPInputBackward", "SMPSequentialInputBackward"]],
    Set["SMPParentRecvBackward"],
]:
    """
    Returns three sets of ancestors for a output torch.Tensor, or a list or tuple of torch.Tensors
    Params, SMPInput nodes, and SMPRecv nodes
    """

    if (
        not isinstance(outputs, torch.Tensor)
        and not isinstance(outputs, list)
        and not isinstance(outputs, tuple)
    ):
        raise SMPUnsupportedError("inputs should be Tensor or list or tuple")

    visited = set()
    stack = []
    if isinstance(outputs, torch.Tensor):
        if outputs.requires_grad and outputs.grad_fn:
            stack.append(outputs.grad_fn)
    else:
        for output in outputs:
            if isinstance(output, torch.Tensor) and output.requires_grad and output.grad_fn:
                stack.append(output.grad_fn)

    if upto_inputs:
        upto_inputs_set = set([x.grad_fn for x in upto_inputs if torch.is_tensor(x)])
        stack = list(set(stack).difference(upto_inputs_set))
    else:
        upto_inputs_set = set()

    params, smp_inputs, parent_recvs = set(), set(), set()
    while len(stack) > 0:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            if node.__class__.__name__ == "AccumulateGrad":
                # this value is saved during forward pass of the RubkInput op
                if isinstance(node.variable, torch.nn.parameter.Parameter):
                    params.add(node.variable)
            elif node.__class__.__name__ in ["SMPInputBackward", "SMPSequentialInputBackward"]:
                smp_inputs.add(node)
            else:
                # these are in else block because from here there are potential paths up the graph
                # unlike the above two if blocks
                if node.__class__.__name__ == "SMPParentRecvBackward":
                    parent_recvs.add(node)
                elif (
                    checkpoint_nodes_cache
                    and node.__class__.__name__ == "CheckpointTupledFunctionBackward"
                ):
                    # these nodes are effectively a shortcut for us to traverse the graph using this cache
                    # which is populated by the checkpointing autograd fn
                    # these params are added in ckpt fn
                    # so they can count +1, here we make a set so the effect of new backward call in ckpting fn isn't visible
                    # params.update(checkpoint_nodes_cache.get_params(node))

                    smp_inputs.update(checkpoint_nodes_cache.get_inputs(node))

                    prs = checkpoint_nodes_cache.get_parent_recvs(node)
                    parent_recvs.update(prs)
                    _add_nodes_to_explore(prs, stack, upto_inputs=upto_inputs_set)

                _add_nodes_to_explore([node], stack, upto_inputs=upto_inputs_set)

    return params, smp_inputs, parent_recvs
