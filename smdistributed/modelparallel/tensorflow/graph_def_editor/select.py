# Copyright 2018 IBM. All Rights Reserved.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Various ways of selecting operations and tensors in a graph."""

# Future
from __future__ import absolute_import, division, print_function

# Standard Library
import re
import sys
import textwrap

# Third Party
from six import iteritems, string_types

# First Party
from smdistributed.modelparallel.tensorflow.graph_def_editor import util
from smdistributed.modelparallel.tensorflow.graph_def_editor.graph import Graph
from smdistributed.modelparallel.tensorflow.graph_def_editor.node import Node
from smdistributed.modelparallel.tensorflow.graph_def_editor.tensor import Tensor

if sys.version >= "3":
    from typing import Dict, List, Iterable, Tuple, Union


__all__ = [
    "can_be_regex",
    "make_regex",
    "filter_ts",
    "filter_ts_from_regex",
    "filter_ops",
    "filter_ops_by_optype",
    "filter_ops_from_regex",
    "get_name_scope_ops",
    "check_cios",
    "get_ops_ios",
    "compute_boundary_ts",
    "get_within_boundary_ops",
    "get_forward_walk_ops",
    "get_backward_walk_ops",
    "get_walks_intersection_ops",
    "get_walks_union_ops",
    "select_ops",
    "select_ts",
    "select_ops_and_ts",
    "TreeExpr",
]

_RE_TYPE = type(re.compile(""))


def can_be_regex(obj):
    """Return True if obj can be turned into a regular expression."""
    return isinstance(obj, string_types + (_RE_TYPE,))


def make_regex(obj):
    """Return a compiled regular expression.

  Args:
    obj: a string or a regular expression.
  Returns:
    A compiled regular expression.
  Raises:
    ValueError: if obj could not be converted to a regular expression.
  """
    if not can_be_regex(obj):
        raise ValueError("Expected a string or a regex, got: {}".format(type(obj)))

    if isinstance(obj, string_types):
        return re.compile(obj)
    else:
        return obj


def _get_input_ts(ops):
    """Compute the list of unique input tensors of all the op in ops.

  Args:
    ops: an object convertible to a list of `gde.Node`.
  Returns:
    The list of unique input tensors of all the op in ops.
  Raises:
    TypeError: if ops cannot be converted to a list of `gde.Node`.
  """
    ops = util.make_list_of_op(ops)
    ts = []
    ts_set = set()
    for op in ops:
        for t in op.inputs:
            if t not in ts_set:
                ts.append(t)
                ts_set.add(t)
    return ts


def _get_output_ts(ops):
    """Compute the list of unique output tensors of all the op in ops.

  Args:
    ops: an object convertible to a list of `gde.Node`.
  Returns:
    The list of unique output tensors of all the op in ops.
  Raises:
    TypeError: if ops cannot be converted to a list of `gde.Node`.
  """
    ops = util.make_list_of_op(ops)
    ts = []
    for op in ops:
        ts += op.outputs
    return ts


def filter_ts(ops, positive_filter):
    """Get all the tensors which are input or output of an op in ops.

  Args:
    ops: an object convertible to a list of `gde.Node`.
    positive_filter: a function deciding whether to keep a tensor or not.
      If `True`, all the tensors are returned.
  Returns:
    A list of `gde.Tensor`.
  Raises:
    TypeError: if ops cannot be converted to a list of `gde.Node`.
  """
    ops = util.make_list_of_op(ops)
    ts = _get_input_ts(ops)
    util.concatenate_unique(ts, _get_output_ts(ops))
    if positive_filter is not True:
        ts = [t for t in ts if positive_filter(t)]
    return ts


def filter_ts_from_regex(ops, regex):
    r"""Get all the tensors linked to ops that match the given regex.

  Args:
    ops: an object convertible to a list of `gde.Node`.
    regex: a regular expression matching the tensors' name.
      For example, "^foo(/.*)?:\d+$" will match all the tensors in the "foo"
      scope.
  Returns:
    A list of `gde.Tensor`.
  Raises:
    TypeError: if ops cannot be converted to a list of `gde.Node`.
  """
    ops = util.make_list_of_op(ops)
    regex_obj = make_regex(regex)
    return filter_ts(ops, positive_filter=lambda op: regex_obj.search(op.name))


def filter_ops(ops, positive_filter):
    """Get the ops passing the given filter.

  Args:
    ops: an object convertible to a list of `gde.Node`.
    positive_filter: a function deciding where to keep an operation or not.
      If True, all the operations are returned.
  Returns:
    A list of selected `gde.Node`.
  Raises:
    TypeError: if ops cannot be converted to a list of `gde.Node`.
  """
    ops = util.make_list_of_op(ops)
    if positive_filter is not True:  # pylint: disable=g-explicit-bool-comparison
        ops = [op for op in ops if positive_filter(op)]
    return ops


def filter_ops_by_optype(
    ops, op_types  # type: Union[str, List[str], Tuple[str]]
):
    # type: (...) -> List[node.Node]
    """
  Filter out all ops that do not have op types in a provided list of types.

  Args:
    ops: an object convertible to a list of `gde.Node`.
    op_types: Either a single op type string or a list/tuple of them

  Returns a list of the `gde.Node`s that have one of the indicated optypes
  """
    if isinstance(op_types, string_types):
        op_types = [op_types]
    return filter_ops(ops, lambda n: n.op_type in op_types)


def filter_ops_from_regex(ops, regex):
    """Get all the operations that match the given regex.

  Args:
    ops: an object convertible to a list of `gde.Node`.
    regex: a regular expression matching the operation's name.
      For example, `"^foo(/.*)?$"` will match all the operations in the "foo"
      scope.
  Returns:
    A list of `gde.Node`.
  Raises:
    TypeError: if ops cannot be converted to a list of `gde.Node`.
  """
    ops = util.make_list_of_op(ops)
    regex_obj = make_regex(regex)
    return filter_ops(ops, lambda op: regex_obj.search(op.name))


def get_name_scope_ops(ops, scope):
    """Get all the operations under the given scope path.

  Args:
    ops: an object convertible to a list of `gde.Node`.
    scope: a scope path.
  Returns:
    A list of `gde.Node`.
  Raises:
    TypeError: if ops cannot be converted to a list of `gde.Node`.
  """
    if scope and scope[-1] == "/":
        scope = scope[:-1]
    return filter_ops_from_regex(ops, "^{}(/.*)?$".format(scope))


def check_cios(control_inputs=False, control_outputs=None, control_ios=None):
    """Do various check on control_inputs and control_outputs.

  Args:
    control_inputs: A boolean indicating whether control inputs are enabled.
    control_outputs: An instance of util.ControlOutputs or None. If not None,
      control outputs are enabled.
    control_ios:  An instance of util.ControlOutputs or None. If not None, both
      control inputs and control outputs are enabled. This is equivalent to set
      control_inputs to True and control_outputs to the util.ControlOutputs
      instance.
  Returns:
    A tuple `(control_inputs, control_outputs)` where:
      `control_inputs` is a boolean indicating whether to use control inputs.
      `control_outputs` is an instance of util.ControlOutputs or None
  Raises:
    ValueError: if control_inputs is an instance of util.ControlOutputs but
      control_outputs is not None
    TypeError: if control_outputs is not None and is not a util.ControlOutputs.
  """
    if control_ios is not None:
        if not isinstance(control_ios, util.ControlOutputs):
            raise TypeError("Expected a gde.ControlOutputs, got: {}".format(type(control_ios)))
        if control_outputs is not None:
            raise ValueError("control_outputs should be None when using control_ios.")
        control_inputs = True
        control_outputs = control_ios
    elif control_outputs is not None:
        if not isinstance(control_outputs, util.ControlOutputs):
            raise TypeError("Expected a gde.ControlOutputs, got: {}".format(type(control_outputs)))

    if control_outputs is not None:
        control_outputs.update()
    return control_inputs, control_outputs


def get_ops_ios(ops, control_inputs=False, control_outputs=None, control_ios=None):
    """Return all the `gde.Node` which are connected to an op in ops.

  Args:
    ops: an object convertible to a list of `gde.Node`.
    control_inputs: A boolean indicating whether control inputs are enabled.
    control_outputs: An instance of `util.ControlOutputs` or `None`. If not
      `None`, control outputs are enabled.
    control_ios:  An instance of `util.ControlOutputs` or `None`. If not `None`,
      both control inputs and control outputs are enabled. This is equivalent to
      set `control_inputs` to `True` and `control_outputs` to the
      `util.ControlOutputs` instance.
  Returns:
    All the `gde.Node` surrounding the given ops.
  Raises:
    TypeError: if `ops` cannot be converted to a list of `gde.Node`.
  """
    control_inputs, control_outputs = check_cios(control_inputs, control_outputs, control_ios)
    ops = util.make_list_of_op(ops)
    res = []
    for op in ops:
        util.concatenate_unique(res, [t.op for t in op.inputs])
        for t in op.outputs:
            util.concatenate_unique(res, t.consumers())
        if control_outputs is not None:
            util.concatenate_unique(res, control_outputs.get(op))
        if control_inputs:
            util.concatenate_unique(res, op.control_inputs)
    return res


def compute_boundary_ts(ops):
    """Compute the tensors at the boundary of a set of ops.

  This function looks at all the tensors connected to the given ops (in/out)
  and classify them into three categories:
  1) input tensors: tensors whose generating operation is not in ops.
  2) output tensors: tensors whose consumer operations are not in ops
  3) inside tensors: tensors which are neither input nor output tensors.

  Note that a tensor can be both an inside tensor and an output tensor if it is
  consumed by operations both outside and inside of `ops`.

  Args:
    ops: an object convertible to a list of `gde.Node`
  Returns:
    A tuple `(outside_input_ts, outside_output_ts, inside_ts)` where:
      `outside_input_ts` is a Python list of input tensors;
      `outside_output_ts` is a python list of output tensors;
      `inside_ts` is a python list of inside tensors.
    Since a tensor can be both an inside tensor and an output tensor,
    `outside_output_ts` and `inside_ts` might intersect.
  Raises:
    TypeError: if ops cannot be converted to a list of `gde.Node`.
  """
    ops = util.make_list_of_op(ops)
    input_ts = _get_input_ts(ops)
    output_ts = _get_output_ts(ops)
    output_ts_set = frozenset(output_ts)
    ops_set = frozenset(ops)

    # Compute inside tensors.
    inside_ts = []
    only_inside_ts = []
    for t in input_ts:
        # Skip if the input tensor is not also an output tensor.
        if t not in output_ts_set:
            continue
        # Mark as "inside".
        inside_ts.append(t)
        # Mark as "only inside" if the tensor is not both inside and output.
        consumers = frozenset(t.consumers())
        if consumers - ops_set:
            continue
        only_inside_ts.append(t)

    inside_ts_set = frozenset(inside_ts)
    only_inside_ts_set = frozenset(only_inside_ts)
    outside_output_ts = [t for t in output_ts if t not in only_inside_ts_set]
    outside_input_ts = [t for t in input_ts if t not in inside_ts_set]
    return outside_input_ts, outside_output_ts, inside_ts


def get_within_boundary_ops(
    ops,
    seed_ops,
    boundary_ops=(),
    inclusive=True,
    control_inputs=False,
    control_outputs=None,
    control_ios=None,
):
    """Return all the `gde.Node` within the given boundary.

  Args:
    ops: an object convertible to a list of `gde.Node`. those ops define the
      set in which to perform the operation (if a `gde.Graph` is given, it
      will be converted to the list of all its operations).
    seed_ops: the operations from which to start expanding.
    boundary_ops: the ops forming the boundary.
    inclusive: if `True`, the result will also include the boundary ops.
    control_inputs: A boolean indicating whether control inputs are enabled.
    control_outputs: An instance of `util.ControlOutputs` or `None`. If not
      `None`, control outputs are enabled.
    control_ios:  An instance of `util.ControlOutputs` or `None`. If not
      `None`, both control inputs and control outputs are enabled. This is
      equivalent to set control_inputs to True and control_outputs to
      the `util.ControlOutputs` instance.
  Returns:
    All the `gde.Node` surrounding the given ops.
  Raises:
    TypeError: if `ops` or `seed_ops` cannot be converted to a list of
      `gde.Node`.
    ValueError: if the boundary is intersecting with the seeds.
  """
    control_inputs, control_outputs = check_cios(control_inputs, control_outputs, control_ios)
    ops = util.make_list_of_op(ops)
    seed_ops = util.make_list_of_op(seed_ops, allow_graph=False)
    boundary_ops = set(util.make_list_of_op(boundary_ops))
    res = set(seed_ops)
    if boundary_ops & res:
        raise ValueError("Boundary is intersecting with the seeds.")
    wave = set(seed_ops)
    while wave:
        new_wave = set()
        ops_io = get_ops_ios(wave, control_inputs, control_outputs)
        for op in ops_io:
            if op in res:
                continue
            if op in boundary_ops:
                if inclusive:
                    res.add(op)
            else:
                new_wave.add(op)
        res.update(new_wave)
        wave = new_wave
    return [op for op in ops if op in res]


def get_forward_walk_ops(
    seed_ops,
    inclusive=True,
    within_ops=None,
    within_ops_fn=None,
    stop_at_ts=(),
    control_outputs=None,
):
    """Do a forward graph walk and return all the visited ops.

  Args:
    seed_ops: an iterable of operations from which the forward graph
      walk starts. If a list of tensors is given instead, the seed_ops are set
      to be the consumers of those tensors.
    inclusive: if True the given seed_ops are also part of the resulting set.
    within_ops: an iterable of `gde.Node` within which the search is
      restricted. If `within_ops` is `None`, the search is performed within
      the whole graph.
    within_ops_fn: if provided, a function on ops that should return True iff
      the op is within the graph traversal. This can be used along within_ops,
      in which case an op is within if it is also in within_ops.
    stop_at_ts: an iterable of tensors at which the graph walk stops.
    control_outputs: a `util.ControlOutputs` instance or None.
      If not `None`, it will be used while walking the graph forward.
  Returns:
    A Python set of all the `gde.Node` ahead of `seed_ops`.
  Raises:
    TypeError: if `seed_ops` or `within_ops` cannot be converted to a list of
      `gde.Node`.
  """
    _, control_outputs = check_cios(False, control_outputs)
    if not util.is_iterable(seed_ops):
        seed_ops = [seed_ops]
    if not seed_ops:
        return []
    if isinstance(seed_ops[0], Tensor):
        ts = util.make_list_of_t(seed_ops, allow_graph=False)
        seed_ops = util.get_consuming_ops(ts)
    else:
        seed_ops = util.make_list_of_op(seed_ops, allow_graph=False)

    seed_ops = frozenset(seed_ops)
    stop_at_ts = frozenset(util.make_list_of_t(stop_at_ts))
    if within_ops:
        within_ops = util.make_list_of_op(within_ops, allow_graph=False)
        within_ops = frozenset(within_ops)
        seed_ops &= within_ops

    def is_within(operator):
        return (within_ops is None or operator in within_ops) and (
            within_ops_fn is None or within_ops_fn(operator)
        )

    result = list(seed_ops)
    wave = set(seed_ops)
    while wave:
        new_wave = set()
        for op in wave:
            for new_t in op.outputs:
                if new_t in stop_at_ts:
                    continue
                for new_op in new_t.consumers():
                    if new_op not in result and is_within(new_op):
                        new_wave.add(new_op)
            if control_outputs is not None:
                for new_op in control_outputs.get(op):
                    if new_op not in result and is_within(new_op):
                        new_wave.add(new_op)
        util.concatenate_unique(result, new_wave)
        wave = new_wave
    if not inclusive:
        result = [op for op in result if op not in seed_ops]
    return result


def get_backward_walk_ops(
    seed_ops,
    inclusive=True,
    within_ops=None,
    within_ops_fn=None,
    stop_at_ts=(),
    control_inputs=False,
):
    """Do a backward graph walk and return all the visited ops.

  Args:
    seed_ops: an iterable of operations from which the backward graph
      walk starts. If a list of tensors is given instead, the seed_ops are set
      to be the generators of those tensors.
    inclusive: if True the given seed_ops are also part of the resulting set.
    within_ops: an iterable of `gde.Node` within which the search is
      restricted. If `within_ops` is `None`, the search is performed within
      the whole graph.
    within_ops_fn: if provided, a function on ops that should return True iff
      the op is within the graph traversal. This can be used along within_ops,
      in which case an op is within if it is also in within_ops.
    stop_at_ts: an iterable of tensors at which the graph walk stops.
    control_inputs: if True, control inputs will be used while moving backward.
  Returns:
    A Python set of all the `gde.Node` behind `seed_ops`.
  Raises:
    TypeError: if `seed_ops` or `within_ops` cannot be converted to a list of
      `gde.Node`.
  """
    if not util.is_iterable(seed_ops):
        seed_ops = [seed_ops]
    if not seed_ops:
        return []
    if isinstance(seed_ops[0], Tensor):
        ts = util.make_list_of_t(seed_ops, allow_graph=False)
        seed_ops = util.get_generating_ops(ts)
    else:
        seed_ops = util.make_list_of_op(seed_ops, allow_graph=False)

    stop_at_ts = frozenset(util.make_list_of_t(stop_at_ts))
    seed_ops = frozenset(util.make_list_of_op(seed_ops))
    if within_ops:
        within_ops = util.make_list_of_op(within_ops, allow_graph=False)
        within_ops = frozenset(within_ops)
        seed_ops &= within_ops

    def is_within(operator):
        return (within_ops is None or operator in within_ops) and (
            within_ops_fn is None or within_ops_fn(operator)
        )

    result = list(seed_ops)
    wave = set(seed_ops)
    while wave:
        new_wave = set()
        for op in wave:
            for new_t in op.inputs:
                if new_t in stop_at_ts:
                    continue
                if new_t.op not in result and is_within(new_t.op):
                    new_wave.add(new_t.op)
            if control_inputs:
                for new_op in op.control_inputs:
                    if new_op not in result and is_within(new_op):
                        new_wave.add(new_op)
        util.concatenate_unique(result, new_wave)
        wave = new_wave
    if not inclusive:
        result = [op for op in result if op not in seed_ops]
    return result


def get_walks_intersection_ops(
    forward_seed_ops,
    backward_seed_ops,
    forward_inclusive=True,
    backward_inclusive=True,
    within_ops=None,
    within_ops_fn=None,
    control_inputs=False,
    control_outputs=None,
    control_ios=None,
):
    """Return the intersection of a forward and a backward walk.

  Args:
    forward_seed_ops: an iterable of operations from which the forward graph
      walk starts. If a list of tensors is given instead, the seed_ops are set
      to be the consumers of those tensors.
    backward_seed_ops: an iterable of operations from which the backward graph
      walk starts. If a list of tensors is given instead, the seed_ops are set
      to be the generators of those tensors.
    forward_inclusive: if True the given forward_seed_ops are also part of the
      resulting set.
    backward_inclusive: if True the given backward_seed_ops are also part of the
      resulting set.
    within_ops: an iterable of `gde.Node` within which the search is
      restricted. If within_ops is None, the search is performed within
      the whole graph.
    within_ops_fn: if provided, a function on ops that should return True iff
      the op is within the graph traversal. This can be used along within_ops,
      in which case an op is within if it is also in within_ops.
    control_inputs: A boolean indicating whether control inputs are enabled.
    control_outputs: An instance of util.ControlOutputs or None. If not None,
      control outputs are enabled.
    control_ios:  An instance of util.ControlOutputs or None. If not None, both
      control inputs and control outputs are enabled. This is equivalent to set
      control_inputs to True and control_outputs to the util.ControlOutputs
      instance.
  Returns:
    A Python set of all the `gde.Node` in the intersection of a forward and a
      backward walk.
  Raises:
    TypeError: if `forward_seed_ops` or `backward_seed_ops` or `within_ops`
      cannot be converted to a list of `gde.Node`.
  """
    control_inputs, control_outputs = check_cios(control_inputs, control_outputs, control_ios)
    forward_ops = get_forward_walk_ops(
        forward_seed_ops,
        inclusive=forward_inclusive,
        within_ops=within_ops,
        within_ops_fn=within_ops_fn,
        control_outputs=control_outputs,
    )
    backward_ops = get_backward_walk_ops(
        backward_seed_ops,
        inclusive=backward_inclusive,
        within_ops=within_ops,
        within_ops_fn=within_ops_fn,
        control_inputs=control_inputs,
    )
    return [op for op in forward_ops if op in backward_ops]


def get_walks_union_ops(
    forward_seed_ops,
    backward_seed_ops,
    forward_inclusive=True,
    backward_inclusive=True,
    within_ops=None,
    within_ops_fn=None,
    control_inputs=False,
    control_outputs=None,
    control_ios=None,
):
    """Return the union of a forward and a backward walk.

  Args:
    forward_seed_ops: an iterable of operations from which the forward graph
      walk starts. If a list of tensors is given instead, the seed_ops are set
      to be the consumers of those tensors.
    backward_seed_ops: an iterable of operations from which the backward graph
      walk starts. If a list of tensors is given instead, the seed_ops are set
      to be the generators of those tensors.
    forward_inclusive: if True the given forward_seed_ops are also part of the
      resulting set.
    backward_inclusive: if True the given backward_seed_ops are also part of the
      resulting set.
    within_ops: restrict the search within those operations. If within_ops is
      None, the search is done within the whole graph.
    within_ops_fn: if provided, a function on ops that should return True iff
      the op is within the graph traversal. This can be used along within_ops,
      in which case an op is within if it is also in within_ops.
    control_inputs: A boolean indicating whether control inputs are enabled.
    control_outputs: An instance of util.ControlOutputs or None. If not None,
      control outputs are enabled.
    control_ios:  An instance of util.ControlOutputs or None. If not None, both
      control inputs and control outputs are enabled. This is equivalent to set
      control_inputs to True and control_outputs to the util.ControlOutputs
      instance.
  Returns:
    A Python set of all the `gde.Node` in the union of a forward and a
      backward walk.
  Raises:
    TypeError: if forward_seed_ops or backward_seed_ops or within_ops cannot be
      converted to a list of `gde.Node`.
  """
    control_inputs, control_outputs = check_cios(control_inputs, control_outputs, control_ios)
    forward_ops = get_forward_walk_ops(
        forward_seed_ops,
        inclusive=forward_inclusive,
        within_ops=within_ops,
        within_ops_fn=within_ops_fn,
        control_outputs=control_outputs,
    )
    backward_ops = get_backward_walk_ops(
        backward_seed_ops,
        inclusive=backward_inclusive,
        within_ops=within_ops,
        within_ops_fn=within_ops_fn,
        control_inputs=control_inputs,
    )
    return util.concatenate_unique(forward_ops, backward_ops)


def select_ops(*args, **kwargs):
    """Helper to select operations.

  Args:
    *args: list of 1) regular expressions (compiled or not) or 2) (array of)
      `gde.Node`. `gde.Tensor` instances are silently ignored.
    **kwargs: 'graph': `gde.Graph` in which to perform the regex query.This is
      required when using regex.
      'positive_filter': an elem if selected only if `positive_filter(elem)` is
        `True`. This is optional.
      'restrict_ops_regex': a regular expression is ignored if it doesn't start
        with the substring "(?#ops)".
  Returns:
    A list of `gde.Node`.
  Raises:
    TypeError: if the optional keyword argument graph is not a `gde.Graph`
      or if an argument in args is not an (array of) `gde.Node`
      or an (array of) `gde.Tensor` (silently ignored) or a string
      or a regular expression.
    ValueError: if one of the keyword arguments is unexpected or if a regular
      expression is used without passing a graph as a keyword argument.
  """
    # get keywords arguments
    g = None
    positive_filter = None
    restrict_ops_regex = False
    for k, v in iteritems(kwargs):
        if k == "graph":
            g = v
            if g is not None and not isinstance(g, Graph):
                raise TypeError("Expected a gde.Graph, got: {}".format(type(g)))
        elif k == "positive_filter":
            positive_filter = v
        elif k == "restrict_ops_regex":
            restrict_ops_regex = v
        elif k == "restrict_ts_regex":
            pass
        else:
            raise ValueError("Wrong keywords argument: {}.".format(k))

    ops = []

    for arg in args:
        if can_be_regex(arg):
            if g is None:
                raise ValueError("Use the keyword argument 'graph' to use regex.")
            regex = make_regex(arg)
            if regex.pattern.startswith("(?#ts)"):
                continue
            if restrict_ops_regex and not regex.pattern.startswith("(?#ops)"):
                continue
            ops_ = filter_ops_from_regex(g, regex)
            for op_ in ops_:
                if op_ not in ops:
                    if positive_filter is None or positive_filter(op_):
                        ops.append(op_)
        else:
            ops_aux = util.make_list_of_op(arg, ignore_ts=True)
            if positive_filter is not None:
                ops_aux = [op for op in ops_aux if positive_filter(op)]
            ops_aux = [op for op in ops_aux if op not in ops]
            ops += ops_aux

    return ops


def select_ts(*args, **kwargs):
    """Helper to select tensors.

  Args:
    *args: list of 1) regular expressions (compiled or not) or 2) (array of)
      `gde.Tensor`. `gde.Node` instances are silently ignored.
    **kwargs: 'graph': `gde.Graph` in which to perform the regex query.This is
      required when using regex.
      'positive_filter': an elem if selected only if `positive_filter(elem)` is
        `True`. This is optional.
      'restrict_ts_regex': a regular expression is ignored if it doesn't start
        with the substring "(?#ts)".
  Returns:
    A list of `gde.Tensor`.
  Raises:
    TypeError: if the optional keyword argument graph is not a `gde.Graph`
      or if an argument in args is not an (array of) `gde.Tensor`
      or an (array of) `gde.Node` (silently ignored) or a string
      or a regular expression.
    ValueError: if one of the keyword arguments is unexpected or if a regular
      expression is used without passing a graph as a keyword argument.
  """
    # get keywords arguments
    cur_graph = None
    positive_filter = None
    restrict_ts_regex = False
    for k, v in iteritems(kwargs):
        if k == "graph":
            cur_graph = v
            if cur_graph is not None and not isinstance(cur_graph, Graph):
                raise TypeError("Expected a gde.Graph, got {}".format(type(cur_graph)))
        elif k == "positive_filter":
            positive_filter = v
        elif k == "restrict_ts_regex":
            restrict_ts_regex = v
        elif k == "restrict_ops_regex":
            pass
        else:
            raise ValueError("Wrong keywords argument: {}.".format(k))

    ts = []

    for arg in args:
        if can_be_regex(arg):
            if cur_graph is None:
                raise ValueError("Use the keyword argument 'graph' to use regex.")
            regex = make_regex(arg)
            if regex.pattern.startswith("(?#ops)"):
                continue
            if restrict_ts_regex and not regex.pattern.startswith("(?#ts)"):
                continue
            ts_ = filter_ts_from_regex(cur_graph, regex)
            for t_ in ts_:
                if t_ not in ts:
                    if positive_filter is None or positive_filter(t_):
                        ts.append(t_)
        else:
            ts_aux = util.make_list_of_t(arg, ignore_ops=True)
            if positive_filter is not None:
                ts_aux = [t for t in ts_aux if positive_filter(t)]
            ts_aux = [t for t in ts_aux if t not in ts]
            ts += ts_aux

    return ts


def select_ops_and_ts(*args, **kwargs):
    """Helper to select operations and tensors.

  Args:
    *args: list of 1) regular expressions (compiled or not) or 2) (array of)
      `gde.Node` 3) (array of) `gde.Tensor`. Regular expressions matching
      tensors must start with the comment `"(?#ts)"`, for instance:
      `"(?#ts)^foo/.*"`.
    **kwargs: 'graph': `gde.Graph` in which to perform the regex query.This is
      required when using regex.
      'positive_filter': an elem if selected only if `positive_filter(elem)` is
        `True`. This is optional.
  Returns:
    A tuple `(ops, ts)` where:
      `ops` is a list of `gde.Node`, and
      `ts` is a list of `gde.Tensor`
  Raises:
    TypeError: if the optional keyword argument graph is not a `gde.Graph`
      or if an argument in args is not an (array of) `gde.Tensor`
      or an (array of) `gde.Node` or a string or a regular expression.
    ValueError: if one of the keyword arguments is unexpected or if a regular
      expression is used without passing a graph as a keyword argument.
  """
    ops = select_ops(*args, restrict_ops_regex=False, **kwargs)
    ts = select_ts(*args, restrict_ts_regex=True, **kwargs)
    return ops, ts


class TreeExpr(object):
    """
  Selection predicate that captures tree-shaped subgraphs a la the Graph
  Transform Tool's `OpTypePattern` DSL.
  """

    def __init__(
        self,
        alias=None,  # type: str
        op=None,  # type: str
        optional=None,  # type: bool
        inputs=None,  # type: Union["TreeExpr", Iterable["TreeExpr"]]
    ):
        """
    Args:
      alias: Name to which to bind the core node of any matches of this
       expression
      op: Regular expression to apply to the op_type field of the root
        node of any potential match.
        A value of None here means "match any op_type value"
      inputs: List of predicates on input nodes of the root node in the
        matching tree. If the list is shorter than the number of inputs of
        a node, only filters on the predicates provided.
        A value of None means "don't filter on inputs"
        Can also be single predicate, in which case it will be converted to a
        one-element tuple.
      optional: If True, match expressions that do not include this node but
        do include any child nodes. Equivalent to "?" in a regular expression.
        Defaults to False.
    """
        if isinstance(inputs, TreeExpr):
            inputs = (inputs,)
        if optional and len(inputs) > 1:
            raise ValueError("'optional' flag only supported with zero or one " "inputs")
        self._alias = alias
        self._op_type_regex = None if op is None else make_regex(op)
        self._input_exprs = inputs
        self._optional = optional

    def __repr__(self):
        ret = "TreeExpr("
        args = []
        if self._alias is not None:
            args.append("alias='{}'".format(self._alias))
        if self._op_type_regex is not None:
            args.append("op='{}'".format(self._op_type_regex))
        if self._optional is not None:
            args.append("optional={}".format(self._optional))
        ret += ", ".join(args)
        if self._input_exprs is not None:
            if len(args) > 0:
                ret += ", "
            ret += "inputs=(\n"
            ret += ",\n".join([textwrap.indent(repr(i), "  ") for i in self._input_exprs])
            ret += ")"
        ret += ")"
        return ret

    def __str__(self):
        return repr(self)

    def eval_from(
        self, potential_root_node  # type: Node
    ):
        # type: (...) -> Dict[str, Node]
        """
    Evaluate a tree expression starting from a potential root node.

    Args:
      potential_root_node: Node to test for a match of the expression

    Returns:
      Returns a Dict mapping node alias to node instance if there is a match
      at the indicated root.
      Returns None if there is no match starting from the indicated root.
    """
        if potential_root_node.graph is None:
            raise ValueError(
                "TreeExpr evaluation hit a node that has been removed "
                "from the graph.\n"
                "Node is {}, and expression is:\n"
                "{}".format(potential_root_node, self)
            )
        result_at_root = self._eval_without_optional(potential_root_node)
        if result_at_root is not None:
            return result_at_root
        elif self._optional:
            # No match at this root, but maybe that's because this optional node of
            # the tree isn't part of the current match.
            return self._input_exprs[0].eval_from(potential_root_node)
        else:
            return None

    def _eval_without_optional(
        self, potential_root_node  # type: Node
    ):
        """
    Handle every part of the expression except the "optional" clause.
    """
        result = {}
        prn = potential_root_node
        if self._op_type_regex is not None and self._op_type_regex.fullmatch(prn.op_type) is None:
            # Failed to match on op_type
            return None
        if self._input_exprs is not None:
            num_exprs = len(self._input_exprs)
            inputs = prn.inputs
            if num_exprs > len(inputs):
                raise ValueError(
                    "Attempted to filter {} inputs of node {}, "
                    "but node only has {} inputs."
                    "".format(num_exprs, prn, len(inputs))
                )
            for i in range(num_exprs):
                input_result = self._input_exprs[i].eval_from(inputs[i].node)
                if input_result is None:
                    # Failed to match on input
                    return None
                result.update(input_result)
        # If we get here, all predicates have passed
        if self._alias is not None:
            result[self._alias] = prn
        return result
