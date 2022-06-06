# Third Party
# Standard Library
import collections
import copy
from contextlib import contextmanager
from enum import Enum
from typing import Any, Dict, List, NamedTuple

import torch

# First Party
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.torch.core import core, rank
from smdistributed.modelparallel.torch.module_manager import TensorModuleInfo
from smdistributed.modelparallel.torch.utils import (
    get_tensor_size,
    is_instance_namedtuple,
    map_structure,
    rmsg,
)


"""
Only attributes of an object which don't start with _ are serialized
"""


class TensorTransmission(NamedTuple):
    tensor: torch.Tensor
    dests: List[int]
    link_id: int


class TensorStub(NamedTuple):
    """
    Represents a tensor in serialized messages. Can be used at the destination to reconstruct
    the original tensor through (src, link_id) pair.
    """

    tensor_index: int
    dtype: torch.dtype
    shape: torch.Size
    requires_grad: bool
    device: str
    is_dummy: bool
    src: int
    link_id: int
    module_info: TensorModuleInfo

    @classmethod
    def create_non_dummy_version(cls, stub, index):
        return cls(
            index,
            stub.dtype,
            stub.shape,
            stub.requires_grad,
            stub.device,
            False,
            stub.src,
            stub.link_id,
            stub.module_info,
        )


class DummyTensorRegistry:
    """
    Keeps track of dummy tensors and their associated stubs. A dummy tensor existing in
    a rank would mean that a child has made a direct transmission to another child. When
    the parent sends a request to the next child which involves this dummy tensor, we pick up
    the metadata needed to stubify the dummy tensor from this registry, since the original
    (src, link_id) pair would be required to reconstruct the tensor at the next child.
    """

    def __init__(self):
        self._registry = {}
        self._handles = {}

    def _add_hook(self, tensor):
        if tensor.requires_grad:
            assert (
                hasattr(tensor, "_smp_is_dummy") and tensor._smp_is_dummy
            ), "Attempt to add error hook to non-dummy tensor."

            # add a hook to raise an error if tensor is used in backward pass, which would
            # imply that it has been used in forward pass too. when we actually need to call
            # backward starting from this tensor, we first confirm that there are no other ops
            # until we hit SMPParentRecvBackward, and then remove the hook so backward pass
            # can continue.
            def raise_error(x):
                raise RuntimeError(
                    "Model is not supported by fast mode, please set 'fast_mode' to False."
                )

            handle = tensor.register_hook(lambda x: raise_error(x))
            self._handles[id(tensor)] = handle

    def remove_hook(self, tensor):
        if tensor.requires_grad and id(tensor) in self._handles:
            self._handles[id(tensor)].remove()
            del self._handles[id(tensor)]

    def put(self, tensor, stub):
        assert hasattr(tensor, "_smp_is_dummy") and tensor._smp_is_dummy
        get_logger().debug(
            rmsg(f"Saving dummy tensor with id {id(tensor)} for link {stub.link_id}")
        )

        self._registry[id(tensor)] = stub
        self._add_hook(tensor)

    def get(self, tensor):
        assert hasattr(tensor, "_smp_is_dummy") and tensor._smp_is_dummy

        get_logger().debug(rmsg(f"Fetching dummy tensor with id {id(tensor)}"))
        return self._registry[id(tensor)]

    def update(self, old_tensor, new_tensor):
        self._registry[id(new_tensor)] = self._registry[id(old_tensor)]
        get_logger().debug(
            rmsg(f"Updating dummy tensor from id {id(old_tensor)} to {id(new_tensor)}")
        )
        del self._registry[id(old_tensor)]
        self.remove_hook(old_tensor)
        self._add_hook(new_tensor)

    def clear(self):
        self._registry.clear()
        for handle in self._handles.values():
            handle.remove()
        self._handles.clear()


class SerializationManager:
    def __init__(self):
        self.dummy_registry = DummyTensorRegistry()
        # used for static mode for cached link ids
        self.tensor_index_to_link_id = None

    def prepare_dummy_tensors_for_backward(self, tensors):
        def _remove_hook(tensor):
            if (
                isinstance(tensor, torch.Tensor)
                and hasattr(tensor, "_smp_is_dummy")
                and tensor._smp_is_dummy
            ):
                # make sure there are no operations (in-place or otherwise) inserted on the dummy at the parent
                if self._has_ops_on_dummy(tensor):
                    raise RuntimeError(
                        "Model is not supported by fast mode, please set 'fast_mode' to False."
                    )
                self.dummy_registry.remove_hook(tensor)

        return map_structure(_remove_hook, tensors)

    def _has_ops_on_dummy(self, tensor):
        if not tensor.requires_grad:
            return True
        grad_fn = tensor.grad_fn
        if grad_fn.__class__.__name__ == "SMPParentRecvBackward":
            return False

        if grad_fn.__class__.__name__ != "CloneBackward":
            return True
        if grad_fn.next_functions[0][0].__class__.__name__ != "SMPParentRecvBackward":
            return True
        return False

    def update_dummy(self, old_tensor, new_tensor):
        self.dummy_registry.update(old_tensor, new_tensor)

    def reset(self):
        self.clear_minibatch_state()

    def clear_minibatch_state(self):
        self.dummy_registry.clear()

    @contextmanager
    def catch_and_raise_for_large_object(self, obj):
        try:
            yield
        except RecursionError:
            raise RuntimeError(
                f"Recursion depth exceeded while serializing object of type {obj.__class__.__name__}. This is probably caused by a large, non-Tensor object being passed as an argument to a module, or being returned from a module or smp.step."
            )

    def serialize(self, obj: Any, c2c_possible: bool, peers: List[int], for_offload: bool = False):
        tx_list = []
        with self.catch_and_raise_for_large_object(obj):
            obj_stripped_of_tensors, seen_class_types = self._replace_tensors_with_stubs(
                obj, {}, tx_list, c2c_possible, peers, for_offload
            )

            # the above will mutate the class types in-place, so if we have encountered any non-Tensor
            # class types, we need to deepcopy the serialized object for transmission, and then reconstruct
            # the original object locally
            if seen_class_types:
                serialized_cpy = copy.deepcopy(obj_stripped_of_tensors)
                self.deserialize(obj_stripped_of_tensors, [t.tensor for t in tx_list])
                return serialized_cpy, tx_list
            else:
                return obj_stripped_of_tensors, tx_list

    def deserialize(self, stubbed_obj, tensors: List[torch.Tensor]):
        def replace_tensor(stub, tensor_list):
            return tensor_list[stub.tensor_index]

        with self.catch_and_raise_for_large_object(stubbed_obj):
            return self._traverse_object(stubbed_obj, tensors, replace_tensor)

    def extract_stubs(self, stubbed_obj):
        def add_stub_to_list(obj, stub_list):
            stub_list.append(obj)
            return obj

        stub_list = []
        with self.catch_and_raise_for_large_object(stubbed_obj):
            self._traverse_object(stubbed_obj, stub_list, add_stub_to_list)
        return stub_list

    def _traverse_object(self, obj, tensor_list, callback):
        if isinstance(obj, TensorStub):
            return callback(obj, tensor_list)

        if isinstance(obj, (bool, str, bytes, bytearray, int, float, Enum)):
            return obj

        if isinstance(obj, (list, tuple, set)):
            l = []
            for item in obj:
                res = self._traverse_object(item, tensor_list, callback)
                l.append(res)
            if is_instance_namedtuple(obj):
                # handling namedtuples
                cast_out = obj.__class__(*l)
            else:
                cast_out = obj.__class__(l)
            return cast_out

        if isinstance(obj, dict):
            # Recreate the instance based on
            # type of obj, and insert keys in the
            # order present in obj
            # Works only for mutable dicts
            instance_type = type(obj)
            if instance_type == collections.defaultdict:
                d = collections.defaultdict(obj.default_factory)
            else:
                d = instance_type()

            # Iteration order is deterministic only on/after python3.7 / cpython3.6
            # This should be fine for Rubik on SM
            for sk, sv in obj.items():
                key = self._traverse_object(sk, tensor_list, callback)
                value = self._traverse_object(sv, tensor_list, callback)
                d[key] = value
            return d

        # class types
        for attr in dir(obj):
            obj_attr = getattr(obj, attr)
            if not callable(obj_attr) and not attr.startswith("__"):
                value = self._traverse_object(obj_attr, tensor_list, callback)
                setattr(obj, attr, value)
        return obj

    def _replace_tensors_with_stubs(
        self,
        obj,
        memo: Dict,
        tx_list: List[TensorTransmission],
        c2c_possible: bool,
        peers: List[int],
        for_offload: bool = False,
    ):
        if id(obj) in memo:
            return memo[id(obj)], False

        if isinstance(obj, (bool, str, bytes, bytearray, int, float, Enum)):
            return obj, False

        if isinstance(obj, (list, tuple, set)):
            l = []
            seen_class_type = False
            for item in obj:
                res, ret_seen_cls_type = self._replace_tensors_with_stubs(
                    item, memo, tx_list, c2c_possible, peers, for_offload
                )
                seen_class_type = seen_class_type or ret_seen_cls_type
                l.append(res)
                memo[id(item)] = res
            if is_instance_namedtuple(obj):
                # handling namedtuples
                cast_out = obj.__class__(*l)
            else:
                cast_out = obj.__class__(l)
            memo[id(obj)] = cast_out
            return cast_out, seen_class_type

        if isinstance(obj, dict):
            # Obtain instance type from object and create an instance
            # of the instance type.
            # Insert keys in the order it is present in the obj
            # This approach only works for mutable dicts
            instance_type = type(obj)
            if instance_type == collections.defaultdict:
                d = collections.defaultdict(obj.default_factory)
            else:
                d = instance_type()
            seen_class_type = False
            for k, v in obj.items():
                stub_key, ret_seen_cls_type_key = self._replace_tensors_with_stubs(
                    k, memo, tx_list, c2c_possible, peers, for_offload
                )
                memo[id(k)] = stub_key
                stub_value, ret_seen_cls_type_value = self._replace_tensors_with_stubs(
                    v, memo, tx_list, c2c_possible, peers, for_offload
                )
                seen_class_type = (
                    seen_class_type or ret_seen_cls_type_key or ret_seen_cls_type_value
                )
                memo[id(v)] = stub_value
                d[stub_key] = stub_value
            memo[id(obj)] = d
            return d, seen_class_type

        if isinstance(obj, torch.Tensor):
            stub, transmission = self._stubify_tensor(
                obj, len(tx_list), c2c_possible, peers, for_offload
            )
            tx_list.append(transmission)
            memo[id(obj)] = stub
            return stub, False

        # class types
        for attr in dir(obj):
            obj_attr = getattr(obj, attr)
            if not callable(obj_attr) and not attr.startswith("__"):
                value, _ = self._replace_tensors_with_stubs(
                    obj_attr, memo, tx_list, c2c_possible, peers, for_offload
                )
                setattr(obj, attr, value)
                memo[id(obj_attr)] = value
        memo[id(obj)] = obj
        return obj, True

    def _stubify_tensor(self, tensor, index, c2c_possible, peers, for_offload=False):
        """
        Algorithm:
            - If the input tensor is dummy:
                  In this case the original tensor has already been sent to the destination.
                  Here we need to fetch the necessary metadata from DummyTensorRegistry
                  and re-create the TensorStub so that the destination can pick up the right tensor.
            - Else if this tensor has direct consumer(s), neither of which is the original destination (parent):
                  Create a TensorStub with is_dummy=True, which indicates to parent to not expect a tensor.
                  Create a new TensorTransmission with the destinations set to the direct consumers so
                  that ServerCommunicator directly sends them to their eventual destination.
            - Else (no direct consumers, or at least one direct consumer is in the parent):
                  Create a TensorStub with is_dummy=False, and create a TensorTransmission to parent (peers)
                  so that the parent can reconstruct the real tensor.
        """
        from smdistributed.modelparallel.torch.state_mod import state

        if hasattr(tensor, "_smp_is_dummy") and tensor._smp_is_dummy:
            # dests is empty because we will not send this
            dummy_transmission = TensorTransmission(tensor, [], -1)

            # the recorded stub will contain the original src and link_id of the tensor
            stub = self.dummy_registry.get(tensor)

            # setting dummy to false because we want the destination to reconstruct the actual tensor
            stub = TensorStub.create_non_dummy_version(stub, index)

            return stub, dummy_transmission

        # the default values. these might be overriden in case of child-to-child transmissions
        dests = peers
        is_dummy = False

        # initiate direct child-to-child transmission
        if (
            c2c_possible
            and state.cfg.fast_mode
            and state.current_minibatch() > 0
            and hasattr(tensor, "_smp_module_info")
            and (tensor._smp_module_info.is_forward == False or tensor.requires_grad)
        ):
            # the order of conditions above matter because there will be cases where state.current_microbatch()
            # is None (we are outside a StepFunction) when c2c_possible == False

            # tensor.requires_grad is not a hard requirement, but if requires_grad is False
            # we cannot detect that a dummy tensor is being used in parent and raise error

            assert len(peers) == 1, "Should not come here for broadcast communication"

            module_info = tensor._smp_module_info
            if state.current_step_func().has_direct_consumers(module_info):
                consumer_partitions = state.current_step_func().get_direct_consumer_partitions(
                    module_info
                )
                consumer_ranks = [core.pp_rank_to_rank(r) for r in consumer_partitions]
                # send the dummy version only if the intended destination is not a consumer
                if peers[0] not in consumer_ranks:
                    is_dummy = True
                    dests = list(set(consumer_ranks))

        module_info = tensor._smp_module_info if hasattr(tensor, "_smp_module_info") else None
        src = rank()

        if for_offload:
            link_id = None
        else:
            # (Warning!) The self.tensor_index_to_link_id is controlled by the ServerCommunicator
            # ServerCommunicator._maybe_record_and_update_link_ids will change this for each message
            if self.tensor_index_to_link_id != None:
                # This only happends for static mode and after record step
                link_id = self.tensor_index_to_link_id[index]
                get_logger().debug(
                    rmsg(
                        f"Static mode: getting cached link_id {link_id} for tensor index {index}, shape {tensor.shape} dtype {tensor.dtype}, src {src}, dests {dests}, is_dummy {is_dummy}"
                    )
                )
            else:
                assert (
                    not state.skip_metadata_transmission()
                ), "Skip_metadata_transmission is True but tensor_index_to_link_id is None!"
                link_id = state.link_manager.get_link(get_tensor_size(tensor))
                get_logger().debug(
                    rmsg(
                        f"Generated_link: {link_id}, src {src}, dests {dests}, is_dummy {is_dummy}"
                    )
                )

            # During the record step for static mode
            if state.should_record_metadata():
                state.exec_server.comm.tensor_index_to_link_id[index] = link_id

        stub = TensorStub(
            index,
            tensor.dtype,
            tensor.shape,
            tensor.requires_grad,
            tensor.device.type,
            is_dummy,
            src,
            link_id,
            module_info,
        )
        transmission = TensorTransmission(tensor, dests, link_id)

        return stub, transmission
