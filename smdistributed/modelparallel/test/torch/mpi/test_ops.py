# Standard Library
import unittest

# Third Party
import torch

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.test.torch.utils import (
    Net1,
    Net2,
    NetManyToOne,
    NetOneToMany,
    mock_get_module_fns,
    mock_is_executor,
    mock_output_stack_size,
    mock_push_output,
    mock_recv_forward,
    mock_send_backward,
)
from smdistributed.modelparallel.torch.ops import SMPInput, SMPParentRecv
from smdistributed.modelparallel.torch.state_mod import state
from smdistributed.modelparallel.torch.utils import check_requires_grad

cfg = {
    "microbatches": 2,
    "placement_strategy": "spread",
    "partitions": 2,
    "pipeline": "simple",
    "optimize": "speed",
}
smp.init(cfg)


class TestSMPInputOp(unittest.TestCase):
    """Tests SMPInput op which runs before running the forward of the module,
    when the current rank is the executor
    """

    '''
    def test_not_executor(self):
        """Test case: The execution server executor the current module, doesn't
        own the module. The backward request needs to be created and sent to a
        different rank.
        """
        # instantiate modules for subgraphs (sg#: subgraph#)
        sg1, sg2 = Net1(), Net2()

        # Forward pass, tests a common use case for SMPInput op
        # 1. the output of subgraph1 feeds into SMPInput op
        # 2. the output of SMPInput op feeds into another subgraph
        x = torch.ones(15, 1000, requires_grad=True)
        sg1_out = sg1(x)

        # Below two lines simulate removal of tensor from the computation graph,
        # followed by copy, when a tensor is sent to another rank.
        # NOTE: We need to set requires_grad=True to ensure SMPInput backward runs.
        sg1_out_copy = sg1_out.detach().clone()
        sg1_out_copy.requires_grad = True
        mock_fns = mock_get_module_fns(parent_module=sg1)
        mock_exec = mock_is_executor(is_executor=False, is_parent_executor=False)
        with mock_fns, mock_exec:
            smp_out = SMPInput.apply(sg1, sg1_out_copy)
        sg2_out = sg2(*smp_out)

        # With the mocks in place, the backward call will trigger backward
        # on sg1_out by calling sg1_out.backward(out_grad)
        out_grad = torch.ones(15, 10)
        mock_fns = mock_get_module_fns(parent_module=sg1)
        mock_send = mock_send_backward((sg1_out,), backward=True)
        mock_exec = mock_is_executor(is_executor=False)
        with mock_send, mock_fns:
            sg2_out.backward(out_grad)

        # Forward, backward with SMPInput op
        y = torch.ones(15, 1000)
        y.requires_grad = True
        sg2_out = sg2(sg1(y))
        sg2_out.backward(out_grad)
        # Check for correctness of resulting grads
        torch.allclose(x.grad, y.grad)
    '''

    def test_is_main(self):
        """Test case: The current module is main (root module)"""
        x = torch.ones(15, 1000, requires_grad=True)
        # Passing None to module for testing purpose
        # since current module is main, module shoudlnt be required
        mock_exec = mock_is_executor(is_executor=False, is_parent_executor=False)
        with mock_get_module_fns(parent_module="main"), mock_exec:
            smp_out, = SMPInput.apply(None, 0, x)
            smp_out.grad_fn.pending_smpinput_bwds = 1

        out_grads = torch.ones(15, 1000)
        # mocks get_parent_module call to return None.
        # This happens when the current module is main module.
        with mock_get_module_fns(parent_module=None):
            smp_out.backward(out_grads)
        torch.allclose(out_grads, x.grad)
        smp.barrier()

    def test_has_multiple_inputs(self):
        """Test case: The current module (executing the SMPInput op) takes
        multiple inputs"""
        x = torch.ones(10, 10, requires_grad=True)
        # Multiple subgraphs, first subgraph takes one input and returns many
        # outputs. Second subgraph takes many inputs and returns one input.
        # First subgraph feeds into SMPInput op and SMPInput's outputs
        # feed into the second subgraph.
        sg1, sg2 = NetOneToMany(), NetManyToOne()
        sg1_out1, sg1_out2 = sg1(x)
        sg1_out1_copy, sg1_out2_copy = sg1_out1.detach().clone(), sg1_out2.detach().clone()
        sg1_out1_copy.requires_grad = True
        sg1_out2_copy.requires_grad = True

        # mocks get_module_fn and friends and is_executor
        mock_fns = mock_get_module_fns(parent_module=sg1)
        mock_exec = mock_is_executor(is_executor=False, is_parent_executor=False)
        with mock_fns, mock_exec:
            smp_out1, smp_out2 = SMPInput.apply(sg1, 0, sg1_out1_copy, sg1_out2_copy)
            smp_out1.grad_fn.pending_smpinput_bwds = 1
        sg2_out = sg2(smp_out1, smp_out2)

        out_grads = torch.ones(10, 10)
        # mocks backward send to calculate backward on
        # sg1_out1, sg1_out2
        mock_send = mock_send_backward((sg1_out1, sg1_out2), backward=True)
        with mock_fns, mock_send, mock_exec:
            sg2_out.backward(out_grads)

        # Forward, backward without SMPInput op
        y = torch.ones(10, 10, requires_grad=True)
        sg1_out1, sg1_out2 = sg1(y)
        sg2_out = sg2(sg1_out1, sg1_out2)
        sg2_out.backward(out_grads)
        torch.allclose(x.grad, y.grad)
        smp.barrier()


@unittest.skip
class TestSMPParentSendOp(unittest.TestCase):
    """Tests SMPParentSendOp which runs if the current rank is
    the parent executor for the current module"""

    def test_standalone(self):
        """Tests the op standalone and runs backward on the saved
        outputs"""
        # using a dummy subgraph for mocking parent_module fns
        sg1 = Net1()
        x = torch.ones(10, 10, requires_grad=True)
        # creates a clone which preserves requires_grad flag and returns that as output in mock
        outputs = (x.clone(),)
        # assign dummy partition to module
        state.module_manager.assign_partition(sg1, 0)
        state.module_manager.assign_partition(None, 0)
        with mock_recv_forward(outputs), mock_get_module_fns(parent_module=sg1):
            # call the forward and pop output and store it in y1
            SMPParentSend.apply(None, x)
            y1 = state.module_manager.pop_output(state.microbatch, sg1)

        # y1 should trigger the backward correctly and since it is an identity
        # x should store the same grads
        out_grad = torch.zeros(10, 10)
        y1[0].backward(out_grad)
        # correctness check
        torch.allclose(x.grad, out_grad)
        smp.barrier()


class TestSMPParentRecvOp(unittest.TestCase):
    """Tests SMPParentRecvOp which runs if the current is
    the parent executor for the current module"""

    def test_standalone(self):
        """Tests the op standalone and runs backward on the saved
        outputs"""
        # using a dummy subgraph for mocking parent_module fns
        sg1 = Net1()
        x = torch.ones(10, 10, requires_grad=True)
        state.module_manager.assign_partition(sg1, 0)

        # mock get_parent_module and friends and run SMPParentRecv
        with mock_get_module_fns(parent_module=sg1), mock_output_stack_size(stack_size=1):
            outputs_op = SMPParentRecv.apply(sg1, sg1, 0, {}, False, x)
            outputs_op[0].grad_fn.next_parent_recvs_bwd_pending += 1

        out_grads = torch.ones(10, 10)
        with mock_send_backward(outputs=None), mock_get_module_fns(parent_module=sg1):
            outputs_op[0].backward(out_grads)

        # check grads are same
        torch.allclose(x.grad, out_grads)
        smp.barrier()


class TestSMPParentSendRecv(unittest.TestCase):
    """Tests SMPParentSendRecv which tests the SMPParenSend
    and SMPParentRecv together"""

    def test_combined(self):
        x = torch.ones(40, 1000, requires_grad=True)
        # First subgraph, outputs feeds into SMPParentSend
        # dummpy subgraph to use when packing get_parent_module calls
        sg1 = Net1()
        sg1_out = sg1(x)
        outputs = (torch.rand(40, 20, requires_grad=True),)

        # SMPParentSend, forward
        with mock_recv_forward(outputs), mock_get_module_fns(parent_module=sg1):
            # outputs_op = SMPParentSend.apply(sg1, sg1_out)
            request = smp.messages.ModuleExecutionRequest.create_forward_req(
                sg1, sg1_out, {}, enable_grads=True, enable_autocast=False
            )
            parent_module = state.module_manager.get_parent_module("dummy")
            outputs = state.current_worker.thread_get_forward_result(request)
            outputs_op = outputs if isinstance(outputs, tuple) else (outputs,)

            if check_requires_grad(outputs_op):
                state.module_manager.push_output(
                    state.microbatch, "dummy", parent_module, (sg1_out,)
                )
                saved_module = "dummy"
                saved_parent_module = str(parent_module)
        # need to detach from graph and make outputs_op_copy leaves
        # This ensures that backward stops here and the rank can handle other
        # requests
        outputs_op_copy = (output.detach().clone() for output in outputs_op)
        for output in outputs_op_copy:
            output.requires_grad = True

        # SMPParentRecv, forward
        with mock_get_module_fns(parent_module=sg1), mock_output_stack_size(stack_size=1):
            outputs_op = SMPParentRecv.apply(sg1, sg1, 0, {}, False, x)
            outputs_op[0].grad_fn.next_parent_recvs_bwd_pending += 1

        # SMPParentRecv, backward
        with mock_send_backward(outputs=None), mock_get_module_fns(sg1):
            outputs_op[0].backward(torch.ones(40, 1000))

        # After the above backward, only the sg1 part of the
        # backward remains. The stack state currently contains
        # the args which were inputs to SMPParentSend which is
        # sg1_out. Thus popping the stack and running the backward
        # will complete our backward pass.
        with mock_get_module_fns(parent_module=sg1):
            outputs_stack = state.module_manager.get_output(
                state.microbatch, saved_module, saved_parent_module, 0
            )
            outputs_stack[0].backward(torch.ones(40, 20))

        # Compute grad values, without smp ops
        y = torch.ones(40, 1000, requires_grad=True)
        sg1 = Net1()
        sg1_out = sg1(y)
        sg1_out.backward(torch.ones(40, 20))

        # Check for correctness
        torch.allclose(y.grad, x.grad)
        smp.barrier()


if __name__ == "__main__":
    unittest.main()
    smp.barrier()
