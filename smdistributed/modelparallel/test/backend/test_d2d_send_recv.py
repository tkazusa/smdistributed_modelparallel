# Third Party
import torch

# First Party
import smdistributed.modelparallel.torch as smp

smp.init(
    {
        "pipeline_parallel_degree": 2,
        "tensor_parallel_degree": 1,
        "ddp": True,
        "placement_strategy": "spread",
    }
)
torch.cuda.set_device(smp.local_rank())

# tensors are saved to S3 and downloaded during container building
input_tensors = torch.load("d2d_small.pt")
input_tensors = [t.cuda() for t in input_tensors]
num_tensors = len(input_tensors)
recvd_tensors = []
recvd_large_tensor = None
wait_recv_artifacts = []

num_transactions = 100

# run num_transactions send/recvs from pp rank 0 to pp rank 1
# send is sycn and recv is async
for i in range(num_transactions):
    current_idx = i % num_tensors
    tensor = input_tensors[current_idx]
    if smp.pp_rank() == 0:
        smp.state.comm.send_large((current_idx, tensor), 1, smp.RankType.PP_RANK)
    else:
        wait_recv_artifacts.append(smp.state.comm.async_recv_from_large(0, smp.RankType.PP_RANK))

# wait the recv to finish and collect the tensors
for handles, stubbed_obj in wait_recv_artifacts:
    recvd_tensors.append(smp.state.comm.wait_recv_large(handles, stubbed_obj))

# verify tensor match
for index, tensor in recvd_tensors:
    if not torch.allclose(tensor, input_tensors[index]):
        raise ValueError(
            f"Tensor mismatch after D2D communication, origin {input_tensors[index]}, received {tensor}"
        )

# Release the tensors after receive
recvd_tensors = None
input_tensors = None
wait_recv_artifacts = None

smp.barrier()
alloc_stats = smp.state.core.get_and_reset_alloc_stats()
success_alloc = alloc_stats.backend_d2d_success_allocated
failure_alloc = alloc_stats.backend_d2d_failure_allocated
total_alloc = failure_alloc + success_alloc
if smp.pp_rank() == 1:
    assert success_alloc > 0, "No successful allocation for small tensors"
    assert (
        total_alloc == num_transactions
    ), f"total_alloc {total_alloc} does not equal to num_transactions {num_transactions}"
if smp.rank() == 0:
    print("Test fallback to mpi with large tensor")

# large tensort is about 0.8G while the d2d buffer for this test is set to 0.6G, allocation should fail
# use sync send/recv
large_tensor = torch.load("d2d_large.pt").cuda()
if smp.pp_rank() == 0:
    smp.state.comm.send_large(large_tensor, 1, smp.RankType.PP_RANK)
else:
    recvd_large_tensor = smp.state.comm.recv_from_large(0, smp.RankType.PP_RANK)

if smp.pp_rank() == 1 and not torch.allclose(recvd_large_tensor, large_tensor):
    raise ValueError(
        f"Tensor mismatch after D2D communication, origin {large_tensor}, received {recvd_large_tensor}"
    )
smp.barrier()

alloc_stats = smp.state.core.get_and_reset_alloc_stats()
success_alloc = alloc_stats.backend_d2d_success_allocated
failure_alloc = alloc_stats.backend_d2d_failure_allocated
if smp.pp_rank() == 1:
    assert (
        success_alloc == 0 and failure_alloc == 1
    ), f"Sending large tensor should fail to allocate, but getting success_alloc {success_alloc} and failure_alloc {failure_alloc}"

print("D2D test finished successfully")
