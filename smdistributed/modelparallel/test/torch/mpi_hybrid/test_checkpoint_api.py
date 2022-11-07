"""
Test all checkpoint API function correctly.
Checkpoint correctness test is defined in test_gpt_checkpoint.py
If you run this test manually, make sure to clear checkpoint_folder between each run
"""
# Standard Library
import copy
import logging
import os
import shutil

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.test.torch.model_zoo import gpt2_base
from smdistributed.modelparallel.torch.exceptions import SMPValidationError

logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.ERROR)


def reset():
    model.reset()
    smp.reset()
    smp.barrier()


def create_new_run(smp_config, model):
    smp.init(smp_config)
    model.create_smp_model()
    model.smp_model = smp.DistributedModel(model.smp_model, **model.smp_dist_model_kwargs)
    model.create_optimizer(model.smp_model, run_smp=True)
    model.smp_step_func(model.smp_model, target, *inputs)


model = copy.deepcopy(gpt2_base)
smp_config = model.smp_config
smp_config["tensor_parallel_degree"] = 2
smp_config["pipeline_parallel_degree"] = 2
smp_config["shard_optimizer_state"] = True
num_steps = 10
num_kept_partial_checkpoints = 5
checkpoint_folder = "/opt/ml/model/checkpoint_api"
model.optimizer = "adam"
user_content = {"test": "smp_checkpoint_api"}

smp.init(smp_config)
model.create_smp_model()
model.smp_model = smp.DistributedModel(model.smp_model, **model.smp_dist_model_kwargs)
model.create_optimizer(model.smp_model, run_smp=True)
model.create_inputs()

inputs, target = model.inputs, model.target

for i in range(10):
    model.smp_step_func(model.smp_model, target, *inputs)
    model.smp_optimizer.step()
    # Save partial
    smp.save_checkpoint(
        checkpoint_folder,
        f"test{i}",
        model=model.smp_model,
        optimizer=model.smp_optimizer,
        num_kept_partial_checkpoints=num_kept_partial_checkpoints,
        user_content=user_content,
    )

smp.barrier()
# Check if num_kept_partial_checkpoints works correctly
remaining_checkpoints = []
newest_file = None
for item in os.listdir(checkpoint_folder):
    if os.path.isdir(os.path.join(checkpoint_folder, item)):
        remaining_checkpoints.append(item)
    else:
        newest_file = item
remaining_checkpoints = sorted(remaining_checkpoints)
assert remaining_checkpoints == [
    "test5_partial",
    "test6_partial",
    "test7_partial",
    "test8_partial",
    "test9_partial",
], remaining_checkpoints
with open(os.path.join(checkpoint_folder, newest_file), "r") as fp:
    remaining_checkpoints = fp.readlines()
    remaining_checkpoints = sorted(remaining_checkpoints)
    assert remaining_checkpoints == [
        "test5_partial\n",
        "test6_partial\n",
        "test7_partial\n",
        "test8_partial\n",
        "test9_partial\n",
    ], remaining_checkpoints

smp.barrier()
# Save full
smp.save_checkpoint(
    checkpoint_folder, "test.pt", partial=False, model=model.smp_model, user_content=user_content
)
smp.barrier()
assert os.path.exists(os.path.join(checkpoint_folder, "test.pt"))
assert os.path.exists(os.path.join(checkpoint_folder, "user_content_test.pt"))
# Verify the newest tag is unchanged
with open(os.path.join(checkpoint_folder, newest_file), "r") as fp:
    remaining_checkpoints = fp.readlines()
    remaining_checkpoints = sorted(remaining_checkpoints)
    assert remaining_checkpoints == [
        "test5_partial\n",
        "test6_partial\n",
        "test7_partial\n",
        "test8_partial\n",
        "test9_partial\n",
    ], remaining_checkpoints
smp.barrier()

reset()

# Test Loading full before model/opt creation
try:
    smp.resume_from_checkpoint(checkpoint_folder, partial=False)
except SMPValidationError as e:
    print(f"Can not load full without tag, error caught: {e}")
user_content_loaded = smp.resume_from_checkpoint(checkpoint_folder, tag="test.pt", partial=False)
assert user_content == user_content_loaded, user_content_loaded
create_new_run(smp_config, model)
reset()

# Test Loading newest partial before model/opt creation
user_content_loaded = smp.resume_from_checkpoint(checkpoint_folder)
assert user_content == user_content_loaded
create_new_run(smp_config, model)
reset()

# Test load partial with a tag
user_content_loaded = smp.resume_from_checkpoint(checkpoint_folder, tag="test5")
create_new_run(smp_config, model)
reset()

# Test loading partial with a wrong tag
try:
    smp.resume_from_checkpoint(checkpoint_folder, tag="test1")
except SMPValidationError as e:
    print(f"Wrong path error caught as {e}")

# Test loading full with a wrong tag
try:
    smp.resume_from_checkpoint(checkpoint_folder, tag="test_no_exist", partial=False)
except SMPValidationError as e:
    print(f"Wrong path error caught as {e}")

reset()

# Test Loading newest partial with different config
smp_config_fail = copy.deepcopy(smp_config)
smp_config_fail["shard_optimizer_state"] = False
smp_config_success = copy.deepcopy(smp_config)
smp_config_success["fast_mode"] = True
smp.init(smp_config_fail)
# Test loading partial with a uncompatiable config change
try:
    smp.resume_from_checkpoint(checkpoint_folder)
except SMPValidationError as e:
    print(f"Wrong config error caught as {e}")

# Test loading partial with a compatiable config change
smp.reset()
smp.init(smp_config_success)
smp.resume_from_checkpoint(checkpoint_folder)

smp.barrier()
if smp.rank() == 0:
    # Clean up
    shutil.rmtree(checkpoint_folder)
    print("Checkpoint API test finished succesfully!")
