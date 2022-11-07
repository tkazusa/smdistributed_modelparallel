# Future
from __future__ import print_function

# Standard Library
import argparse
import random

# Third Party
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import StepLR
from torchnet.dataset import SplitDataset
from torchvision import datasets, transforms

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.test.torch.utils import ATOL, FP16_Module

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        return x


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, 1)
        return output


class GroupedNet(nn.Module):
    def __init__(self):
        super(GroupedNet, self).__init__()
        self.net1 = Net1()
        self.net2 = Net2()

    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x)
        return x


@smp.step
def train_step(args, model, scaler, data, target):
    with autocast(args.amp > 0):
        output = model(data)
    loss = F.nll_loss(output, target, reduction="mean")
    scaled_loss = scaler.scale(loss) if args.amp else loss
    model.backward(scaled_loss)
    return output, loss


def train(args, model, scaler, device, train_loader, optimizer, epoch):
    model.train()
    if args.join:
        num_batches_by_rank = (
            args.num_batches
            if args.num_batches > 0
            else len(train_loader.dataset) / (2 * args.batch_size * smp.rdp_size())
        )
        if smp.rdp_rank() == 0:
            num_batches_by_rank = num_batches_by_rank / 2
        elif smp.rdp_rank() == 1:
            num_batches_by_rank = 3 * num_batches_by_rank / 4

    def inner_train():
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output_mb, loss_mb = train_step(args, model, scaler, data, target)
            loss = loss_mb.reduce_mean()
            output = output_mb.stack()
            if args.amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.fp32_grad_accumulation:
                    # Only cast, when params are in FP16 i.e. when FP32 grad
                    # accumulation is set, otherwise this will throw an error
                    # since grad should be of the same type as params
                    for param in model.local_parameters():
                        param.grad = param.grad.to(torch.float16)
                optimizer.step()
            if smp.rank() == 0 and batch_idx % args.log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

                if args.dry_run:
                    break
            if args.join:
                if batch_idx + 1 > num_batches_by_rank:
                    break
            if args.num_batches and batch_idx + 1 == args.num_batches:
                break

    if args.join:
        with model.join():
            inner_train()
    else:
        inner_train()


@smp.step
def test_step(model, data, target):
    output = model(data)
    loss = F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct = pred.eq(target.view_as(pred)).sum().item()
    return loss, correct


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            loss_batch, correct_batch = test_step(model, data, target)
            test_loss += sum(loss_batch)
            correct += sum(correct_batch)
            if args.num_batches and batch_idx + 1 == args.num_batches:
                break

    test_loss /= len(test_loader.dataset)
    if smp.pp_rank() == 0:
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )
    return test_loss


def test_loss_equal(args, device, test_loss):
    """
    Loss should be equal at the beginning and the end even though seeds for each process were set differently.
    At the beginning this is because of broadcast, at the end due to allreduce.
    """
    import math

    losses = smp.allgather(test_loss, group=smp.DP_GROUP)
    for l in losses:
        assert math.isclose(l, losses[0], abs_tol=ATOL)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, metavar="N", help="number of epochs to train (default: 14)"
    )
    parser.add_argument(
        "--lr", type=float, default=4.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False, help="quickly check a single pass"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--partial-checkpoint", type=str, default="", help="The checkpoint path to load"
    )
    parser.add_argument(
        "--full-checkpoint", type=str, default="", help="The checkpoint path to load"
    )
    parser.add_argument(
        "--save-full-model", action="store_true", default=False, help="For Saving the current Model"
    )
    parser.add_argument(
        "--save-partial-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--back_compat",
        action="store_true",
        default=False,
        help="Load the old checkpoint in back compatible mode",
    )
    parser.add_argument("--load-in-hook", type=int, default=0)
    parser.add_argument("--num-microbatches", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument("--num-partitions", type=int, default=2)
    parser.add_argument("--tensor-parallel-degree", type=int, default=1)
    parser.add_argument("--horovod", type=int, default=0)
    parser.add_argument("--overlapping_allreduce", type=int, default=1)
    parser.add_argument("--ddp", type=int, default=0)
    parser.add_argument("--amp", type=int, default=0)
    parser.add_argument("--join", type=int, default=0)
    parser.add_argument("--pipeline", type=str, default="interleaved")
    parser.add_argument("--fp32_grad_accumulation", type=int, default=0)
    args = parser.parse_args()
    use_ddp = args.ddp > 0
    use_horovod = args.horovod > 0

    cfg = {
        "microbatches": args.num_microbatches,
        "placement_strategy": "spread",
        "pipeline": args.pipeline,
        "optimize": "speed",
        "partitions": args.num_partitions,
        "tensor_parallel_degree": args.tensor_parallel_degree,
        "horovod": use_horovod,
        "ddp": use_ddp,
        "_fp32_grad_accumulation": args.fp32_grad_accumulation > 0,
        "fp16": args.fp32_grad_accumulation > 0,
    }

    smp.init(cfg)

    # different seeds for each rank so they initialize params differently
    random.seed(args.seed + smp.rank())
    np.random.seed(args.seed + smp.rank())
    torch.manual_seed(args.seed + smp.rank())
    torch.cuda.manual_seed(args.seed + smp.rank())

    torch.cuda.set_device(smp.local_rank())
    device = torch.device("cuda")
    kwargs = {"batch_size": args.batch_size, "num_workers": 1, "pin_memory": True, "shuffle": False}

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    if smp.local_rank() == 0:
        dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    smp.barrier()
    dataset1 = datasets.MNIST("../data", train=True, download=False, transform=transform)

    if (use_ddp or use_horovod) and smp.dp_size() > 1:
        partitions_dict = {f"{i}": 1 / smp.dp_size() for i in range(smp.dp_size())}
        dataset1 = SplitDataset(dataset1, partitions=partitions_dict)
        dataset1.select(f"{smp.dp_rank()}")

    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    model = GroupedNet()
    if args.tensor_parallel_degree > 1:
        smp.set_tensor_parallelism(model.net2.fc1, True)

    model.to(device)
    if args.fp32_grad_accumulation:
        torch.set_default_dtype(torch.float32)
        model = FP16_Module(model)
        torch.set_default_dtype(torch.float16)
    model = smp.DistributedModel(
        model,
        overlapping_allreduce=args.overlapping_allreduce > 0,
        gradient_as_bucket_view=args.fp32_grad_accumulation > 0,
    )
    if args.fp32_grad_accumulation:
        torch.set_default_dtype(torch.float32)
    scaler = smp.amp.GradScaler()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = smp.DistributedOptimizer(optimizer)

    if args.partial_checkpoint:
        checkpoint = smp.load(args.partial_checkpoint, partial=True, back_compat=args.back_compat)

        def model_optim_callable(model, optimizer):
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if args.load_in_hook:
            # delete partition_info as loading happened after partition
            del checkpoint["model_state_dict"]["_smp_load_info"]["partition_info"]
            model.register_post_partition_hook(model_optim_callable)
        else:
            model_optim_callable(model, optimizer)

    elif args.full_checkpoint:
        checkpoint = smp.load(args.full_checkpoint, partial=False)

        def model_optim_callable(model, optimizer):
            model.load_state_dict(checkpoint["model_state_dict"])
            # do not load optimizer states when loading full checkpoint
            # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if args.load_in_hook:
            model.register_post_partition_hook(model_optim_callable)
        else:
            model_optim_callable(model, optimizer)

    opt = optimizer.optimizer if args.fp32_grad_accumulation > 0 else optimizer

    scheduler = StepLR(opt, step_size=1, gamma=args.gamma)

    test_loss = test(args, model, device, test_loader)
    test_loss_equal(args, device, test_loss)

    if args.partial_checkpoint or args.full_checkpoint:
        assert test_loss < 0.08
        return

    for epoch in range(1, args.epochs + 1):
        train(args, model, scaler, device, train_loader, optimizer, epoch)
        test_loss = test(args, model, device, test_loader)
        scheduler.step()

    if args.save_partial_model:
        if smp.rdp_rank() == 0:
            model_dict = model.local_state_dict()
            opt_dict = optimizer.local_state_dict()
            smp.save(
                {"model_state_dict": model_dict, "optimizer_state_dict": opt_dict},
                f"./pt_mnist_checkpoint.pt",
                partial=True,
            )

    if args.save_full_model:
        if smp.rdp_rank() == 0:
            model_dict = model.state_dict()
            opt_dict = optimizer.state_dict()
            smp.save(
                {"model_state_dict": model_dict, "optimizer_state_dict": opt_dict},
                "./pt_mnist_checkpoint.pt",
                partial=False,
            )

    # Waiting the save checkpoint to be finished before run another allgather_object
    smp.barrier()

    test_loss_equal(args, device, test_loss)
    if not args.join:
        assert test_loss < 0.1, test_loss


if __name__ == "__main__":
    main()
