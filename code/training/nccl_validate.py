"""All-reduce sanity + bandwidth check via torch.distributed over NCCL --
exercises the same stack the training job uses (PyTorch -> NCCL -> IB).

Submit via test_nccl.slurm, or by hand (one srun task per node -- torchrun
spawns the 8 workers itself):

    srun -N2 --ntasks-per-node=1 --gpus-per-node=8 \\
        torchrun --nproc-per-node=8 --nnodes=2 \\
            --rdzv-backend=c10d --rdzv-endpoint=<head-ip>:29500 \\
            nccl_validate.py
"""

import os
import time
import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"world_size={world_size} backend=nccl "
              f"gpu={torch.cuda.get_device_name(local_rank)}")

    dist.barrier()
    if rank == 0:
        print("all ranks synchronized")

    # correctness: sum of rank ids across the world
    small_tensor = torch.ones(1024, device=device) * rank
    dist.all_reduce(small_tensor, op=dist.ReduceOp.SUM)
    expected = sum(range(world_size))
    if rank == 0:
        if torch.allclose(small_tensor, torch.tensor(float(expected), device=device)):
            print(f"small all-reduce: PASS (sum of ranks = {expected})")
        else:
            print(f"small all-reduce: FAIL (got {small_tensor[0].item()}, expected {expected})")

    # bandwidth: 1 GB fp32 tensor
    tensor_size_bytes = 1024 * 1024 * 1024
    num_elements = tensor_size_bytes // 4
    big_tensor = torch.ones(num_elements, dtype=torch.float32, device=device)

    for _ in range(3):  # warmup
        dist.all_reduce(big_tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    dist.barrier()

    num_iters = 10
    start = time.time()
    for _ in range(num_iters):
        dist.all_reduce(big_tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    # ring all-reduce moves 2*(N-1)/N * size bytes per rank
    bytes_per_iter = 2 * (world_size - 1) / world_size * tensor_size_bytes
    bandwidth_gbs = bytes_per_iter * num_iters / elapsed / (1024 ** 3)

    if rank == 0:
        print(f"1 GB all-reduce x {num_iters}: {elapsed:.2f}s total, "
              f"{elapsed/num_iters*1000:.1f}ms/iter, ~{bandwidth_gbs:.1f} GB/s")

        if bandwidth_gbs < 10:
            print("WARNING: bandwidth well below expectation for H200 over IB (50+ GB/s).")
            print("Check: NCCL_IB_DISABLE, NCCL_SOCKET_IFNAME, NCCL_NET_GDR_LEVEL, fabric topology")
        elif bandwidth_gbs > 40:
            print("bandwidth in expected range for H200 over IB")
        else:
            print("bandwidth lower than peak but functional")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
