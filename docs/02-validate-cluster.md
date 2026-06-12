# 02. Validate the cluster

Confirm the cluster is healthy and all 16 GPUs are visible to the scheduler
before spending time on data or training.

## Connect to the login node

```
ssh -i <SSH_KEY> root@<LOGIN_NODE_IP>
```

You should see the Soperator banner and land at `root@login-0`. The banner shows
the node list and the shared filesystem mounted at `/`.

## Check the scheduler

```
sinfo
```

Expected: two worker nodes, state `idle`. If a node shows `down` or `drain`, run
`scontrol show node <name>` and read the `Reason` field.

```
sinfo -N -o "%N %G %c %m %T"
```

Expected: each worker reports `gpu:nvidia_h200:8`, 128 CPUs, and its memory.

## Confirm all 16 GPUs through the scheduler

```
srun -N2 --ntasks-per-node=1 --gpus-per-node=8 nvidia-smi -L
```

Expected: 16 lines, each `GPU n: NVIDIA H200 (UUID: ...)`, eight per node. This
is the acceptance test for the infrastructure half of the project. The GPUs are
allocated through Slurm, the same way a training job will request them.

## Note on paths

Inside the Soperator jail, the 1 TB shared filesystem is the root filesystem. A
path like `/opt/finetune` is identical on the login node and on every worker.
All model weights, datasets, virtual environments, and checkpoints in the
following docs live under `/opt/finetune` so that every node can read them.

## Next

Continue to [03-data-and-training.md](03-data-and-training.md).
