#!/bin/bash
#SBATCH --job-name=nccl-validate
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=00:10:00
#SBATCH --output=nccl-validate-%j.out
#SBATCH --error=nccl-validate-%j.err

# Multi-node NCCL validation via PyTorch torch.distributed
# Submit with: sbatch test_nccl.slurm

set -euo pipefail

echo "==========================================="
echo "NCCL validation job: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "Start: $(date)"
echo "==========================================="

source /opt/finetune/venv/bin/activate

HEAD_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname --ip-address)

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_SOCKET_IFNAME=^lo,docker

srun torchrun \
    --nproc-per-node=${SLURM_GPUS_PER_NODE} \
    --nnodes=${SLURM_NNODES} \
    --rdzv-id=${SLURM_JOB_ID} \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${HEAD_NODE_IP}:29500" \
    --rdzv-conf="timeout=300" \
    /opt/finetune/nccl_validate.py

echo "End: $(date)"
