#!/bin/bash
#SBATCH --job-name=lora-finetune
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Multi-node LoRA fine-tune launcher.
#   sbatch --export=MODEL_SIZE=32b train.slurm
# torchrun across 2 nodes, DDP. The base model is frozen under LoRA so only
# the adapter gradients cross the wire between nodes.

set -euo pipefail

MODEL_SIZE="${MODEL_SIZE:-32b}"
echo "job=${SLURM_JOB_ID} model=${MODEL_SIZE} nodes=${SLURM_JOB_NODELIST} start=$(date)"

source /opt/finetune/venv/bin/activate

HEAD_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname --ip-address)
HEAD_NODE_PORT=29500

echo "head node: ${HEAD_NODE} (${HEAD_NODE_IP}:${HEAD_NODE_PORT})"

# NCCL over InfiniBand
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_ASYNC_ERROR_HANDLING=1

# caches on the shared FS so workers don't re-download
export HF_HOME=/opt/finetune/hf-cache
export TRANSFORMERS_CACHE=/opt/finetune/hf-cache
export TORCH_HOME=/opt/finetune/torch-cache

export WANDB_PROJECT="nebius-csa-demo"
export WANDB_NAME="lora-${MODEL_SIZE}-${SLURM_JOB_ID}"

# srun forks one task per node, torchrun spawns 8 workers per node
srun --output=slurm-%j-%n.out torchrun \
    --nproc-per-node=${SLURM_GPUS_PER_NODE} \
    --nnodes=${SLURM_NNODES} \
    --rdzv-id=${SLURM_JOB_ID} \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${HEAD_NODE_IP}:${HEAD_NODE_PORT}" \
    --rdzv-conf="timeout=600" \
    /opt/finetune/train.py \
    --model-size "${MODEL_SIZE}"

echo "end: $(date)"
