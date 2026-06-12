#!/bin/bash
#SBATCH --job-name=vllm-serve
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=vllm-%j.out

# vLLM serving as a Slurm job. On a Soperator cluster Slurm owns the GPUs,
# so inference runs through the same scheduler as training.
#
# Usage:
#   sbatch --export=ALL,ROLE=base,PORT=8000 vllm_serve.slurm
#   sbatch --export=ALL,ROLE=finetuned,PORT=8001 vllm_serve.slurm

set -euo pipefail

ROLE="${ROLE:-base}"
PORT="${PORT:-8000}"
MODEL_DIR=/opt/finetune/models/qwen3-32b
ADAPTER_DIR=/opt/finetune/checkpoints/qwen3-32b-xlam-lora/final

echo "ROLE=${ROLE} PORT=${PORT} NODE=$(hostname)"
echo "SERVING_ON $(hostname):${PORT}"

source /opt/finetune/venv-vllm/bin/activate

# /root is on the shared jail FS (virtiofs). Triton's JIT cache must live on
# node-local disk: shared-FS caches both race across jobs and break the
# write-then-dlopen sequence Triton relies on. /tmp inside the jail is
# node-local ext4. Same for vLLM's compile cache.
export TRITON_CACHE_DIR=/tmp/triton-cache-${ROLE}
export VLLM_CACHE_ROOT=/tmp/vllm-cache-${ROLE}
rm -rf "${TRITON_CACHE_DIR}" "${VLLM_CACHE_ROOT}"
mkdir -p "${TRITON_CACHE_DIR}" "${VLLM_CACHE_ROOT}"

EXTRA_ARGS=()
if [ "${ROLE}" = "finetuned" ]; then
    EXTRA_ARGS+=(
        --enable-lora
        --max-lora-rank 16
        --lora-modules "qwen3-32b-xlam-lora=${ADAPTER_DIR}"
    )
fi

exec vllm serve "${MODEL_DIR}" \
    --served-model-name "qwen3-32b-${ROLE}" \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --host 0.0.0.0 \
    --port "${PORT}" \
    "${EXTRA_ARGS[@]}"
