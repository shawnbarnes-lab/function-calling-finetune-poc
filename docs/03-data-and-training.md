# 03. Data and training

Download the model and dataset, prepare the data, and run the fine-tune. All
paths are under `/opt/finetune`, which is the shared filesystem visible to every
node.

## Stage the code

From your workstation, copy the training scripts to the shared filesystem:

```
scp -i <SSH_KEY> code/training/prepare_data.py code/training/train.py \
    code/training/train.slurm root@<LOGIN_NODE_IP>:/opt/finetune/
```

## Create the training environment

On the login node. The environment lives on the shared filesystem, so every
worker uses the same one. No per-node setup.

```
cd /opt/finetune
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install torch==2.7.1 transformers==4.55.0 trl==0.19.1 \
    accelerate==1.9.0 peft==0.16.0 datasets==4.0.0 sentencepiece protobuf \
    huggingface_hub hf_transfer
```

`transformers` must be 4.51 or newer for the Qwen3 architecture. A large pip
install can be killed for memory on a small login node; if it finishes without a
`Successfully installed` line, it failed. Re-run; the wheel cache makes the
retry fast.

## Authenticate and prepare data

Accept the xLAM dataset terms in a browser first (logged in), then:

```
./venv/bin/huggingface-cli login        # paste a read token
nohup ./venv/bin/python prepare_data.py --dataset xlam > prepare_data.log 2>&1 &
tail -f prepare_data.log
```

This downloads Qwen3-32B (about 62 GB) to `/opt/finetune/models/qwen3-32b`,
downloads the dataset, formats every example through the Qwen3 chat template
with thinking mode disabled, and writes the splits to
`/opt/finetune/data/xlam-function-calling`.

Expected: 59,500 training examples and 500 held-out evaluation examples.

The chat template is applied with `enable_thinking=False` on purpose. Qwen3 can
emit a reasoning trace before its answer; for structured function calls that is
the wrong output, so it is turned off during data preparation.

## Run the fine-tune

```
cd /opt/finetune
sbatch --export=ALL,MODEL_SIZE=32b train.slurm
squeue
```

What `train.slurm` does: requests 2 nodes with 8 GPUs each, picks a head node
for rendezvous, then `srun` launches one `torchrun` per node, which spawns 8
worker processes per node. The 16 processes form one group over InfiniBand and
train with TRL's SFT trainer.

### Why LoRA with DDP, not full fine-tuning with FSDP

The base model is frozen. Only a LoRA adapter trains: 134M parameters, 0.41% of
the 32.9B total. Because the base model does not change, there is no value in
sharding it across GPUs (FSDP) and gathering it every step. Each GPU holds a
full copy of the model (about 64 GB, well under the 141 GB H200 memory), and the
only data crossing the network between steps is the adapter gradients, a few
megabytes. This keeps the GPUs busy rather than waiting on the network, which is
why utilization stayed near 96%.

The H200's memory is what makes the full-copy approach affordable. On a smaller
GPU the full model would not fit and sharding would be forced.

## Watch it

```
tail -f /opt/finetune/slurm-<jobid>-1.out                       # loss curve
srun --jobid=<jobid> --overlap -w worker-0 nvidia-smi           # live GPU table
```

Expected: about 12.5 seconds per step, GPU utilization 94-100%, around 670 watts
per GPU against a 700 watt cap. Three epochs in about 78 minutes. The loss falls
from about 0.47 to 0.085.

## Confirm completion

```
sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode
```

Expected: `COMPLETED`, elapsed about 1:18, exit `0:0`. The adapter is saved to
`/opt/finetune/checkpoints/qwen3-32b-xlam-lora/final/` and is about 537 MB.

## Publish the adapter (optional)

```
./venv/bin/hf upload <HF_USER>/qwen3-32b-xlam-function-calling-lora \
    checkpoints/qwen3-32b-xlam-lora/final . --repo-type model --private
```

This needs a write token. The adapter then survives independently of the
cluster.

## Next

Continue to [04-inference-and-comparison.md](04-inference-and-comparison.md).
