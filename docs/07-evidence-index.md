# 07. Evidence index

Each claim and how to verify it. Where a file in this repo proves it, the file
is named. Where it is verified on the running cluster or in the Nebius console,
the command or location is given so you can confirm it yourself rather than
trust a screenshot.

## Cluster deployed as specified

| Claim | How to verify |
|---|---|
| 2 nodes, 8 H200 each, registered to Slurm | On the cluster: `srun -N2 --ntasks-per-node=1 --gpus-per-node=8 nvidia-smi -L` returns 16 H200 UUIDs |
| Cluster healthy, both workers idle | On the cluster: `sinfo` shows two workers idle; the login banner lists them |
| 1 TB shared filesystem | On the cluster: `df -h /` shows the 1 TB jail mount; Nebius console, Storage, Shared filesystems |
| 1 TB network-SSD disk | Nebius console, Storage, Disks |
| Per-node GPUs and Slurm version | On the cluster: `sinfo -N -o "%N %G %c %m %T"` and `sinfo --version` |

## Training and utilization

| Claim | How to verify |
|---|---|
| 16 GPUs at 94-100% during training | During a run: `srun --jobid=<id> --overlap -w worker-0 nvidia-smi` |
| ~96% sustained for the full run | Cluster Grafana, GPU/DCGM dashboard; or Nebius console, Monitoring tab, over the run window |
| ~670 W per GPU sustained | Nebius console, Monitoring tab, power graph |
| Job ran on both nodes | During a run: `squeue` shows the job on `worker-[0-1]` |

## Training completed

| Claim | How to verify |
|---|---|
| 3 epochs, ~78 minutes, exit 0 | On the cluster: `sacct -j <id> --format=JobID,State,Elapsed,ExitCode` |
| Eval token accuracy ~98% (training health) | Final eval line in `/opt/finetune/slurm-<id>-1.out` |
| 134M trainable params (0.41%) | The "trainable params" line in the training log |
| Adapter saved, ~537 MB | On the cluster: `ls -la /opt/finetune/checkpoints/qwen3-32b-xlam-lora/final/` |
| Adapter published | Hugging Face repo `<HF_USER>/qwen3-32b-xlam-function-calling-lora` |

## Inference comparison

| Claim | How to verify |
|---|---|
| Both endpoints served via Slurm | On the cluster: `squeue` shows two `vllm-serve` jobs |
| Both endpoints answering | On the cluster: `curl -s http://<node>:8000/v1/models` and `:8001` |
| base 3/20 vs fine-tuned 16/20 valid calls | `code/inference/results/comparison_output.md` (committed in this repo) |
| Schema difference on a live prompt | Run the single-request curl in [04-inference-and-comparison.md](04-inference-and-comparison.md) against each endpoint |

## Code

| Item | File |
|---|---|
| Cluster definition | `code/terraform/terraform.tfvars` |
| Data preparation | `code/training/prepare_data.py` |
| Training | `code/training/train.py`, `code/training/train.slurm` |
| Serving | `code/inference/vllm_serve.slurm` |
| Comparison harness | `code/inference/compare.py`, `code/inference/test_prompts.json` |
| Comparison result | `code/inference/results/comparison_output.md` |

The committed `comparison_output.md` holds the full per-prompt output, not just
the aggregate, so the inference result can be read directly from this repo
without rerunning anything.
