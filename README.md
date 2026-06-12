# Nebius function-calling fine-tune PoC

This repository documents a Nebius-based fine-tuning workflow for function
calling, including infrastructure deployment, distributed training, monitoring,
and inference validation.

The PoC used a Slurm cluster provisioned with Soperator to fine-tune Qwen3-32B
on the Salesforce xLAM 60k dataset using LoRA, then compared base and adapted
model behavior under identical inference conditions.

## Environment

| Component | Configuration |
|---|---|
| Scheduler | Slurm via Soperator |
| Compute | 2 GPU worker nodes |
| GPUs | 16x NVIDIA H200 (8 per node) |
| Interconnect | InfiniBand |
| Shared filesystem | 1 TB cluster jail (`filestore_jail`), mounted on every node |
| SSD network disk | 1 TB on `NETWORK_SSD_IO_M3` (`nfs_in_k8s`), mounted at `/home` |
| Monitoring | Nebius console, Grafana, DCGM |
| Inference | vLLM endpoints for base and fine-tuned models |

## Results

| Area | Result |
|---|---|
| Model | Qwen3-32B |
| Dataset | Salesforce xLAM 60k |
| Fine-tuning method | LoRA |
| Distributed training | DDP across 16 H200 GPUs |
| Training time | 78 minutes for 3 epochs |
| GPU utilization | 94-100% during training |
| Trainable parameters | 134M of 32.9B |
| Eval signal | 98.2% token accuracy |
| Inference comparison | Base: 3/20 valid calls, fine-tuned: 16/20 valid calls |
| Adapter size | 537 MB |

The primary outcome was improved function-call validity under inference. Token
accuracy is included as a training signal, but endpoint behavior is the more
relevant measure for agent-oriented workloads.

## Storage layout

The assignment calls for both a 1 TB SSD network disk and a 1 TB shared
filesystem. Both are provisioned in
[`code/terraform/terraform.tfvars`](code/terraform/terraform.tfvars):

**1 TB shared filesystem** — the cluster jail (`filestore_jail`,
`size_gibibytes = 1024`), mounted on the controller, worker, and login nodes.
Holds the dataset, training checkpoints, and the trained LoRA adapter.

**1 TB SSD network disk** — an NFS share backed by a `NETWORK_SSD_IO_M3` disk
(`nfs_in_k8s`, `size_gibibytes = 1023`, ext4), mounted at `/home` across the
cluster. It is sized at 1023 GiB rather than 1024 because `NETWORK_SSD_IO_M3`
capacity must be a multiple of 93 GiB, so 1023 (= 11 x 93) is the nearest
1 TB-class value.

## Repo layout

```
code/
  terraform/     Terraform configuration and tfvars used for the PoC
  training/      Data preparation, training, and validation scripts
  inference/     vLLM serving and comparison scripts
docs/
  01-terraform-and-deploy.md
  02-validate-cluster.md
  03-data-and-training.md
  04-inference-and-comparison.md
  05-monitoring.md
  06-troubleshooting.md
  07-evidence-index.md
```

## Reading order

1. [docs/01-terraform-and-deploy.md](docs/01-terraform-and-deploy.md)
   Infrastructure deployment and Terraform structure.
2. [docs/02-validate-cluster.md](docs/02-validate-cluster.md)
   Cluster validation and GPU visibility checks.
3. [docs/03-data-and-training.md](docs/03-data-and-training.md)
   Dataset preparation, model staging, and training execution.
4. [docs/05-monitoring.md](docs/05-monitoring.md)
   Runtime telemetry, utilization, and observability.
5. [docs/04-inference-and-comparison.md](docs/04-inference-and-comparison.md)
   Endpoint deployment and behavioral comparison.
6. [docs/06-troubleshooting.md](docs/06-troubleshooting.md)
   Build issues encountered and remediation steps.
7. [docs/07-evidence-index.md](docs/07-evidence-index.md)
   Source mapping for reported outcomes.

## Using this repository

The repository supports three workflows:

- Recreate the infrastructure and training setup.
- Validate cluster readiness and runtime behavior.
- Verify reported outcomes against the committed results and live checks.

The accompanying slide deck summarizes outcomes; this repository contains the
implementation details.

## Reproduction notes

Environment-specific values are represented as placeholders and should be
replaced before execution:

```
<LOGIN_NODE_IP>
<SSH_KEY>
<HF_USER>
<PROJECT_ID>
<TENANT_ID>
```

Secrets, credentials, kubeconfigs, and private keys are intentionally excluded.

## Monitoring notes

Training telemetry was collected through:

- Nebius GPU utilization metrics
- Grafana dashboards
- DCGM telemetry
- Slurm job logs
- nvidia-smi sampling during execution

Observed utilization remained between 94-100% across the 78-minute training run.
See [docs/07-evidence-index.md](docs/07-evidence-index.md) for how to verify each
reported figure.
