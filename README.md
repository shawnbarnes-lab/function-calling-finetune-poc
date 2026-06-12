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
| Shared storage | 1 TB filesystem |
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
