# 01. Terraform and deploy

Deploy the Slurm cluster from one Terraform configuration. A single
`terraform apply` provisions the infrastructure (GPU nodes, network, storage,
managed Kubernetes) and then installs Soperator and the Slurm stack on top of
that Kubernetes cluster. The apply does both layers and waits until Slurm
reports ready.

## Prerequisites

- Terraform 1.5 or newer
- The Nebius CLI, authenticated (`nebius init`)
- `kubectl`
- A sandbox or project with quota for: 1 GPU cluster, 16 H200 GPUs, enough
  non-GPU vCPU for the support nodes (see sizing note below), and network-SSD
  disk for the filesystems
- An SSH public key to authorize on the login node

Check quota before deploying. The default Soperator sizing assumes a large
project. A constrained sandbox needs the resizing described below.

## Get the recipe at a release tag

```
git clone https://github.com/nebius/nebius-solutions-library.git
cd nebius-solutions-library
git tag --list 'soperator*' | tail -5
git checkout <release-tag>
cd soperator/installations
cp -r example my-demo && cd my-demo
```

Do not build from `master`. See troubleshooting item 1 for why.

## The tfvars values used

The full file is in `code/terraform/terraform.tfvars`. The values that define
the cluster:

```
slurm_nodeset_workers = [{
  size = 2
  resource = { platform = "gpu-h200-sxm", preset = "8gpu-128vcpu-1600gb" }
  infiniband_fabric = "eu-north2-a"
  node_local_jail_submounts = []
}]

filestore_jail = { spec = { size_gibibytes = 1024 } }
```

`node_local_jail_submounts = []` is deliberate. The example ships optional
per-worker scratch disks that are not needed here and that exceeded the sandbox
disk quota. See troubleshooting item 2.

## Sizing the support nodes for a constrained quota

The default support-node sizing did not fit a 150 non-GPU vCPU quota. The
values used here, with the reason each was chosen:

```
system     3 x 32vcpu-128gb   (validation requires a minimum of 3)
controller 1 x 8vcpu-32gb
login      1 x 16vcpu-64gb    (8vcpu is rejected as insufficient; boot disk min 256 GiB)
accounting 1 x 8vcpu-32gb
nfs        1 x 16vcpu-64gb
```

Component resource requests also had to fit those nodes:

```
mariadb: 2 cores / 12Gi   (default 8 cores / 48Gi does not fit an 8-core / 32Gi node)
rest:    8 cores / 32Gi   (default 120Gi exceeds a 128GB node's allocatable memory)
```

Network-SSD disk sizes must be a multiple of 93 GiB. Use 1023, not 1024, where
that storage class applies. See troubleshooting item 3.

## Apply

```
export NEBIUS_TENANT_ID='<tenant-id>'
export NEBIUS_PROJECT_ID='<project-id>'
export NEBIUS_REGION='eu-north2'
source ./.envrc
terraform init
terraform apply
```

`init` downloads the provider plugins. It builds nothing and takes no input from
you. `apply` is the only command that creates resources.

Expected time: 45 to 75 minutes. The machines come up first, then Kubernetes,
then Flux installs the Soperator stack, then the apply waits for the Slurm
cluster to report available. The login node IP prints at the end.

## If the apply is interrupted

A re-run can fail on a resource that was created but left out of Terraform
state, with `BucketAlreadyOwnedByYou` or a Helm "cannot re-use a name" error.
Nothing is broken. Import the existing resource into state and re-apply. See
troubleshooting item 4 for the exact import commands.

## Next

Continue to [02-validate-cluster.md](02-validate-cluster.md).
