# 06. Troubleshooting

Every problem hit during this build and how it was resolved. These are real, in
the order they tend to appear. Each one is something a team reproducing this work
could hit, so each has a symptom and a fix rather than a description.

## 1. Build from a release tag, not master

**Symptom:** workers stuck in `Init`, or `terraform apply` failing during the
Slurm cluster install, with one of: a CRD rejecting a field
(`.spec.plugStackConfig.pyxis.useSquashfuse: field not declared in schema`), a
missing configmap (`configmap "soperator-slurm-scripts" not found`), or `srun`
failing with `dlopen(chroot.so): cannot open shared object file`.

**Cause:** the solutions library was cloned at `master`, which had a commit ahead
of the released operator and charts. The rendered configuration referenced
fields and names the released components did not match.

**Fix:** clone and check out a tagged release so the Terraform, operator, and
charts are a matched set.
```
git tag --list 'soperator*' | tail -5
git checkout <release-tag>
```

## 2. Drop the optional per-worker scratch disks

**Symptom:** PVC provisioning fails with `ResourceExhausted` /
`compute.disk.size.network-ssd` quota exceeded; worker pods stay `Pending`.

**Cause:** the example tfvars allocates optional 1 TiB scratch disks per worker
(`node_local_jail_submounts`). On a constrained sandbox these exceed the disk
quota and are not needed for this workload.

**Fix:** set `node_local_jail_submounts = []` in the tfvars before the first
apply. If you hit this after the fact, note that Kubernetes will not let you
remove a `volumeClaimTemplate` from a live StatefulSet. Delete the StatefulSet
with `--cascade=orphan` and let the operator recreate it from the corrected
spec:
```
kubectl delete statefulset <worker-sts> -n soperator --cascade=orphan
kubectl delete pvc -l <worker-label> -n soperator
```

## 3. Network-SSD disk sizes are multiples of 93 GiB

**Symptom:** a filesystem size is rejected at apply.

**Cause:** the NETWORK_SSD_IO_M3 storage class requires sizes that are multiples
of 93 GiB.

**Fix:** use 1023 (which is 93 x 11), not 1024, where that storage class
applies.

## 4. Terraform state drift after an interrupted apply

**Symptom:** a re-run fails with `BucketAlreadyOwnedByYou`, or a Helm error
`cannot re-use a name that is still in use`. The resource exists but is not in
Terraform state.

**Cause:** an earlier apply created the resource, then was interrupted before
recording it in state.

**Fix:** import the existing resource into state, then re-apply.
```
terraform import 'module.backups_store[0].nebius_storage_v1_bucket.backups_bucket' <bucket-id>
terraform import 'module.slurm.helm_release.soperator_fluxcd_cm' flux-system/terraform-fluxcd-values
terraform apply
```
Adjust the resource addresses to match what the error names.

## 5. Triton kernel cache must be node-local

**Symptom:** a vLLM endpoint fails at startup with
`ImportError: ...__triton_launcher...so: cannot open shared object file`,
usually the fine-tuned (LoRA) endpoint.

**Cause:** inside the jail, the home directory is on the shared filesystem
(virtiofs). Triton writes a JIT kernel cache there and then loads it, and the
shared filesystem both races across concurrent jobs and breaks the
write-then-load sequence. The LoRA endpoint is the usual victim because
`--enable-lora` compiles fresh kernels.

**Fix:** point the cache at node-local disk. The serve script sets
`TRITON_CACHE_DIR` and `VLLM_CACHE_ROOT` to per-role directories under `/tmp`
(node-local inside the jail). If you hit it manually:
```
rm -rf ~/.triton
# then resubmit with TRITON_CACHE_DIR set to a /tmp path
```

## 6. Pin transformers below 5 for vLLM

**Symptom:** vLLM server crashes at startup with
`AttributeError: ... has no attribute all_special_tokens_extended`.

**Cause:** vLLM 0.10.2 allows transformers 5.x, which removed that attribute.

**Fix:** after installing vLLM, pin transformers down:
```
pip install "transformers<5"
```

## 7. CRLF line endings break sbatch

**Symptom:** `sbatch: error: Batch script contains DOS line breaks (\r\n)`.

**Cause:** a script copied from a Windows checkout carries CRLF line endings.

**Fix:**
```
sed -i 's/\r$//' <script>
```

## 8. The serving jobs expire after 8 hours

**Symptom:** endpoints that worked earlier now return connection errors or
non-200; `squeue` no longer lists the `vllm-serve` jobs.

**Cause:** the serving jobs carry an 8-hour Slurm time limit by design.

**Fix:** relaunch them (see
[04-inference-and-comparison.md](04-inference-and-comparison.md)) and wait for
the models to load.

## General notes

- A stale SSH session can show errors from a state the cluster has already moved
  past. If an error looks impossible after a repair, reconnect and retry once.
- `/tmp` on the login node is node-local and is cleared if the pod restarts.
  Keep anything that must persist under `/opt/finetune`.
- Do not run `terraform destroy` if the environment must be kept.
