# 05. Monitoring

Where to see GPU utilization, power, and logs. All of this came with the cluster.
No exporters were installed and no dashboards were built.

## Three places the same run is visible

1. **nvidia-smi on a node**, for an instantaneous view during a job:
   ```
   srun --jobid=<jobid> --overlap -w worker-0 nvidia-smi
   ```
   Shows per-GPU utilization, power draw, and memory on an allocated node.

2. **Grafana with DCGM**, shipped with the cluster, for a time-series view.
   Open a tunnel from your workstation, then browse to it:
   ```
   ssh -i <SSH_KEY> -L 3000:metrics-grafana.monitoring-system.svc:80 -N root@<LOGIN_NODE_IP>
   ```
   The `ssh -N` command prints nothing and does not return; a silent, hanging
   window means the tunnel is up. Open `http://localhost:3000` in a browser. The
   admin password is stored in a Kubernetes secret in the cluster, not in this
   repo. Look for the GPU / DCGM dashboard.

3. **Nebius console**, for the platform's own telemetry. Compute, then the H200
   instance, then the Monitoring tab. Graphs for GPU utilization, power, and
   memory per node.

## Reading the utilization trace

During the training run the trace shows:

- A vertical rise when the job starts.
- A flat plateau near 96% for the length of the run.
- Two brief dips at the epoch boundaries, where training pauses to evaluate and
  checkpoint.
- A drop to idle when the job completes.

Power draw held near 670 watts per GPU against a 700 watt cap. Power is the most
reliable signal that real work happened; sustained near-cap power for over an
hour cannot come from an idle allocation.

Memory occupancy was high (around 120 of 141 GB per GPU). That is the full model
copy plus the batch, not a sign of memory pressure. The run was compute-bound,
which is the intended state: the limit was arithmetic throughput, not memory
bandwidth.

## What the assignment asked for

The target was over 80% GPU utilization. The run held about 96% sustained,
confirmed by all three views above. See
[07-evidence-index.md](07-evidence-index.md) for how to verify each figure.
