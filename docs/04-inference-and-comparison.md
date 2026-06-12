# 04. Inference and comparison

Serve the base and fine-tuned models, then score both on the same prompts. The
comparison is the result that answers the use case. Token accuracy from training
is only a health signal; this is the test that matters.

## One-time vLLM environment

vLLM pins its own torch build, so it gets a separate virtual environment from
training.

```
python3 -m venv /opt/finetune/venv-vllm
/opt/finetune/venv-vllm/bin/pip install vllm==0.10.2
/opt/finetune/venv-vllm/bin/pip install "transformers<5"
```

The `transformers<5` pin is required. vLLM 0.10.2 allows transformers 5.x, which
removed an attribute vLLM still calls and crashes the server at startup. See
troubleshooting item 6.

## Launch both endpoints as Slurm jobs

vLLM runs as Slurm jobs, not separate Kubernetes deployments. On a Soperator
cluster Slurm owns the GPUs, so serving through the scheduler keeps inference in
the same capacity accounting as training.

```
cd /opt/finetune
sbatch --export=ALL,ROLE=base,PORT=8000 vllm_serve.slurm
sbatch --export=ALL,ROLE=finetuned,PORT=8001 vllm_serve.slurm
grep SERVING_ON vllm-*.out
```

Each endpoint uses 2 GPUs with tensor parallelism. The base endpoint serves
Qwen3-32B unchanged. The fine-tuned endpoint serves the same base weights with
the LoRA adapter loaded via `--enable-lora`; it is not a second full model.

Note the node each job lands on. Later commands must use that node, not a
hardcoded `worker-1`.

## Confirm both are up

```
curl -s -m 5 -o /dev/null -w "%{http_code}\n" http://<node>:8000/v1/models
curl -s -m 5 -o /dev/null -w "%{http_code}\n" http://<node>:8001/v1/models
```

`200` means alive. Anything else, or a timeout, means the endpoint is not ready;
wait or check the job log. The `-m 5` timeout matters: without it, a curl
against a dead endpoint hangs. Models take 10 to 15 minutes to load.

A full model list on the fine-tuned endpoint lists both the base and the
`qwen3-32b-xlam-lora` adapter:

```
curl -s http://<node>:8001/v1/models
```

## Run the comparison

```
cd /opt/finetune
./venv/bin/python compare.py \
  --base http://<node>:8000/v1 \
  --finetuned http://<node>:8001/v1 \
  --prompts test_prompts.json \
  --output results/comparison_output.md
head -16 results/comparison_output.md
```

This sends 20 tool-use prompts to each endpoint at temperature 0 and scores each
response on one thing: does it parse as a valid function call in the expected
schema.

### Result

Base model: 3 of 20 valid function calls. Fine-tuned: 16 of 20.

The base model's common failure is wrapping the answer in a markdown code fence
and using its own field names (`function` and `parameters` instead of the xLAM
schema's `name` and `arguments`), so the JSON fails to parse even when the
content is close. The fine-tune emits the expected schema directly.

Four of the fine-tuned misses are prompts that do not require a function call,
where plain text is the correct answer; the scorer counts only valid JSON by
design. The honest read is that the fine-tune traded some general conversation
quality for function-call reliability. For production, route non-tool turns to
the base model or mix general chat data into training.

## Single live request

```
curl -s http://<node>:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @demo_request.json \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

Expected: one line, the function calls in the xLAM schema.

## Relaunching after the time limit

The serving jobs carry an 8-hour Slurm time limit. They are not a permanent
service. To bring them back, re-run the two `sbatch` lines above and wait for
`Application startup complete` in the job logs. See troubleshooting item 5 for
the Triton cache requirement that the serve script already handles.

## Next

Reference [05-monitoring.md](05-monitoring.md) for where to see utilization, and
[07-evidence-index.md](07-evidence-index.md) for where each result is recorded.
