"""Run the test prompts against two vLLM endpoints (base Qwen3-32B vs the
LoRA fine-tune) and write a side-by-side markdown report.

    python compare.py --base http://<node>:8000/v1 \\
        --finetuned http://<node>:8001/v1 \\
        --prompts test_prompts.json --output results/comparison_output.md
"""

import argparse
import json
import time
from pathlib import Path

import requests


SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant with access to the following functions. When the user asks something that requires calling a function, respond with a JSON object containing the function name and parameters. If the user asks something that does not require a function call, respond conversationally.

Available functions:
{tools_json}"""


def call_endpoint(endpoint, model, system, user, timeout=120):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
        # Qwen3's default template emits <think> traces; we want bare JSON.
        # vLLM passes this through to apply_chat_template.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    start = time.time()
    try:
        r = requests.post(f"{endpoint}/chat/completions", json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        latency_ms = int((time.time() - start) * 1000)
        return {"text": text, "latency_ms": latency_ms, "ok": True}
    except Exception as e:
        return {"text": f"ERROR: {e}", "latency_ms": 0, "ok": False}


def list_model(endpoint):
    # a LoRA-enabled endpoint lists the base model first and the adapter
    # after it; the adapter is the one we want to hit
    try:
        r = requests.get(f"{endpoint}/models", timeout=10)
        r.raise_for_status()
        ids = [m["id"] for m in r.json()["data"]]
        for mid in ids:
            if "lora" in mid.lower():
                return mid
        return ids[0]
    except Exception as e:
        return f"unknown ({e})"


def is_valid_function_call(text):
    # accept a dict or a list of dicts as long as every call has a name.
    # xLAM-style output is a list; base Qwen tends to emit a single object,
    # often wrapped in a ```json fence
    text = text.strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") or part.startswith("["):
                text = part
                break
    if not (text.startswith("{") or text.startswith("[")):
        return False
    try:
        parsed = json.loads(text)
    except Exception:
        return False
    calls = parsed if isinstance(parsed, list) else [parsed]
    return len(calls) > 0 and all(
        isinstance(c, dict) and "name" in c for c in calls
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="vLLM endpoint serving base model")
    parser.add_argument("--finetuned", required=True, help="vLLM endpoint serving fine-tuned model")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    endpoints = {"base": args.base, "finetuned": args.finetuned}

    prompts_data = json.loads(Path(args.prompts).read_text())
    tools = prompts_data["tools"]
    prompts = prompts_data["prompts"]
    system = SYSTEM_PROMPT_TEMPLATE.format(tools_json=json.dumps(tools, indent=2))

    model_names = {name: list_model(url) for name, url in endpoints.items()}
    for name, model in model_names.items():
        print(f"{name}: {model}")
    print(f"Testing {len(prompts)} prompts against both endpoints")

    results = []
    for i, p in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] {p['category']}: {p['user'][:60]}...")
        row = {"prompt": p}
        for name, url in endpoints.items():
            response = call_endpoint(url, model_names[name], system, p["user"])
            response["valid_call"] = is_valid_function_call(response["text"])
            row[name] = response
        results.append(row)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("# Inference Comparison: Base vs Fine-tuned (Qwen3-32B)\n\n")
        for name, model in model_names.items():
            f.write(f"- **{name}**: `{model}`\n")
        f.write(f"\n## Test set: {len(prompts)} prompts\n\n")

        f.write("## Aggregate results\n\n")
        f.write("| Endpoint | Valid function calls | Avg latency (ms) | Success |\n")
        f.write("|---|---|---|---|\n")
        for name in endpoints:
            valid = sum(1 for r in results if r[name].get("valid_call"))
            ok = sum(1 for r in results if r[name]["ok"])
            avg_latency = (
                sum(r[name]["latency_ms"] for r in results if r[name]["ok"]) / max(ok, 1)
            )
            f.write(f"| {name} | {valid}/{len(results)} | {avg_latency:.0f} | {ok}/{len(results)} |\n")

        f.write("\n---\n\n")

        for r in results:
            p = r["prompt"]
            f.write(f"## Prompt {p['id']} - {p['category']}\n\n")
            f.write(f"**User:** {p['user']}\n\n")
            for name in endpoints:
                resp = r[name]
                valid_tag = "[valid function call]" if resp.get("valid_call") else "[invalid]"
                f.write(f"### {name} ({resp['latency_ms']}ms) {valid_tag}\n\n")
                f.write(f"```\n{resp['text']}\n```\n\n")
            f.write("---\n\n")

    print(f"Comparison written to {args.output}")


if __name__ == "__main__":
    main()
