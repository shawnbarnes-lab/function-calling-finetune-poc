"""Download Qwen3-32B plus a function-calling dataset and write the formatted
training set to the shared filesystem.

Tries Salesforce xLAM-function-calling-60k first (gated -- accept the terms on
HF), falls back to glaive-function-calling-v2. The Qwen3 chat template is
pre-applied with enable_thinking=False so the model doesn't learn to wrap
tool calls in <think> traces.

    huggingface-cli login --token "$HF_TOKEN"
    python prepare_data.py [--dataset xlam|glaive]
"""

import argparse
import json
import os

from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


SHARED_FS = "/opt/finetune"

# Qwen3-32B is instruction-tuned already, there's no separate -Instruct repo
MODEL_REPO = "Qwen/Qwen3-32B"
MODEL_DIR = f"{SHARED_FS}/models/qwen3-32b"

DATASETS = {
    "xlam": {
        "repo": "Salesforce/xlam-function-calling-60k",
        "local_dir": f"{SHARED_FS}/data/xlam-function-calling",
        "gated": True,
        "gate_url": "https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k",
        "format_fn": "format_xlam",
    },
    "glaive": {
        "repo": "glaiveai/glaive-function-calling-v2",
        "local_dir": f"{SHARED_FS}/data/glaive-function-calling",
        "gated": False,
        "gate_url": None,
        "format_fn": "format_glaive",
    },
}

SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful assistant with access to the following functions. "
    "When the user asks something that requires calling a function, respond "
    "with a JSON object (or list of objects) representing the function call(s). "
    "If the user asks something conversational that does not need a function, "
    "respond in plain text.\n\n"
    "Available functions:\n{tools_json}"
)


def parse_json_field(field_value, default=None):
    # xLAM stores tools/answers as JSON strings, sometimes malformed
    if isinstance(field_value, (list, dict)):
        return field_value
    if not isinstance(field_value, str):
        return default if default is not None else []
    try:
        return json.loads(field_value)
    except json.JSONDecodeError:
        return default if default is not None else []


def format_xlam(example):
    # xLAM rows: query (text), tools (JSON string), answers (JSON string)
    query = (example.get("query") or "").strip()
    tools = parse_json_field(example.get("tools"), default=[])
    answers = parse_json_field(example.get("answers"), default=[])

    if not query or not answers:
        return None

    tools_pretty = json.dumps(tools, indent=2) if tools else "[]"
    system_content = SYSTEM_PROMPT_TEMPLATE.format(tools_json=tools_pretty)
    assistant_content = json.dumps(answers, separators=(",", ":"))

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query},
        {"role": "assistant", "content": assistant_content},
    ]


def format_glaive(example):
    # Glaive rows are plain text with USER:/ASSISTANT:/FUNCTION RESPONSE:
    # prefixes, so this parse is necessarily a bit crude
    system_msg = (example.get("system") or "").strip()
    chat = (example.get("chat") or "").strip()
    if not chat:
        return None

    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})

    chat = chat.replace("USER: ", "<|user|>")
    chat = chat.replace("ASSISTANT: ", "<|assistant|>")
    chat = chat.replace("FUNCTION RESPONSE: ", "<|function|>")
    chat = chat.replace("<|endoftext|>", "")

    for part in chat.split("<|"):
        if part.startswith("user|>"):
            messages.append({"role": "user", "content": part[6:].strip()})
        elif part.startswith("assistant|>"):
            messages.append({"role": "assistant", "content": part[11:].strip()})
        elif part.startswith("function|>"):
            messages.append({"role": "tool", "content": part[10:].strip()})

    if len(messages) < 2:
        return None
    return messages


PARSERS = {
    "format_xlam": format_xlam,
    "format_glaive": format_glaive,
}


def try_load_dataset(key):
    cfg = DATASETS[key]
    print(f"\nAttempting to load: {cfg['repo']}")
    try:
        raw = load_dataset(cfg["repo"])
        return raw, cfg
    except Exception as e:
        if cfg["gated"]:
            print(
                f"[WARN] Could not load {cfg['repo']} (gated dataset).\n"
                f"  Accept the terms at {cfg['gate_url']} and re-run.\n"
                f"  Underlying error: {e}\n"
            )
        else:
            print(f"[WARN] Could not load {cfg['repo']}: {e}")
        return None, None


def acquire_dataset(requested):
    if requested != "auto":
        raw, cfg = try_load_dataset(requested)
        if raw is None:
            raise RuntimeError(f"Could not load requested dataset: {requested}")
        return raw, cfg

    for key in ("xlam", "glaive"):
        raw, cfg = try_load_dataset(key)
        if raw is not None:
            print(f"[OK] Using dataset: {cfg['repo']}")
            return raw, cfg

    raise RuntimeError("No dataset could be loaded. Check HF auth and dataset terms.")


def make_text_builder(tokenizer, format_fn):
    def build(example):
        messages = format_fn(example)
        if not messages:
            return {"text": ""}
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            # tokenizer doesn't know enable_thinking (non-Qwen3 fallback)
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        return {"text": text}
    return build


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["auto", "xlam", "glaive"],
        default="auto",
        help="'auto' tries xLAM first, falls back to Glaive",
    )
    args = parser.parse_args()

    os.makedirs(SHARED_FS, exist_ok=True)

    print(f"\nDownloading model: {MODEL_REPO}")
    snapshot_download(repo_id=MODEL_REPO, local_dir=MODEL_DIR)
    print(f"  saved to {MODEL_DIR}")

    raw_dataset, cfg = acquire_dataset(args.dataset)
    dataset_dir = cfg["local_dir"]
    format_fn = PARSERS[cfg["format_fn"]]

    print("Formatting examples (Qwen3 chat template, thinking disabled)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    build = make_text_builder(tokenizer, format_fn)
    formatted = raw_dataset.map(
        build,
        remove_columns=raw_dataset["train"].column_names,
        num_proc=16,
    )

    before = len(formatted["train"])
    formatted = formatted.filter(lambda ex: len(ex.get("text") or "") > 50)
    after = len(formatted["train"])
    print(f"  filtered {before - after} malformed records ({after} remaining)")

    if "validation" not in formatted:
        train = formatted["train"].shuffle(seed=42)
        eval_size = min(500, len(train) // 20)
        eval_split = train.select(range(eval_size))
        train_split = train.select(range(eval_size, len(train)))
        from datasets import DatasetDict
        formatted = DatasetDict({"train": train_split, "validation": eval_split})

    formatted.save_to_disk(dataset_dir)

    print(f"\nDataset saved to {dataset_dir}")
    print(f"Train examples: {len(formatted['train']):,}")
    print(f"Eval examples:  {len(formatted['validation']):,}")
    print(f"Dataset used: {cfg['repo']}")


if __name__ == "__main__":
    main()
