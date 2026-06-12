"""LoRA fine-tune of Qwen3-32B for function calling.

Launched by train.slurm via torchrun, 2 nodes x 8 H200 with DDP.
See docs/03-run-training.md for the full workflow.
"""

import argparse
import os

import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# the shared 1TB filesystem is the root mount inside the Soperator jail,
# so this path resolves the same on login and worker nodes
SHARED_FS = "/opt/finetune"

# prepare_data.py writes whichever dataset it managed to pull
DATASET_PATH_CANDIDATES = [
    f"{SHARED_FS}/data/xlam-function-calling",
    f"{SHARED_FS}/data/glaive-function-calling",
]

MODEL_CONFIGS = {
    # 12b kept around for smoke tests; the actual run is 32b
    "12b": {
        "repo": "mistralai/Mistral-Nemo-Instruct-2407",
        "local_dir": f"{SHARED_FS}/models/mistral-nemo-12b",
        "output_dir": f"{SHARED_FS}/checkpoints/mistral-nemo-12b-xlam-lora",
        "run_name": "mistral-nemo-12b-xlam-lora-ddp",
        "lora_target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    },
    "32b": {
        "repo": "Qwen/Qwen3-32B",
        "local_dir": f"{SHARED_FS}/models/qwen3-32b",
        "output_dir": f"{SHARED_FS}/checkpoints/qwen3-32b-xlam-lora",
        "run_name": "qwen3-32b-xlam-lora-ddp",
        # same module names as Qwen2.5
        "lora_target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    },
}


def resolve_dataset_path():
    for path in DATASET_PATH_CANDIDATES:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        "No prepared dataset found, run prepare_data.py first. "
        f"Looked in: {DATASET_PATH_CANDIDATES}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for function calling")
    parser.add_argument("--model-size", choices=["12b", "32b"], required=True)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--per-device-batch-size", type=int, default=None,
                        help="override the per-size default")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    return parser.parse_args()


def default_batch_size(model_size):
    # starting points -- raise until OOM, then back off one notch
    return {"12b": 8, "32b": 4}[model_size]


def main():
    args = parse_args()
    config = MODEL_CONFIGS[args.model_size]

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main_process = rank == 0

    if is_main_process:
        print(f"model={config['repo']} world_size={world_size} "
              f"lora_rank={args.lora_rank} max_len={args.max_seq_length}")
        print(f"output: {config['output_dir']}")

    tokenizer = AutoTokenizer.from_pretrained(config["local_dir"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Plain DDP, no FSDP: the base is frozen under LoRA so there is nothing
    # worth sharding -- FSDP would all-gather ~64 GB of dead weights per step
    # over IB. Full replica per GPU fits fine in 141 GB.
    base_model = AutoModelForCausalLM.from_pretrained(
        config["local_dir"],
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # close enough to FA2 here, no build dep
        use_cache=False,  # incompatible with gradient checkpointing
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        target_modules=config["lora_target_modules"],
    )
    model = get_peft_model(base_model, lora_config)

    if is_main_process:
        model.print_trainable_parameters()

    dataset_path = resolve_dataset_path()
    if is_main_process:
        print(f"Loading dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)
    train_dataset = dataset["train"]
    eval_dataset = dataset.get("validation") or train_dataset.select(range(500))

    if is_main_process:
        print(f"train={len(train_dataset):,} eval={len(eval_dataset):,}")

    per_device_bs = args.per_device_batch_size or default_batch_size(args.model_size)

    # Batch size, packing and the epoch-only eval cadence are what keep
    # utilization above the 80% bar. If there's memory headroom, bump the
    # batch size before touching anything else.
    training_args = SFTConfig(
        output_dir=config["output_dir"],
        run_name=config["run_name"],
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=per_device_bs,
        gradient_accumulation_steps=1,
        bf16=True,
        tf32=True,
        # needed at 32b / bs4 / 4096 even on 141 GB
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_seq_length,
        packing=True,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=50,
        eval_strategy="epoch",
        dataloader_num_workers=8,
        dataloader_prefetch_factor=4,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=200,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # only the adapter gets saved, the base is unchanged
    if is_main_process:
        final_dir = f"{config['output_dir']}/final"
        print(f"Saving LoRA adapter to {final_dir}")
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
