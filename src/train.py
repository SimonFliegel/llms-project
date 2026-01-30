import torch # pyright: ignore[reportMissingImports]
import sys
import json
import os
from typing import List
from huggingface_hub import login # pyright: ignore[reportMissingImports]
from unsloth import FastLanguageModel # pyright: ignore[reportMissingImports]
from trl import SFTTrainer, SFTConfig # pyright: ignore[reportMissingImports]
from datasets import Dataset # pyright: ignore[reportMissingImports]
from data_structures import TaskType, TrainingSample

BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B"
MAX_SEQ_LENGTH = 2048
DTYPE = torch.bfloat16
LOAD_IN_4BIT = False

DATA_DIR = "../data/train"
TASK_FILES = {
    TaskType.MCQ: "mcq.jsonl",
    TaskType.SAQ: "saq.jsonl",
    "combined": "combined.jsonl"
}

def hf_login(token_path="../.hf_token"):
    if os.path.exists(token_path):
        token = open(token_path, "r").read().strip()
        login(token=token)

def build_dataset_from_jsonl(task_mode: str) -> Dataset:
    file_name = TASK_FILES.get(task_mode)
    file_path = os.path.join(DATA_DIR, file_name) # type: ignore
    
    print(f"Loading {task_mode} training data from {file_path}...")
    
    raw_dicts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            obj = TrainingSample.from_dict(data)
            raw_dicts.append(obj.to_dict())
            
    return Dataset.from_list(raw_dicts)

def load_model_and_tokenizer():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    return model, tokenizer

def add_lora_adapters(model):
    return FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "lm_head"
        ],
        lora_alpha=128,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=True,
    )

def format_prompts(examples, tokenizer):
    texts = []
    for i in range(len(examples["input"])):
        sample_obj = TrainingSample(
            id=examples["id"][i],
            country=examples["country"][i],
            task_type=examples["task_type"][i],
            instruction=examples["instruction"][i],
            input=examples["input"][i],
            output=examples["output"][i]
        )
        prompt = sample_obj.format_alpaca_training_prompt(tokenizer)
        texts.append(prompt)
    
    return {"text": texts}

def train_and_save(model, tokenizer, dataset, ft_model_dir, max_seq_length):
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        args = SFTConfig(
            output_dir = ft_model_dir,
            per_device_train_batch_size = 16,
            gradient_accumulation_steps = 4,
            warmup_ratio=0.1,
            num_train_epochs = 5,
            learning_rate = 1e-5,
            lr_scheduler_type = "cosine",
            bf16 = True,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.1,
            neftune_noise_alpha = 5.0,
            seed = 3407,
        ),
    )
    trainer.train()
    model.save_pretrained(ft_model_dir)
    tokenizer.save_pretrained(ft_model_dir)
    print(f"Training complete. Model saved to {ft_model_dir}")

def main(mode_input: str):
    if mode_input == "combined":
        task_mode = "combined"
    else:
        task_mode = TaskType(mode_input)

    hf_login()
    ft_model_dir = f"../models/ft-{mode_input}"
    
    dataset = build_dataset_from_jsonl(task_mode) # type: ignore
    model, tokenizer = load_model_and_tokenizer()
    model = add_lora_adapters(model)

    dataset = dataset.map(
        format_prompts, 
        batched=True,
        fn_kwargs={"tokenizer": tokenizer}
    )

    train_and_save(model, tokenizer, dataset, ft_model_dir, MAX_SEQ_LENGTH)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py [mcq|saq|combined]")
        sys.exit(1)

    raw_arg = sys.argv[1].lower().strip()

    valid_task_values = [t.value for t in TaskType] 
    
    if raw_arg == "combined" or raw_arg in valid_task_values:
        print(f"Starting training pipeline for mode: {raw_arg}")
        main(raw_arg)
    else:
        print(f"Error: '{raw_arg}' is not a valid training mode.")
        print(f"Allowed modes: {['combined'] + valid_task_values}")
        sys.exit(1)