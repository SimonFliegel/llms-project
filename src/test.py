import torch # pyright: ignore[reportMissingImports]
import re
import sys
import os
import pandas as pd # pyright: ignore[reportMissingModuleSource]
import json
import csv
from pathlib import Path
from peft import PeftModel # pyright: ignore[reportMissingImports]
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, logging # pyright: ignore[reportMissingImports]
from huggingface_hub import login  # pyright: ignore[reportMissingImports]
from tqdm import tqdm # pyright: ignore[reportMissingModuleSource]
from data_structures import MCQTestSample, SAQTestSample, MCQAnswer, SAQAnswer, MCQChoice, TaskType
from typing import Type

INPUT_MCQ = "../data/test/mcq.jsonl"
OUTPUT_NAME_MCQ = "mcq_prediction.tsv"
INPUT_SAQ = "../data/test/saq.jsonl"
OUTPUT_NAME_SAQ = "saq_prediction.tsv"

FT_MODELS_DIR = "../models/"
BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B"

def load_model_and_tokenizer(model_id):
    login(token=open("../.hf_token", "r").read().strip())

    print(f"Loading Base: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token # padding is needed to process batches
    tokenizer.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    is_adapter = False
    model_dir = os.path.join(FT_MODELS_DIR + model_id)
    if os.path.isdir(model_dir):
        if os.path.exists(os.path.join(model_dir, "adapter_config.json")):
            is_adapter = True

    if is_adapter:
        print(f"Merging LoRA Adapter '{model_id}'...")
        model = PeftModel.from_pretrained(base_model, model_dir)
    elif model_id != BASE_MODEL_ID:
        print(f"Loading specific weights from {model_id}...")
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
    else:
        print("using Base...")
        model = base_model

    return model, tokenizer


def test_task(model_id: str, task_type: TaskType):
    model, tokenizer = load_model_and_tokenizer(model_id)

    if task_type == TaskType.MCQ:
        input_file = INPUT_MCQ
        sample_class = MCQTestSample
        answer_class = MCQAnswer
        output_name = OUTPUT_NAME_MCQ
    else:
        input_file = INPUT_SAQ
        sample_class = SAQTestSample
        answer_class = SAQAnswer
        output_name = OUTPUT_NAME_SAQ

    samples = []
    print(f"Loading {task_type} samples from {input_file}...")
    with open(input_file, "r") as f:
        for line in f:
            samples.append(sample_class.from_jsonl(line))
        
    prompts = [s.get_inference_prompt() for s in samples]

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        return_full_text=False
    )
    print(f"Running inference on {len(prompts)} items...")
    results = gen(
        prompts,
        max_new_tokens=64,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        batch_size=32
    )

    folder_name = model_id.replace("/", "_") if model_id != BASE_MODEL_ID else "base"
    output_path = Path(f"../answers/{folder_name}/{output_name}")
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with output_path.open("w") as out:
        out.write(answer_class.header_row() + "\n")

        for sample, resp in zip(samples, tqdm(results)):
            generated_text = resp[0].get("generated_text", "").strip()

            if task_type == TaskType.MCQ:
                choice = answer_class.extract_answer(generated_text) # type: ignore always MCQAnswer
                answer = MCQAnswer(sample.id, choice) # type: ignore
            else:
                answer = SAQAnswer(sample.id, generated_text.replace("\n", " "))

            out.write(answer.to_tsv() + "\n")

    print(f"Predictions written to {output_path}.")



if __name__ == "__main__":
    m_id = sys.argv[1].strip() if len(sys.argv) > 1 else BASE_MODEL_ID
    mode_raw = sys.argv[2].lower() if len(sys.argv) > 2 else "mcq"
    
    try:
        mode_enum = TaskType(mode_raw) 
        test_task(m_id, mode_enum)
        
    except ValueError:
        valid_options = [t.value for t in TaskType]
        print(f"Error: Invalid mode '{mode_raw}'.")
        print(f"Supported modes are: {valid_options}")
        sys.exit(1)






