import pandas as pd # pyright: ignore[reportMissingModuleSource]
import ast
import json
import re
import random 
from data_structures import Sample, MCQTestSample, MCQTrainingSample, SAQTestSample, SAQTrainingSample
from typing import List

MCQ_TRAIN_INPUT_FILE = "../data/train/train_dataset_mcq.csv"
SAQ_TRAIN_INPUT_FILE = "../data/train/train_dataset_saq.csv"

MCQ_TEST_INPUT_FILE = "../data/test/test_dataset_mcq.csv"
SAQ_TEST_INPUT_FILE = "../data/test/test_dataset_saq.csv"

MCQ_TRAIN_OUTPUT_FILE = "../data/train/mcq.jsonl"
SAQ_TRAIN_OUTPUT_FILE = "../data/train/saq.jsonl"
COMBINED_TRAIN_OUTPUT_FILE = "../data/train/combined.jsonl"

MCQ_TEST_OUTPUT_FILE = "../data/test/mcq.jsonl"
SAQ_TEST_OUTPUT_FILE = "../data/test/saq.jsonl"

def read_mcq_training_data():
    df = pd.read_csv(MCQ_TRAIN_INPUT_FILE)
    samples = []
    for _, row in df.iterrows():
        choices_countries = json.loads(row["choice_countries"])
        choices_text = row["choices"]
        original_country = row["country"].replace("_", " ")
        original_prompt = row["prompt"].replace("_", " ")

        for char_idx, country_name in choices_countries.items():
            if country_name == "dummy":
                continue
            # replace original country with country from other answer option
            pattern = re.compile(re.escape(original_country), re.IGNORECASE)
            augmented_prompt = pattern.sub(country_name, original_prompt)

            aug_id = f"{row['MCQID']}_AUG_{char_idx}"

            sample = MCQTrainingSample(
                id=aug_id,
                country=country_name,
                input=_get_mcq_input(augmented_prompt, choices_text),
                output=char_idx
            )
            samples.append(sample)

    return samples

def read_mcq_test_data():
    df = pd.read_csv(MCQ_TEST_INPUT_FILE)
    samples = []
    for _, row, in df.iterrows():
        sample = MCQTestSample(
            id=row["MCQID"],
            country=row["country"],
            input=_get_mcq_input(row["prompt"], row["choices"])
        )
        samples.append(sample)
    return samples

def _get_mcq_input(raw_input, choices_json):
    clean_input = raw_input.split("Without any explanation")[0].strip()
    choices_dict = json.loads(choices_json)
    options_text = "\n".join([f"{k}. {v}" for k, v in choices_dict.items()])
    return f"{clean_input}\n\n{options_text}"

def read_saq_training_data():
    df = pd.read_csv(SAQ_TRAIN_INPUT_FILE)
    samples = []
    
    for _, row in df.iterrows():
        best_answer = _get_best_saq_answer(row["annotations"])
        
        if best_answer == "<idk>" or not best_answer:
            continue

        sample = SAQTrainingSample(
            id=f"{row['ID']}",
            country=row["country"],
            input=row["en_question"],
            output=str(best_answer)
        )
        samples.append(sample)

    return samples

def _get_best_saq_answer(annotation_str):
    try:
        annotations = ast.literal_eval(annotation_str)
        if not annotations:
            return None 
            
        annotations.sort(key=lambda x: x.get("count", 0), reverse=True)
        max_count = annotations[0]["count"]

        winners = []
        for item in annotations:
            if item["count"] == max_count:
                ans_list = item.get("en_answers") or item.get("answers")
                if ans_list:
                    winners.append(ans_list[0])
            else:
                break 
        
        return min(winners, key=len) if winners else None
        
    except Exception as e:
        print(f"Error parsing annotations: {e}")
        return None

def read_saq_test_data():
    df = pd.read_csv(SAQ_TEST_INPUT_FILE)
    samples = []
    for _, row in df.iterrows():
        sample = SAQTestSample(
            id=row["ID"],
            country=row["country"],
            input=row["en_question"]
        )
        samples.append(sample)
    return samples
    
def _write_samples_to_file(samples: List[Sample], file):
    with open(file, "w") as f:
        for s in samples:
            f.write(s.to_jsonl() + "\n")

def _prepare_and_write_combined(mcq_list, saq_list, output_file):
    target_count = max(len(mcq_list), len(saq_list))
    
    if len(mcq_list) < target_count:
        mcq_final = (mcq_list * (target_count // len(mcq_list) + 1))[:target_count]
    else:
        mcq_final = mcq_list
        
    if len(saq_list) < target_count:
        saq_final = (saq_list * (target_count // len(saq_list) + 1))[:target_count]
    else:
        saq_final = saq_list

    combined = mcq_final + saq_final
    
    random.seed(3407) 
    random.shuffle(combined)
    
    print(f"Final balanced dataset size: {len(combined)} (MCQ: {len(mcq_final)}, SAQ: {len(saq_final)})")
    
    # 4. Write to file
    _write_samples_to_file(combined, output_file)

def main():
    mcq_train = read_mcq_training_data()
    saq_train = read_saq_training_data()
    _write_samples_to_file(mcq_train, MCQ_TRAIN_OUTPUT_FILE)
    _write_samples_to_file(saq_train, SAQ_TRAIN_OUTPUT_FILE)
    _prepare_and_write_combined(mcq_train, saq_train, COMBINED_TRAIN_OUTPUT_FILE)

    mcq_test = read_mcq_test_data()
    saq_test = read_saq_test_data()
    _write_samples_to_file(mcq_test, MCQ_TEST_OUTPUT_FILE)
    _write_samples_to_file(saq_test, SAQ_TEST_OUTPUT_FILE)


if __name__ == "__main__":
    main()


