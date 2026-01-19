import pandas as pd # pyright: ignore[reportMissingModuleSource]
import ast
import json
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
        sample = MCQTrainingSample(
            id=row["MCQID"],
            country=row["country"],
            input=_get_mcq_input(row["prompt"], row["choices"]),
            output=json.dumps({"answer_choice": row["answer_idx"]})
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
    clean_input = raw_input.split("Without any explaination")[0].strip()
    choices_dict = json.loads(choices_json)
    options_text = "\n".join([f"{k}. {v}" for k, v in choices_dict.items()])
    return f"{clean_input}\n\n{options_text}"

def read_saq_training_data():
    df = pd.read_csv(SAQ_TRAIN_INPUT_FILE)
    samples = []
    for _, row in df.iterrows():
        sample = SAQTrainingSample(
            id=row["ID"],
            country=row["country"],
            input=row["en_question"],
            output=_get_best_saq_answer(row["annotations"])
        )
        samples.append(sample)
    return samples

def _get_best_saq_answer(annotation_str):
    try:
        annotations = ast.literal_eval(annotation_str)
        if not annotations:
            return "<idk>" # TODO: What to set here for scoring if there is no answer?
        annotations.sort(key=lambda x: x.get("count", 0), reverse=True)

        max_count = annotations[0]["count"]

        winners = []
        for item in annotations:
            if item["count"] == max_count:
                ans = item.get("en_answers", []) or item.get("answers", [])
                if ans:
                    winners.append(ans[0])
            else:
                break # only take answers with max count
        return random.choice(winners)
    except:
        print("Failed getting best answers for annotation string: " + annotation_str)
        return []

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

def main():
    mcq_train = read_mcq_training_data()
    saq_train = read_saq_training_data()
    _write_samples_to_file(mcq_train, MCQ_TRAIN_OUTPUT_FILE)
    _write_samples_to_file(saq_train, SAQ_TRAIN_OUTPUT_FILE)
    _write_samples_to_file(mcq_train + saq_train, COMBINED_TRAIN_OUTPUT_FILE)

    mcq_test = read_mcq_test_data()
    saq_test = read_saq_test_data()
    _write_samples_to_file(mcq_test, MCQ_TEST_OUTPUT_FILE)
    _write_samples_to_file(saq_test, SAQ_TEST_OUTPUT_FILE)


if __name__ == "__main__":
    main()


