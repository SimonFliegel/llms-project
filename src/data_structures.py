import json
from abc import ABC, abstractmethod
from enum import Enum
from typing_extensions import Self # pyright: ignore[reportMissingModuleSource]
import re

class TaskType(str, Enum):
    MCQ = "mcq"
    SAQ = "saq"

class Sample(ABC):
    def __init__(self, id, country, task_type, instruction, input):
        self.id = id
        self.country = country
        self.task_type = task_type
        self.instruction = instruction
        self.input = input

    @abstractmethod
    def to_dict(self) -> dict:
        pass

    def to_jsonl(self):
        return json.dumps(self.to_dict())
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> Self:
        pass

    @classmethod
    def from_jsonl(cls, json_str: str):
        return cls.from_dict(json.loads(json_str))

class TrainingSample(Sample):
    def __init__(self, id, country, task_type, instruction, input, output):
        super().__init__(id, country, task_type, instruction, input)
        self.output = output

    def to_dict(self):
        return {
            "id": self.id,
            "country": self.country,
            "task_type": self.task_type,
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(
            id=data.get("id"),
            country=data.get("country"),
            task_type=TaskType(data.get("task_type")),
            instruction=data.get("instruction"),
            input=data.get("input"),
            output=data.get("output")
        )
    
    @staticmethod
    def format_for_llama(instruction, input_text, output_text, tokenizer):
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output_text}"
        )
        return prompt + tokenizer.eos_token

class TestSample(Sample):
    def __init__(self, id, country, task_type, instruction, input):
        super().__init__(id, country, task_type, instruction, input)

    def to_dict(self):
        return {
            "id": self.id,
            "country": self.country,
            "task_type": self.task_type,
            "instruction": self.instruction,
            "input": self.input,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(
            id=data.get("id"),
            country=data.get("country"),
            task_type=TaskType(data.get("task_type")),
            instruction=data.get("instruction"),
            input=data.get("input"),
        )

    def get_inference_prompt(self):
        return (
            f"### Instruction:\n{self.instruction}\n\n"
            f"### Input:\n{self.input}\n\n"
            f"### Response:\n"
        )
    
class MCQTrainingSample(TrainingSample):
    def __init__(self, id, country, input, output):
        super().__init__(id, country, TaskType.MCQ, "You are a cultural expert. Select the correct option for the following question. Provide the answer in the requested JSON format.", input, output)

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(
            id=data.get("id"),
            country=data.get("country"),
            input=data.get("input"),
            output=data.get("output")
        )

class MCQTestSample(TestSample):
    def __init__(self, id, country, input):
        super().__init__(id, country, TaskType.MCQ, "You are a cultural expert. Select the correct option for the following question. Provide the answer in the requested JSON format.", input)

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(
            id=data.get("id"),
            country=data.get("country"),
            input=data.get("input")
        )

class SAQTrainingSample(TrainingSample):
    def __init__(self, id, country, input, output):
        super().__init__(id, country, TaskType.SAQ, "Answer the following question concisely and accurately.", input, output)

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(
            id=data.get("id"),
            country=data.get("country"),
            input=data.get("input"),
            output=data.get("output")
        )

class SAQTestSample(TestSample):
    def __init__(self, id, country, input):
        super().__init__(id, country, TaskType.SAQ, "Answer the following question concisely and accurately.", input)

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(
            id=data.get("id"),
            country=data.get("country"),
            input=data.get("input")
        )

class MCQChoice(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"

class Answer(ABC):
    @classmethod
    @abstractmethod
    def header_row(cls) -> str:
        pass

    @abstractmethod
    def to_tsv(self) -> str:
        pass


class MCQAnswer():
    def __init__(self, mcqid: str, answer: MCQChoice):
        self.mcqid = mcqid
        self.answer = answer

    @classmethod
    def header_row(cls):
        return f"MCQID\tA\tB\tC\tD"
    
    @classmethod
    def extract_answer(cls, response: str) -> MCQChoice | None:
        response = (response or "").strip()
        try:
            clean_text = re.sub(r'^```json\s*|\s*```$', '', response, flags=re.MULTILINE)
            obj = json.loads(clean_text)
            if isinstance(obj, dict):
                v = obj.get("answer_choice") or obj.get("answer")
                if v and str(v).upper() in "ABCD":
                    return MCQChoice[str(v).upper()]
        except:
            pass
    
    def to_tsv(self):
        choices = [
            "True" if self.answer == choice else "False" 
            for choice in MCQChoice
        ]
        return f"{self.mcqid}\t" + "\t".join(choices)

class SAQAnswer():
    def __init__(self, id, answer: str):
        self.id = id
        self.answer = answer

    @classmethod
    def header_row(cls) -> str:
        return "ID\tanswer"

    def to_tsv(self) -> str:
        return f"{self.id}\t{self.answer}"