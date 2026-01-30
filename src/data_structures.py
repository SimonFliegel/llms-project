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
    
    def format_alpaca_training_prompt(self, tokenizer):
        prompt = (
            f"### Task Context:\nCountry: {self.country}\nMode: {self.task_type}\n\n"
            f"### Instruction:\n{self.instruction}\n\n"
            f"### Question:\n{self.input}\n\n"
            f"### Response:\n{self.output}"
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

    def get_alpaca_inference_prompt(self):
        return (
            f"### Task Context:\nCountry: {self.country}\nMode: {self.task_type}\n\n"
            f"### Instruction:\n{self.instruction}\n\n"
            f"### Question:\n{self.input}\n\n"
            f"### Response:\n"
        )
    
class MCQTrainingSample(TrainingSample):
    def __init__(self, id, country, input, output):
        super().__init__(id, country, TaskType.MCQ, "The following is a multiple-choice question about cultural practices and preferences. Based on the specified country, select the single most accurate letter choice (A, B, C, or D).", input, output)

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
        super().__init__(id, country, TaskType.MCQ, "The following is a multiple-choice question about cultural practices and preferences. Based on the specified country, select the single most accurate letter choice (A, B, C, or D).", input)

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(
            id=data.get("id"),
            country=data.get("country"),
            input=data.get("input")
        )

class SAQTrainingSample(TrainingSample):
    def __init__(self, id, country, input, output):
        super().__init__(id, country, TaskType.SAQ, "Provide a direct and concise answer to the cultural question below. Use only a single word or short phrase that represents the general consensus in the specified country.", input, output)

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
        super().__init__(id, country, TaskType.SAQ, "Provide a direct and concise answer to the cultural question below. Use only a single word or short phrase that represents the general consensus in the specified country.", input)

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
        response = (response or "").strip().upper()
        match = re.search(r'\b([A-D])\b', response)
        if match:
            letter = match.group(1)
            return MCQChoice[letter]
        if response and response[0] in "ABCD":
            return MCQChoice[response[0]]
        return None
    
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