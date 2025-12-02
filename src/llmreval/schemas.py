from pydantic import BaseModel
from typing import Optional, Literal, Dict

TaskType = Literal["qa", "summarization"]

class Item(BaseModel):
    id: str
    task: TaskType
    dataset: str
    prompt: str
    gold: Optional[str] = None
    source: Optional[str] = None      # for summarization
    meta: Dict = {}

class Prediction(BaseModel):
    id: str
    task: TaskType
    dataset: str
    variant: Literal["clean","perturbed"]
    prompt: str
    prediction: str
    raw: Dict = {}
