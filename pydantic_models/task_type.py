from enum import Enum

class TaskType(str, Enum):
    classification = "classification"
    regression = "regression"
