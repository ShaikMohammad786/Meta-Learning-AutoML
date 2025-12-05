from pydantic import BaseModel
from pydantic_models.task_type import TaskType

class DatasetUploadResponse(BaseModel):
    original_name : str
    stored_name :str
    user_id :str
    path : str
    task_type: TaskType
    
class TrainDataset(BaseModel):
    dataset: DatasetUploadResponse
    target_col: str
    tuning: bool = False