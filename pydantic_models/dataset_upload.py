from pydantic import BaseModel


class DatasetUploadResponse(BaseModel):
    original_name : str
    stored_name :str
    user_id :str
    path : str