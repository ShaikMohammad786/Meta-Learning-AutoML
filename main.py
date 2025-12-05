import os
import shutil
import json
import bcrypt
import threading
import logging
from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File, Form,APIRouter, Query
from fastapi.responses import FileResponse
from pymongo import MongoClient
from dotenv import load_dotenv
from constants import *
from pydantic_models.user_model import UserRegister, UserLogin
from pydantic_models.dataset_upload import DatasetUploadResponse, TaskType , TrainDataset
from utils.jwt_handler import create_access_token, verify_token
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
import pandas as pd
from user_section.main import User
from user_section.training.status_tracker import TrainingStatusTracker
from io import BytesIO
from pathlib import Path
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

client = MongoClient(os.getenv("MONGO_DB_URI"))
db = client["MetaML"]
user_collection = db["users"]


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
def root():
    return {"msg": "Welcome to Metaml"}


@app.post("/users/register")
def register(user: UserRegister):

    print("PASSWORD LENGTH:", len(user.password))

    if not all(
        [
            user.fname,
            user.lname,
            user.username,
            user.email,
            user.password,
            user.cpassword,
        ]
    ):
        raise HTTPException(status_code=400, detail="all fields are required")

    if user_collection.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="username already taken")

    if user_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already exists!")

    if user.password != user.cpassword:
        raise HTTPException(status_code=400, detail="password doesn't match")

    hashed_pass = bcrypt.hashpw(user.password.encode("utf-8"), bcrypt.gensalt()).decode(
        "utf-8"
    )

    user_data = {
        "fname": user.fname,
        "lname": user.lname,
        "username": user.username,
        "email": user.email,
        "password": hashed_pass,
    }

    user_collection.insert_one(user_data)

    return {"msg": "User Added Successfully"}


@app.post("/users/login", status_code=200)
def login(user: UserLogin):
    if not all([user.email, user.password]):
        raise HTTPException(status_code=400, detail="All fields are required")

    existing = db["users"].find_one({"email": user.email})

    if not existing:
        raise HTTPException(status_code=404, detail="Email not found")

    if not bcrypt.checkpw(
        user.password.encode("utf-8"), existing["password"].encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"email": user.email})

    return {"message": "Login Successful", "token": token}


@app.get("/users/profile")
def profile(user=Depends(verify_token)):
    return {"user": user}


@app.post("/users/get_columns")
def get_columns(file : UploadFile = File(...),user=Depends(verify_token)):
    
    content=file.file.read()
    df = pd.read_csv(BytesIO(content))  
    return list(df.columns)



@app.post("/users/send_dataset")
def get_dataset(
    task_type: TaskType = Form(...),
    user=Depends(verify_token),
    file: UploadFile = File(...),
    target_col : str = Form(...),
    tuning : bool = Form(...)
):
    
    try : 
        print(task_type, file)
        if not file.filename:
            raise HTTPException(status_code=400, detail="file not found")

        user_folder = f"{USERS_FOLDER}/{user['username']}/datasets/{task_type.value}"
        os.makedirs(user_folder, exist_ok=True)

        unique_name = f"{uuid4().hex}_{file.filename}"
        dataset_id = Path(unique_name).stem
        target_path = f"{user_folder}/{unique_name}"

        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        dataset = DatasetUploadResponse(
            original_name=file.filename,
            stored_name=unique_name,
            user_id=user["username"],
            path=str(target_path),
            task_type=task_type.value,
        )
        
        
        train_dataset=TrainDataset(
            dataset=dataset,
            tuning=tuning,
            target_col=target_col
        )
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500,detail="Internal Server Error! Dont worry it is not your fault!")
    
    # Start training in background thread
    dataset_id = Path(unique_name).stem
    thread = threading.Thread(
        target=start_train_async,
        args=(train_dataset,),
        daemon=True
    )
    thread.start()
    logging.info(f"[UPLOAD] Started background training for dataset {dataset_id}")

    # Return immediately with dataset_id for polling
    return {
        "message": "Dataset uploaded successfully. Training started.",
        "dataset": {
            "original_name": file.filename,
            "stored_name": unique_name,
            "task_type": task_type.value,
            "target_col": target_col,
            "dataset_id": dataset_id,
        },
        "status": {
            "dataset_id": dataset_id,
            "poll_url": f"/users/training_status?dataset_id={dataset_id}",
        },
    }

def start_train_async(train_dataset: TrainDataset):
    """Run training in background thread with error handling"""
    dataset_id = Path(train_dataset.dataset.stored_name).stem
    try:
        logging.info(f"[TRAINING] Starting async training for dataset {dataset_id}")
        start_train(train_dataset)
        logging.info(f"[TRAINING] Completed training for dataset {dataset_id}")
    except Exception as e:
        logging.error(f"[TRAINING] Failed for dataset {dataset_id}: {str(e)}", exc_info=True)
        try:
            tracker = TrainingStatusTracker(train_dataset.dataset.user_id, dataset_id)
            tracker.error(f"Training failed: {str(e)}")
        except Exception as tracker_error:
            logging.error(f"[TRAINING] Failed to update status tracker: {tracker_error}")

def start_train(
    train_dataset: TrainDataset
):
    dataset_id = Path(train_dataset.dataset.stored_name).stem
    tracker = TrainingStatusTracker(train_dataset.dataset.user_id, dataset_id)
    tracker.update("queued", "Dataset received. Preparing preprocessing pipeline.")
    user = User(
        train_dataset.dataset.path,
        train_dataset.dataset.user_id,
        train_dataset.target_col,
        train_dataset.tuning,
        train_dataset.dataset.task_type,
        dataset_id,
        status_tracker=tracker,
    )
    return user.start()


@app.get("/users/get_models")
def get_models(user = Depends(verify_token)):
    user_base = Path(USERS_FOLDER) / user["username"] / "models"

    response = {
        "classification": [],
        "regression": []
    }

    for t in ["classification", "regression"]:
        folder = user_base / t

        if not folder.exists() or not folder.is_dir():
            continue

        for item in folder.iterdir():
            if not item.is_file() or item.suffix != ".pkl":
                continue

            rel_path = item.relative_to(Path(USERS_FOLDER))
            meta = {}
            meta_file = item.with_suffix(".meta.json")

            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                except Exception as meta_error:
                    print(f"[META] Failed to read {meta_file}: {meta_error}")

            response[t].append({
                "name": item.name,
                "path": str(rel_path),
                "download_url": f"/users/download_model?file_path={rel_path}",
                "metric_name": meta.get("metric_name"),
                "metric_value": meta.get("metric_value"),
                "explanations": meta.get("explanations", []),
                "generated_at": meta.get("generated_at"),
                "model_label": meta.get("model_name"),
                "model_reason": meta.get("model_reason"),
                "human_metric": meta.get("human_metric"),
            })

    return response


@app.get("/users/download_model")
def download_model(file_path: str = Query(...)):
    base_dir = Path(USERS_FOLDER).resolve()
    abs_path = (base_dir / file_path).resolve()

    # Security: prevent path traversal
    if base_dir not in abs_path.parents:
        raise HTTPException(status_code=400, detail="Invalid file path")

    if not abs_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=abs_path,
        filename=abs_path.name,
        media_type="application/octet-stream"
    )


@app.get("/users/get_bundles")
def get_bundles(user=Depends(verify_token)):
    templates_base = Path(USERS_FOLDER) / user["username"] / "templates"
    response = {"classification": [], "regression": []}

    for task in response.keys():
        folder = templates_base / task
        if not folder.exists():
            continue

        for zip_file in folder.glob("*.zip"):
            rel_path = zip_file.relative_to(Path(USERS_FOLDER))
            stats = zip_file.stat()
            response[task].append(
                {
                    "name": zip_file.stem,
                    "path": str(rel_path),
                    "size_bytes": stats.st_size,
                    "modified_ts": stats.st_mtime,
                    "download_url": f"/users/download_bundle?file_path={rel_path}",
                }
            )

    return response


@app.get("/users/download_bundle")
def download_bundle(file_path: str = Query(...), user=Depends(verify_token)):
    base_dir = (Path(USERS_FOLDER) / user["username"] / "templates").resolve()
    abs_path = (Path(USERS_FOLDER) / file_path).resolve()

    if base_dir not in abs_path.parents:
        raise HTTPException(status_code=403, detail="Not allowed")

    if not abs_path.exists() or abs_path.suffix.lower() != ".zip":
        raise HTTPException(status_code=404, detail="Bundle not found")

    return FileResponse(
        path=abs_path,
        filename=abs_path.name,
        media_type="application/zip"
    )


@app.get("/users/training_status")
def get_training_status(dataset_id: str = Query(...), user=Depends(verify_token)):
    status_path = Path(USERS_FOLDER) / user["username"] / "status" / f"{dataset_id}.json"
    if not status_path.exists():
        return {
            "dataset": dataset_id,
            "history": [],
            "current": None,
        }
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to read training status: {exc}")


@app.get("/users/active_training_runs")
def get_active_training_runs(user=Depends(verify_token)):
    """Get all active (non-terminal) training runs for the authenticated user"""
    status_dir = Path(USERS_FOLDER) / user["username"] / "status"
    
    if not status_dir.exists():
        return {"active_runs": []}
    
    active_runs = []
    
    for status_file in status_dir.glob("*.json"):
        try:
            status_data = json.loads(status_file.read_text(encoding="utf-8"))
            current = status_data.get("current")
            
            # Only include runs that are not in terminal state
            if current and current.get("state") not in ["completed", "error"]:
                dataset_id = status_file.stem
                active_runs.append({
                    "dataset_id": dataset_id,
                    "name": status_data.get("dataset_name", dataset_id),
                    "current_phase": current.get("phase"),
                    "current_state": current.get("state"),
                })
        except Exception as e:
            logging.warning(f"Failed to read status file {status_file}: {e}")
            continue
    
    return {"active_runs": active_runs}



@app.delete("/users/delete_model")
def delete_model(file_path: str = Query(...), user=Depends(verify_token)):
    user_base = (Path(USERS_FOLDER) / user["username"] / "models").resolve()
    abs_path = (Path(USERS_FOLDER) / file_path).resolve()

    if user_base not in abs_path.parents:
        raise HTTPException(status_code=403, detail="Not allowed")

    if not abs_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    abs_path.unlink()
    meta_file = abs_path.with_suffix(".meta.json")
    if meta_file.exists():
        meta_file.unlink()

    return {"msg": "Model deleted"}



@app.delete("/users/delete_bundle")
def delete_bundle(file_path: str = Query(...), user=Depends(verify_token)):
    """Delete a bundle file"""
    user_base = (Path(USERS_FOLDER) / user["username"] / "templates").resolve()
    abs_path = (Path(USERS_FOLDER) / file_path).resolve()

    if user_base not in abs_path.parents:
        raise HTTPException(403, "Not allowed")

    if not abs_path.exists():
        raise HTTPException(404, "Bundle not found")

    abs_path.unlink()
    logging.info(f"[DELETE] Bundle deleted: {file_path}")
    return {"msg": "Bundle deleted"}


@app.get("/users/get_datasets")
def get_all_datasets(user = Depends(verify_token)):
    base_path = f"{USERS_FOLDER}/{user['username']}/datasets"

    response = {"classification": [], "regression": []}

    for t in ["classification", "regression"]:
        folder = f"{base_path}/{t}"

        if not os.path.exists(folder):
            response[t] = []
            continue

        files = []
        for file in os.listdir(folder):
            full_path = os.path.join(folder, file)

            if not os.path.isfile(full_path):
                continue

            files.append({
                "name": file.split("_",1)[1],
                "download_url": f"/users/download_dataset?file_path={full_path}",
                "path": full_path
            })

        response[t] = files

    return response

@app.get("/users/download_dataset")
def download_dataset(
    file_path: str,
    user = Depends(verify_token)
):
    # Only allow files inside user's directory
    user_base = f"{USERS_FOLDER}/{user['username']}/datasets"

    if not file_path.startswith(user_base):
        raise HTTPException(403, "Not allowed")

    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found")

    return FileResponse(
        path=file_path,
        filename=os.path.basename(file_path).split("_",1)[1],
        media_type="application/octet-stream"
    )


@app.delete("/users/delete_dataset")
def delete_dataset_entry(file_path: str = Query(...), user=Depends(verify_token)):
    user_base = Path(USERS_FOLDER) / user["username"] / "datasets"
    abs_path = Path(file_path).resolve()

    if user_base.resolve() not in abs_path.parents:
        raise HTTPException(403, "Not allowed")

    if not abs_path.exists():
        raise HTTPException(404, "File not found")

    abs_path.unlink()

    return {"msg": "Dataset deleted"}
