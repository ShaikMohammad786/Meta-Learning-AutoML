import os
import shutil
import bcrypt
from fastapi import FastAPI, HTTPException,Depends,Request,UploadFile, File
from pymongo import MongoClient
from dotenv import load_dotenv
from constants import *
from pydantic_models.user_model import UserRegister,UserLogin
from pydantic_models.dataset_upload import DatasetUploadResponse
from utils.jwt_handler import create_access_token,verify_token
from uuid import uuid4

load_dotenv()

client = MongoClient(os.getenv("MONGO_DB_URI"))
db = client["MetaML"]
user_collection = db["users"]



app = FastAPI()



@app.get("/")
def root():
    return {"msg": "Welcome to Metaml"}

@app.post("/register")
def register(user : UserRegister):

    print("PASSWORD LENGTH:", len(user.password))
    
    if not all([user.fname , user.lname , user.username , user.email , user.password , user.cpassword]):
        raise HTTPException(status_code=400 , detail="all fields are required")

    if(user_collection.find_one({"username":user.username})):
        raise HTTPException(status_code=400 , detail = "username already taken")
    
    if(user_collection.find_one({"email":user.email})):
        raise HTTPException(status_code=400 , detail = "Email already exists!")
    
    if(user.password != user.cpassword):
        raise HTTPException(status_code=400 ,detail="password doesn't match")


    hashed_pass = bcrypt.hashpw(user.password.encode('utf-8'),bcrypt.gensalt()).decode('utf-8')
    
    user_data = {
        "fname":user.fname,
        "lname":user.lname,
        "username":user.username,
        "email":user.email,
        "password":hashed_pass
    }

    user_collection.insert_one(user_data)

    return {
        "msg" : "User Added Successfully"
    }


@app.post("/login",status_code=200)
def login(user:UserLogin):
    if not all([user.email, user.password]):
        raise HTTPException(status_code=400, detail='All fields are required')

    existing = db["users"].find_one({"email":user.email})
    
    if not existing:
        raise HTTPException(status_code=404, detail="Email not found")
    
    if not bcrypt.checkpw(user.password.encode('utf-8'), existing["password"].encode('utf-8')):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"email":user.email})

    return {"message": "Login Successful" , "token" : token}


@app.get("/profile")
def profile(user=Depends(verify_token)):
    return {"user":user}


@app.post("/users/send_dataset")
def get_dataset(task_type,user = Depends(verify_token),file : UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail ="file not found")
    

    user_folder = f"{USERS_FOLDER}/datasets/{task_type}/{user['username']}"
    os.makedirs(user_folder,exist_ok = True)

    unique_name = f"{uuid4().hex}_{file.filename}"
    target_path = f"{user_folder}/{unique_name}"

    with open(target_path,"wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return DatasetUploadResponse(
        original_name = file.filename,
        stored_name = unique_name,
        user_id = user['username'],
        path = str(target_path)
    )