from fastapi import FastAPI
from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv()



client = MongoClient(os.getenv("MONGO_DB_URI"))
db = client["MetaML"]

def get_user_collection():
    return db["users"]

