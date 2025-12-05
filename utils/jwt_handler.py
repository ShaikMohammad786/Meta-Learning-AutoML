from datetime import timedelta,datetime,timezone
from jose import jwt
import os
import sys
from utils.db_connection import get_user_collection
from fastapi import Request,HTTPException
sys.path.append(os.path.abspath(os.getcwd()))
from config import JWT_ALGORITHM,JWT_SECRET,ACCESS_TOKEN_EXPIRE_MINUTES
def create_access_token(data:dict,expires_delta:timedelta=None):
    to_encode=data.copy()
    
    if expires_delta:
        expire=datetime.now(timezone.utc) + expires_delta
    else:
        expire=datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp":expire})
    
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)



def verify_token(req:Request):
    auth=req.headers.get('Authorization')
    if not auth:
        raise HTTPException(status_code=401,detail='Missing bearer token')
    try:
        scheme,token=auth.split()
        if scheme.lower()!='bearer':
            raise HTTPException(status_code=401, detail="Invalid auth scheme")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=401, detail="Invalid Authorization header")
    
    try:
        payload=jwt.decode(token,JWT_SECRET,algorithms=[JWT_ALGORITHM])
        email=payload['email']
        user_collection=get_user_collection()
        user=user_collection.find_one({"email":email},{"password":0})
        user['_id']=str(user["_id"])
        if not user:
            raise HTTPException(status_code=404, detail="User Not Found")
        return user
    except Exception as e:
        print(e)
        raise HTTPException(status_code=401, detail="Invalid or expired token")
            
    
        