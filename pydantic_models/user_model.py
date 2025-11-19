from pydantic import BaseModel,EmailStr


class UserRegister(BaseModel):
    fname:str
    lname :str
    username:str
    email:EmailStr
    password:str
    cpassword:str
    
class UserLogin(BaseModel):
    email:EmailStr
    password:str