import requests
from typing import Optional

from fastapi import FastAPI

from router.image_receive import router

app = FastAPI() #REST API

app.include_router(router)

@app.get("/")
def read_root():
    
    return {"Hello" : "this is main page"}
