from typing import ByteString

import numpy as np

import json
import requests
from pydantic import BaseModel
from fastapi import FastAPI, Request

from utils.visualization import plot_model_results

app = FastAPI() #REST API

class Item(BaseModel):
    # req = {
    #     "image_as_list" : List # [H, W, C]
    # }
    _arr_bytes: bytes

# MODEL_URL = 'http://49.50.175.108:6010/items/'
# MODEL_URL = 'http://model_server:8000/items/'

@app.get("/")
def main_page():
    return {"main":"page"}

@app.post("/backend/")
async def create_item(
    item: Request
):
    print("in /backend/")
    arr_bytes: bytes = await item.body()
    arr_bytes = np.fromstring(arr_bytes, dtype=np.uint8)
    
    return {"bytes" : arr_bytes.tostring()}
