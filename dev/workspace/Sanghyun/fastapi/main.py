from typing import List

import numpy as np

import json
import requests
from pydantic import BaseModel
from fastapi import FastAPI

from utils.visualization import plot_model_results

app = FastAPI() #REST API

class Item(BaseModel):
    # req = {
    #     "image_as_list" : List # [H, W, C]
    # }
    image_as_list: List

# MODEL_URL = 'http://49.50.175.108:6010/items/'
MODEL_URL = 'http://model_server:8000/items/'

@app.get("/")
def main_page():
    return {"main":"page"}

@app.post("/backend/")
async def create_item(
    item: Item
):
    print("in /backend/")
    image_list = item.image_as_list
    image_arr = np.array(image_list, dtype=np.uint8)
    # send to model server
    req_dict = {
        "instances" : image_list # [H, W, C]
    }
    response = requests.post(
        url=MODEL_URL,
        data=json.dumps(req_dict)
    )
    # parse a response from model server
    parsed_response = eval(response.text)

    visualised_image = plot_model_results(image_arr, parsed_response)
    
    parsed_response.update({"im_plot" : visualised_image.tolist()})

    return parsed_response
