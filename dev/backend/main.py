import json
import requests
from typing import List

import numpy as np

import easyocr
from fastapi import FastAPI
from pydantic import BaseModel

from utils.ocr_utils import ocr_predict
from utils.visualization import plot_model_results
from utils.keypoints import get_flattened_keypoints
app = FastAPI()

ocr_reader = []
@app.on_event("startup")
def set_ocr_reader():
    reader = easyocr.Reader(['en', 'fr', 'is'], recognizer='Transformer', gpu=False)
    ocr_reader.append(reader)



@app.get("/")
def main_page():
    return {"main":"page"}



# MODEL_URL = 'http://49.50.175.108:6010/items/'
MODEL_URL = 'http://sanghyun.ddns.net:8000/items/'

class Item(BaseModel):
    image_as_list: List # [H, W, C] : List[List[int, int, int]]

@app.post("/backend/")
async def create_item(
    item: Item
):
    response = dict()

    # get image
    image_list = item.image_as_list
    image_arr = np.array(image_list, dtype=np.uint8)
    
    # send to model server
    req_dict = {
        "instances" : image_list # [H, W, C]
    }
    model_response = requests.post(
        url=MODEL_URL,
        data=json.dumps(req_dict)
    )
    # parse a response from model server
    model_result_dict = eval(model_response.text)
    response.update(model_result_dict)

    im_shape = model_result_dict["im_shape"]
    degree_list = model_result_dict["dgr"]
    group_list = model_result_dict["grp"]

    flattened_keypoints = get_flattened_keypoints(group_list)
    
    reader = ocr_reader[0]
    ocr_result = ocr_predict(
        reader,
        image_arr, 
        degree_list, 
        flattened_keypoints, 
        debug=False
    )
    response.update({"ocr_result" : ocr_result})

    visualised_image = plot_model_results(
        image_arr, 
        im_shape, 
        degree_list, 
        flattened_keypoints
    )
    response.update({"im_plot" : visualised_image.tolist()})

    return response


