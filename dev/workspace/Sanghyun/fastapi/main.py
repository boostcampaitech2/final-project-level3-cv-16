from typing import List

import numpy as np
# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt

import json
import requests
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI() #REST API

class Item(BaseModel):
    # req = {
    #     "image_as_list" : List # [H, W, C]
    # }
    image_as_list: list

MODEL_URL = 'http://49.50.175.108:6010/items/'

@app.get("/")
def main_page():
    return {"main":"page"}

@app.post("/items/")
async def create_item(
    item: Item
):
    # image = np.array(item.image_as_list)
    # print(image.shape)
    return 0 #{"hey": "wassap"}
    image = item.image_as_list

    # send to model server
    req_dict = {
        "instances" : image # [H, W, C]
    }
    response = requests.post(
        url=MODEL_URL,
        data=json.dumps(req_dict)
    )
    
    # parse a response
    resp_eval = eval(response.text)
    
    im_shape = resp_eval["im_shape"]
    # degree_list = resp_eval["drg"]
    group_list = resp_eval["grp"]
    
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.imshow(np.array(image))
    for group in group_list:
        ax.plot(group[0], "*r", markersize = 15)
        ax.plot(group[1], ".b", markersize = 20)
        ax.plot(group[2], ".b", markersize = 20)
    
    fig_string = fig.canvas.to_string_rgb()
    arr_image = np.fromstring(fig_string, dtype=np.int8, sep='')
    arr_image = np.reshape(arr_image, im_shape)
    
    return {"im_plot" : arr_image.tolist()}

