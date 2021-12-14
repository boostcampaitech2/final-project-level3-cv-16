from typing import Optional, List

from fastapi import FastAPI
from starlette.responses import JSONResponse

from pydantic import BaseModel
import numpy as np

from dl_model.test_pipe_fastapi import dl_model

app = FastAPI() #REST API

@app.get("/")
def read_root():
    return {"Hello" : "post your image"}



class Item(BaseModel):

    # req = {
    #     "instances" : List # [H, W, C]
    # }
    
    instances: List
    
@app.post("/items/")
async def create_item(
    item: Item
):
    
    image = np.array(item.instances)
    dgr, grp = dl_model(image)
    
    return {
        "im_shape": image.shape,
        "dgr" : dgr,
        "grp" : grp
    }