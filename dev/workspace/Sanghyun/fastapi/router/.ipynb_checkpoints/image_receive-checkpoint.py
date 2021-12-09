import sys
import shutil
from os import path as osp

from PIL import Image
import numpy as np

from fastapi import APIRouter, File, UploadFile
from starlette.responses import FileResponse
from dl_model import test_pipe_fastapi as model

router = APIRouter(
    prefix="/upload",
    tags=["upload"]
)

BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
IMG_DIR = osp.join(BASE_DIR, "asset/")
SERVER_IMG_DIR = osp.join("http://localhost:8000/", "asset/")

@router.post("/")
async def image_file(
    image : UploadFile = File(...)
):
    if image is not None:
        local_image = osp.join(IMG_DIR, image.filename)
        with open(local_image, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        result = model.test(local_image)
    
    return {
        "filename": image.filename,
        "result" : result
    }



@router.get("/asset/{filename}")
async def image_show(filename: str):
    return FileResponse(f"{IMG_DIR}/{filename}")