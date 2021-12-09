import torch
import mlflow
from mlflow.models import Model
import mlflow.pyfunc
import base64
from PIL import Image
import numpy as np

from dl_model.test_pipe_fastapi import dl_model


local_image = "/opt/ml/final/final-project-level3-cv-16/dev/workspace/Sanghyun/fastapi/dl_model/data/piedata(1008)/pie/images/test2019/f4a0f74507725f767cfd7769e6df3e53_d3d3LnNhZ2lzLm9yZy56YQkxOTcuMTU3LjI0Mi4xNTA=.xls-2-0.png"

img = Image.open(local_image).convert("RGB")
input_example = np.array(img)

result, groups = dl_model(input_example)

mlflow.pyfunc.add_to_model(
    model=dl_model,
    loader_module=__name__,
    
)
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model("model")
