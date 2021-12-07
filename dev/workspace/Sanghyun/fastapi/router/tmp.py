from dl_model import test_pipe_fastapi as model

local_image = "/home/davi/workspace/final-project-level3-cv-16/dev/workspace/Sanghyun/fastapi/dl_model/data/piedata(1008)/pie/images/test2019/f4a0f74507725f767cfd7769e6df3e53_d3d3LnNhZ2lzLm9yZy56YQkxOTcuMTU3LjI0Mi4xNTA=.xls-2-0.png"
result = model.test(local_image)
