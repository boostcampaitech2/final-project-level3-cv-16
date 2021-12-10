#!/usr/bin/env python
import cv2
import importlib
import json
import math
import os
import os.path as osp

from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
torch.backends.cudnn.benchmark = False

from config import system_configs
from nnet.py_factory import NetworkFactory
from db.datasets import datasets
from RuleGroup.Pie import GroupPie


def load_net(testiter, cfg_name, data_dir, cache_dir, cuda_id=0):
    cfg_file = os.path.join(osp.dirname(osp.abspath(__file__)), "config", cfg_name + ".json")
    with open(cfg_file, "r") as f:
        configs = json.load(f)
    configs["system"]["snapshot_name"] = cfg_name
    configs["system"]["data_dir"] = data_dir
    configs["system"]["cache_dir"] = cache_dir
    configs["system"]["result_dir"] = 'result_dir'
    configs["system"]["tar_data_dir"] = "Cls"
    system_configs.update_config(configs["system"])

    split = system_configs.test_split
    
    test_iter = system_configs.max_iter if testiter is None else testiter
    print("loading parameters at iteration: {}".format(test_iter))
    dataset = system_configs.dataset
    db = datasets[dataset](configs["db"], split)
    print("building neural network...")
    nnet = NetworkFactory(db)
    print("loading parameters...")
    nnet.load_params(test_iter)
    if torch.cuda.is_available():
        nnet.cuda(cuda_id)
    nnet.eval_mode()
    return db, nnet

def Pre_load_nets(type, id_cuda, data_dir, cache_dir):
    methods = {}
    if type == "Pie":
        db_pie, nnet_pie = load_net(50000, "CornerNetPurePie", data_dir, cache_dir,
                                    id_cuda)
        path = 'testfile.test_%s' % "CornerNetPurePie"
        testing_pie = importlib.import_module(path).testing
        methods['Pie'] = [db_pie, nnet_pie, testing_pie]
    return methods

def get_distance(x1, y1, x2, y2):
    return ((((x2 - x1 )**2) + ((y2-y1)**2) )**0.5)

def get_arc_center(points):
    x_center, y_center, x_left, y_left, x_right, y_right = points
    assert abs(get_distance(x_left,y_left, x_center, y_center) - get_distance(x_right, y_right, x_center, y_center)) < 5

    radius = get_distance(x_left,y_left, x_center, y_center)
    norm_x_left, norm_x_right = (x_left - x_center)/radius, (x_right - x_center)/radius

    # y가 역방향임
    norm_y_left, norm_y_right = (y_center - y_left)/radius, (y_center - y_right)/radius

    angle = math.atan2((norm_y_left + norm_y_right)/2, (norm_x_left + norm_x_right)/2)
    coord = (math.cos(angle)*radius + x_center, -math.sin(angle)*radius + y_center)

    return list(map(int, coord))

class TEST:
    def __init__(self, methods):
        self.methods = methods
    def test(self, image_path):
        methods = self.methods
        image = Image.fromarray(cv2.imread(image_path))
        with torch.no_grad():
            results = methods['Pie'][2](image, methods['Pie'][0], methods['Pie'][1], debug=False)
            cens = results[0]
            keys = results[1]
            theta_data, group_data = GroupPie(image, cens, keys)
            return theta_data, group_data

data_path = os.path.join(osp.dirname(osp.abspath(__file__)),"data/piedata")
methods = Pre_load_nets("Pie", 0, data_path, "/opt/ml/DeepRule/data/piedata/cache/")
test = TEST(methods)
# path = "/opt/ml/DeepRule/data/piedata/pie/images/val2019/fa2ea7d00fb6500b77b9a7b9e349d3b2_d3d3LmNhbm9la2F5YWstbGVtYW5zLm5ldAkyMTcuMTYuMTAuMw==.xls-1-0.png"
path = "/opt/ml/DeepRule/data/piedata/pie/images/val2019/fa282ce5ed55721823a18eb8c99432de_d3d3LmluZWdpLmdvYi5teAkyMDAuMjMuOC41.xls-4-0.png"
deg_li, points_li = test.test(path)
img = Image.open(path)
for deg, point in zip(deg_li, points_li):
    center_x = point[0][0]
    center_y = point[0][1]
    left_x = point[2][0]
    left_y = point[2][1]
    right_x = point[1][0]
    right_y = point[1][1]
    arc_center = get_arc_center([center_x, center_y, left_x, left_y, right_x, right_y])
    x = (arc_center[0] + center_x)/2
    y = (arc_center[1] + center_y)/2
    if deg < 180:
        plt.text(
            x, 
            y, 
            "%d%%" % math.ceil(float(deg)/360*100),
            fontsize=5,
            bbox = {
                'boxstyle': 'round',
                'ec': (0.8, 0.8, 0.8),
                'fc': (1.0, 1.0, 1.0)
            }
        )
    else:
        x = 2*center_x - x
        y = 2*center_y - y
        plt.text(
            x, 
            y, 
            "%d%%" % math.ceil(float(deg)/360*100),
            fontsize=5,
            bbox = {
                'boxstyle': 'round',
                'ec': (0.8, 0.8, 0.8),
                'fc': (1.0, 1.0, 1.0)
            }
        )
        
plt.imshow(img)
plt.axis("off")
plt.savefig('savefig.png')
print("done")