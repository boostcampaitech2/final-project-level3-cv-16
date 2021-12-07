#!/usr/bin/env python
import os
import json
import importlib
import os.path as osp
from PIL import Image

import torch
torch.backends.cudnn.benchmark = False

import cv2
from dl_model.config import system_configs
from dl_model.nnet.py_factory import NetworkFactory
from dl_model.db.datasets import datasets
from dl_model.RuleGroup.Pie import GroupPie


#######################
# classes & functions #
#######################

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

    split = system_configs.val_split
    
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

class TEST:
    def __init__(self, methods):
        self.methods = methods
    def test(self, image_path, data_type):
        methods = self.methods
        image = Image.fromarray(cv2.imread(image_path))
        with torch.no_grad():
            if data_type == 'Pie':
                print("Predicted as PieChart")
                results = methods['Pie'][2](image, methods['Pie'][0], methods['Pie'][1], debug=False)
                cens = results[0]
                keys = results[1]
                pie_data = GroupPie(image, cens, keys)
                return pie_data

#methods = Pre_load_nets(args.type, 0, args.data_dir, args.cache_path)
data_path = os.path.join(osp.dirname(osp.abspath(__file__)),"data/piedata(1008)")
methods = Pre_load_nets("Pie", 0, data_path, "cache_path")
test = TEST(methods)

print("model loaded")