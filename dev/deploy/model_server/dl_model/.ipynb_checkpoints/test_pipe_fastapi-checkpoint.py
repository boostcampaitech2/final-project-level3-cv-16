#!/usr/bin/env python
import os
import json
from io import BytesIO
import importlib
import os.path as osp
from PIL import Image
import numpy as np
from abc import ABCMeta, abstractmethod

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

#methods = Pre_load_nets(args.type, 0, args.data_dir, args.cache_path)
data_path = os.path.join(osp.dirname(osp.abspath(__file__)),"data/piedata(1008)")
db_pie, nnet_pie = load_net(50000, "CornerNetPurePie", data_path,  "cache_path", 0)
path = "dl_model.testfile.test_CornerNetPurePie"
testing_pie = importlib.import_module(path).testing

class TEST_MODEL:
    def __init__(self, db_pie, nnet_pie, debug=False):
        self.db_pie = db_pie
        self.nnet_pie = nnet_pie
        self.debug = debug
    
    @abstractmethod
    def __call__(self, x):
        
        results = testing_pie(x, db_pie, nnet_pie, debug=False)
        cens = results[0]
        keys = results[1]
        # print(results)
        pie_data, groups = GroupPie(x, cens, keys)
        
        return pie_data, groups
dl_model = TEST_MODEL(db_pie, nnet_pie, debug=False)
print("model loaded")
