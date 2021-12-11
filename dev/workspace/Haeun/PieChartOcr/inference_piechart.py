#!/usr/bin/env python
import argparse
import cv2
import importlib
import json
import os
import os.path as osp

import matplotlib
matplotlib.use("Agg")
import torch
torch.backends.cudnn.benchmark = False
from tqdm import tqdm
from PIL import Image

from config import system_configs
from db.datasets import datasets
from nnet.py_factory import NetworkFactory
from RuleGroup.Pie import GroupPie


def load_net(testiter, cfg_name, data_dir, cache_dir, cuda_id=0):
    cfg_file = os.path.join(osp.dirname(osp.abspath(__file__)), "config", cfg_name + ".json")
    with open(cfg_file, "r") as f:
        configs = json.load(f)
    configs["system"]["snapshot_name"] = cfg_name
    configs["system"]["data_dir"] = data_dir
    configs["system"]["cache_dir"] = cache_dir
    system_configs.update_config(configs["system"])

    train_split = system_configs.train_split
    val_split = system_configs.val_split
    test_split = system_configs.test_split

    split = {
        "training": train_split,
        "validation": val_split,
        "testing": test_split
    }["testing"]

    test_iter = system_configs.max_iter if testiter is None else testiter
    print("loading parameters at iteration: {}".format(test_iter))
    dataset = system_configs.dataset
    db = datasets[dataset](configs["db"], split)
    print("building neural network...")
    nnet = NetworkFactory()
    print("loading parameters...")
    nnet.load_params(test_iter)
    if torch.cuda.is_available():
        nnet.cuda(cuda_id)
    nnet.eval_mode()
    return db, nnet

def test(db_pie, nnet_pie, image_path):
    image = Image.fromarray(cv2.imread(image_path))
    path = 'testfile.test_%s' % "CornerNetPurePie"
    testing_pie = importlib.import_module(path).testing
    with torch.no_grad():
        results = testing_pie(image, db_pie, nnet_pie)
        cens = results[0]
        keys = results[1]
        theta_data, group_data = GroupPie(image, cens, keys)
        return theta_data, group_data

def parse_args():
    parser = argparse.ArgumentParser(description="Test DeepRule")
    parser.add_argument("--image_path", dest="image_path", help="test images", default="/opt/ml/PieChartOcr/data/piedata/pie/images/test2019", type=str)
    parser.add_argument("--save_path", dest="save_path", help="where to save results", default="./test_data_save.json", type=str)
    parser.add_argument("--data_dir", dest="data_dir", default="/opt/ml/PieChartOcr/data/piedata/", type=str)
    parser.add_argument('--cache_path', dest="cache_path", type=str, default="/opt/ml/PieChartOcr/data/piedata/cache/")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    db_pie, nnet_pie = load_net(50000, "CornerNetPurePie", args.data_dir,  args.cache_path, 0)
    tar_path = args.image_path
    save_path = args.save_path
    rs_dict = {}
    images = os.listdir(tar_path)
    for image in tqdm(images):
        path = os.path.join(tar_path, image)
        deg_li, points_li = test(db_pie, nnet_pie, path)
        rs_dict[image] = [deg_li, points_li]

    with open(save_path, "w") as f:
        json.dump(rs_dict, f)
