#!/usr/bin/env python
import os
import json
import torch
import pprint
import argparse

import matplotlib
matplotlib.use("Agg")
import cv2
from tqdm import tqdm
from config import system_configs
from nnet.py_factory import NetworkFactory
from db.datasets import datasets
import importlib
import json
from RuleGroup.Cls import GroupCls
from RuleGroup.Bar import GroupBarRaw
from RuleGroup.LineQuiry import GroupQuiryRaw
from RuleGroup.LIneMatch import GroupLineRaw
from RuleGroup.Pie import GroupPie
import math
from PIL import Image, ImageDraw, ImageFont
torch.backends.cudnn.benchmark = False
import requests
import time
import re
def make_dirs(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
def load_net(testiter, cfg_name, data_dir, cache_dir, cuda_id=0):
    cfg_file = os.path.join(system_configs.config_dir, cfg_name + ".json")
    with open(cfg_file, "r") as f:
        configs = json.load(f)
    configs["system"]["snapshot_name"] = cfg_name
    configs["system"]["data_dir"] = data_dir
    configs["system"]["cache_dir"] = cache_dir
    configs["system"]["result_dir"] = 'result_dir'
    configs["system"]["tar_data_dir"] = "Cls"
    system_configs.update_config(configs["system"])

    train_split = system_configs.train_split
    val_split = system_configs.val_split
    test_split = system_configs.test_split

    split = {
        "training": train_split,
        "validation": val_split,
        "testing": test_split
    }["validation"]


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
    if type == "Bar":
        db_bar, nnet_bar = load_net(50000, "CornerNetPureBar", data_dir, cache_dir,
                                    id_cuda)
        path = 'testfile.test_%s' % "CornerNetPureBar"
        testing_bar = importlib.import_module(path).testing
        methods['Bar'] = [db_bar, nnet_bar, testing_bar]
    if type == "Pie":
        db_pie, nnet_pie = load_net(50000, "CornerNetPurePie", data_dir, cache_dir,
                                    id_cuda)
        path = 'testfile.test_%s' % "CornerNetPurePie"
        testing_pie = importlib.import_module(path).testing
        methods['Pie'] = [db_pie, nnet_pie, testing_pie]
    if type == "Line":
        db_line, nnet_line = load_net(50000, "CornerNetLine", data_dir, cache_dir,
                                      id_cuda)
        path = 'testfile.test_%s' % "CornerNetLine"
        testing_line = importlib.import_module(path).testing
        methods['Line'] = [db_line, nnet_line, testing_line]
        db_line_cls, nnet_line_cls = load_net(20000, "CornerNetLineClsReal",  data_dir, cache_dir,
                                               id_cuda)
        path = 'testfile.test_%s' % "CornerNetLineCls"
        testing_line_cls = importlib.import_module(path).testing
        methods['LineCls'] = [db_line_cls, nnet_line_cls, testing_line_cls]
    return methods

def ocr_result(image_path):
    subscription_key = "ad143190288d40b79483aa0d5c532724"
    vision_base_url = "https://westus2.api.cognitive.microsoft.com/vision/v2.0/"
    ocr_url = vision_base_url + "read/core/asyncBatchAnalyze"
    headers = {'Ocp-Apim-Subscription-Key': subscription_key, 'Content-Type': 'application/octet-stream'}
    params = {'language': 'unk', 'detectOrientation': 'true'}
    image_data = open(image_path, "rb").read()
    response = requests.post(ocr_url, headers=headers, params=params, data=image_data)
    response.raise_for_status()
    op_location = response.headers['Operation-Location']
    analysis = {}
    while "recognitionResults" not in analysis.keys():
        time.sleep(3)
        binary_content = requests.get(op_location, headers=headers, params=params).content
        analysis = json.loads(binary_content.decode('ascii'))
    line_infos = [region["lines"] for region in analysis["recognitionResults"]]
    word_infos = []
    for line in line_infos:
        for word_metadata in line:
            for word_info in word_metadata["words"]:
                word_infos.append(word_info)
    return word_infos

def check_intersection(box1, box2):
    if (box1[2] - box1[0]) + ((box2[2] - box2[0])) > max(box2[2], box1[2]) - min(box2[0], box1[0]) \
            and (box1[3] - box1[1]) + ((box2[3] - box2[1])) > max(box2[3], box1[3]) - min(box2[1], box1[1]):
        Xc1 = max(box1[0], box2[0])
        Yc1 = max(box1[1], box2[1])
        Xc2 = min(box1[2], box2[2])
        Yc2 = min(box1[3], box2[3])
        intersection_area = (Xc2-Xc1)*(Yc2-Yc1)
        return intersection_area/((box2[3]-box2[1])*(box2[2]-box2[0]))
    else:
        return 0

def try_math(image_path, cls_info):
    title_list = [1, 2, 3]
    title2string = {}
    max_value = 1
    min_value = 0
    max_y = 0
    min_y = 1
    word_infos = ocr_result(image_path)
    for id in title_list:
        if id in cls_info.keys():
            predicted_box = cls_info[id]
            words = []
            for word_info in word_infos:
                word_bbox = [word_info["boundingBox"][0], word_info["boundingBox"][1], word_info["boundingBox"][4], word_info["boundingBox"][5]]
                if check_intersection(predicted_box, word_bbox) > 0.5:
                    words.append([word_info["text"], word_bbox[0], word_bbox[1]])
            words.sort(key=lambda x: x[1]+10*x[2])
            word_string = ""
            for word in words:
                word_string = word_string + word[0] + ' '
            title2string[id] = word_string
    if 5 in cls_info.keys():
        plot_area = cls_info[5]
        y_max = plot_area[1]
        y_min = plot_area[3]
        x_board = plot_area[0]
        dis_max = 10000000000000000
        dis_min = 10000000000000000
        for word_info in word_infos:
            word_bbox = [word_info["boundingBox"][0], word_info["boundingBox"][1], word_info["boundingBox"][4], word_info["boundingBox"][5]]
            word_text = word_info["text"]
            word_text = re.sub('[^-+0123456789.]', '',  word_text)
            word_text_num = re.sub('[^0123456789]', '', word_text)
            word_text_pure = re.sub('[^0123456789.]', '', word_text)
            if len(word_text_num) > 0 and word_bbox[2] <= x_board+4:
                dis2max = math.sqrt(math.pow((word_bbox[0]+word_bbox[2])/2-x_board, 2)+math.pow((word_bbox[1]+word_bbox[3])/2-y_max, 2))
                dis2min = math.sqrt(math.pow((word_bbox[0] + word_bbox[2]) / 2 - x_board, 2) + math.pow(
                    (word_bbox[1] + word_bbox[3]) / 2 - y_min, 2))
                y_mid = (word_bbox[1]+word_bbox[3])/2
                if dis2max <= dis_max:
                    dis_max = dis2max
                    max_y = y_mid
                    max_value = float(word_text_pure)
                    if word_text[0] == '-':
                        max_value = -max_value
                if dis2min <= dis_min:
                    dis_min = dis2min
                    min_y = y_mid
                    min_value = float(word_text_pure)
                    if word_text[0] == '-':
                        min_value = -min_value
        delta_min_max = max_value-min_value
        delta_mark = min_y - max_y
        delta_plot_y = y_min - y_max
        delta = delta_min_max/delta_mark
        if abs(min_y-y_min)/delta_plot_y > 0.1:
            print(abs(min_y-y_min)/delta_plot_y)
            print("Predict the lower bar")
            min_value = int(min_value + (min_y-y_min)*delta)

    return title2string, round(min_value, 2), round(max_value, 2)

class TEST:
    def __init__(self, methods):
        self.methods = methods

    def test(self, image_path):
        methods = self.methods
        image = Image.fromarray(cv2.imread(image_path))
        with torch.no_grad():
            print("Predicted as PieChart")
            results = methods['Pie'][2](image, methods['Pie'][0], methods['Pie'][1], debug=False)
            cens = results[0]
            keys = results[1]
            pie_data = GroupPie(image, cens, keys)
        return pie_data

#methods = Pre_load_nets(args.type, 0, args.data_dir, args.cache_path)
methods = Pre_load_nets("Pie", 0, "./data/piedata(1008)", "cache_path")
test = TEST(methods)
