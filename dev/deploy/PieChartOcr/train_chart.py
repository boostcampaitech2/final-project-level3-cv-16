#!/usr/bin/env python
import argparse
import os
import os.path as osp
import importlib
import json
import numpy as np
import re
import traceback

from azureml.core.run import Run
import torch
from tqdm import tqdm
torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True

from config import system_configs
from nnet.py_factory import NetworkFactory
from db.datasets import datasets


def train(training_dbs, validation_db, start_iter=0):
    learning_rate    = system_configs.learning_rate
    max_iteration    = system_configs.max_iter
    pretrained_model = system_configs.pretrain
    snapshot         = system_configs.snapshot
    val_iter         = system_configs.val_iter
    display          = system_configs.display
    decay_rate       = system_configs.decay_rate
    stepsize         = system_configs.stepsize
    val_ind = 0
    train_ind = 0
    min_loss = -1
    print("building model...")
    nnet = NetworkFactory()

    # load data sampling function
    data_file = "sample.{}".format(training_dbs.data)
    sample_data = importlib.import_module(data_file).sample_data

    run = Run.get_context()
    if pretrained_model is not None:
        if not os.path.exists(pretrained_model):
            raise ValueError("pretrained model does not exist")
        print("loading from pretrained model")
        nnet.load_pretrained_params(pretrained_model)

    if start_iter:
        if start_iter == -1:
            print("training starts from the latest iteration")
            save_list = os.listdir(system_configs.snapshot_dir)
            save_list.sort(reverse=True)
            if len(save_list) > 0:
                target_save = save_list[0]
                start_iter = int(re.findall(r'\d+', target_save)[0])
                learning_rate /= (decay_rate ** (start_iter // stepsize))
                nnet.load_params(start_iter)
            else:
                start_iter = 0
        nnet.set_lr(learning_rate)
        print("training starts from iteration {} with learning_rate {}".format(start_iter + 1, learning_rate))
    else:
        nnet.set_lr(learning_rate)

    print("training start...")
    nnet.cuda()
    nnet.train_mode()
    if not os.path.exists('./outputs'):
        os.makedirs('./outputs')
        print('outputs file created')
    else:
        print(os.listdir('./outputs'))
    for iteration in tqdm(range(start_iter + 1, max_iteration + 1)):
        print("start")
        training, train_ind = sample_data(training_dbs, train_ind, debug=False)
        training_loss = nnet.train(**training)

        if display and iteration % display == 0:
            print("training loss at iteration {}: {}".format(iteration, training_loss.item()))
            run.log('train_loss', training_loss.item())

        if val_iter and validation_db.db_inds.size and iteration % val_iter == 0:
            nnet.eval_mode()
            validation, val_ind = sample_data(validation_db, val_ind, debug=False)
            validation_loss = nnet.validate(**validation)
            print("validation loss at iteration {}: {}".format(iteration, validation_loss.item()))
            run.log('val_loss', validation_loss.item())
            if min_loss < 0:
                min_loss = validation_loss.item()
            nnet.train_mode()

        if iteration % snapshot == 0 and validation_loss.item() < min_loss:
            nnet.save_params(iteration)
            min_loss = validation_loss.item()

        if iteration % stepsize == 0:
            learning_rate /= decay_rate
            nnet.set_lr(learning_rate)

def parse_args():
    parser = argparse.ArgumentParser(description="Train CornerNet")
    parser.add_argument("--data_dir", dest="data_dir", default="/opt/ml/PieChartOcr/data/piedata", type=str)
    parser.add_argument("--iter", dest="start_iter", help="train at iteration i", default=0, type=int)
    parser.add_argument('--cache_path', dest="cache_path", default="/opt/ml/PieChartOcr/data/piedata/cache", type=str)
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = parse_args()
    cfg_name = "CornerNetPurePie"
    cfg_file = os.path.join(osp.dirname(osp.abspath(__file__)), "config", cfg_name + ".json")
    with open(cfg_file, "r") as f:
        configs = json.load(f)
    configs["system"]["snapshot_name"] = cfg_name
    configs["system"]["data_dir"] = args.data_dir
    configs["system"]["cache_dir"] = args.cache_path
    system_configs.update_config(configs["system"])

    train_split = system_configs.train_split
    val_split   = system_configs.val_split

    print("loading all datasets...")
    dataset = system_configs.dataset
    training_dbs  = datasets[dataset](configs["db"], train_split)
    validation_db = datasets[dataset](configs["db"], val_split)

    train(training_dbs, validation_db, args.start_iter)
