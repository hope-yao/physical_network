from __future__ import print_function

import os
import json
import logging
from datetime import datetime


def prepare_dirs_and_logger(config):
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    if config.load_path:
        if config.load_path.startswith(config.log_dir):
            config.model_dir = config.load_path
        else:
            if config.load_path.startswith(config.dataset):
                config.model_name = config.load_path
            else:
                config.model_name = "{}_{}".format(config.dataset, config.load_path)
    else:
        config.model_name = "{}_{}".format(config.dataset, get_time())

    if not hasattr(config, 'model_dir'):
        config.model_dir = os.path.join(config.log_dir, config.model_name)
    config.data_path = os.path.join(config.data_dir, config.dataset)

    for path in [config.log_dir, config.data_dir, config.model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")

def save_config(config):
    param_path = os.path.join(config.model_dir, "params.json")

    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def rank(array):
    return len(array.shape)

import tensorflow as tf
# def list2tensor(alist,dim=0):
#     for i, list_i in enumerate(alist):
#         if i == 0:
#             atensor = list_i
#         else:
#             atensor = tf.concat([atensor, list_i], dim)
#     return atensor


import dateutil.tz

def creat_dir(network_type):
    """code from on InfoGAN"""
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    root_log_dir = "./saved_logs/" + network_type
    exp_name = network_type + "_%s" % timestamp
    log_dir = os.path.join(root_log_dir, exp_name)

    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    root_model_dir = "./saved_models/" + network_type
    exp_name = network_type + "_%s" % timestamp
    model_dir = os.path.join(root_model_dir, exp_name)

    for path in [log_dir, model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)
    return log_dir, model_dir


