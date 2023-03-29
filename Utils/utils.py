import os
import sys
import torch
import pickle
import random
import pandas as pd
import torch.nn as nn
from mmcv import Config
import numpy as np
from Utils.transformser import ClassLabel,target_Compose,VideoID

from datasets.c3d_dataset.dataset_c3d import get_test_set
from mmaction.apis import init_recognizer

target_transform = target_Compose([VideoID(), ClassLabel()])

# 获取每个视频的总帧数
def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))
    return value


# 将图片转换到分类器可以使用的向量
def image_to_vector(model_name, dataset_name, x):
    # convert (0-255) image to specify range
    if model_name == 'c3d':
        if dataset_name=='k400':
            means = torch.tensor([114.7748, 107.7354, 99.4750], dtype=torch.float32)[:, None, None, None]
            x.add_(means)
            x[x > 255] = 255
            x[x < 0] = 0
            x = x / 255
            x = x.permute(1, 0, 2, 3)  # 帧数，通道，长宽
        else:
            means = torch.tensor([101.2198, 97.5751, 89.5303], dtype=torch.float32)[:, None, None, None]
            x.add_(means)
            x[x > 255] = 255
            x[x < 0] = 0
            x= x/255
            x=x.permute(1, 0, 2 , 3)   # 帧数，通道，长宽
    else:
        x=x
    return x

def vector_to_image(model_name,x):
    if model_name == 'c3d':
        means = torch.tensor([101.2198, 97.5751, 89.5303], dtype=torch.float32)[:, None, None, None].cuda()
        x[x > 1] = 1
        x[x < 0] = 0
        x=x*255
        x.sub_(means)
    return x




# 获取数据集
def generate_dataset(model_name, dataset_name):
    if model_name == 'c3d':
        sys.path.append('./datasets/c3d_dataset')
        test_dataset = get_test_set(dataset_name)
        sys.path.remove('./datasets/c3d_dataset')
    return test_dataset


# 获取数据集
def generate_train_dataset(model_name, dataset_name):
    if model_name == 'c3d':
        sys.path.append('./datasets/c3d_dataset')
        from datasets.c3d_dataset.dataset_c3d import get_training_set # 获取数据
        train_dataset = get_training_set(dataset_name)
        sys.path.remove('./datasets/c3d_dataset')

    return train_dataset



# 加载网络模型
def generate_model(model_name, dataset_name):
    assert model_name in ['c3d', 'tsm', 'tsn', 'slowfast']
    if model_name == 'c3d':
        if dataset_name == 'k400':
            from models.c3d.c3d import generate_model_c3d  # 加载模型
            model = generate_model_c3d(dataset_name)
            model.eval()
        else:
            from models.c3d.c3d import generate_model_c3d  # 加载模型
            model = generate_model_c3d(dataset_name)
            model.eval()

    if model_name=='slowfast':
        if dataset_name == 'ucf101':
            cfg = Config.fromfile('configs/recognition/slowfast/slowonly_k400_pretrained_r50_8x4x1_40e_ucf101_rgb.py')
            checkpoint_path ='configs/checkpoints/slowonly_k400_pretrained_r50_8x4x1_40e_ucf101_rgb_20210630-ee8c850f.pth'
            model = init_recognizer(cfg, checkpoint_path, device='cuda:0').eval()
        if dataset_name =='hmdb51':
            cfg = Config.fromfile('configs/recognition/slowfast/slowonly_k400_pretrained_r50_8x4x1_40e_hmdb51_rgb.py')
            checkpoint_path ='configs/checkpoints/slowonly_k400_pretrained_r50_8x4x1_40e_hmdb51_rgb_20210630-cee5f725.pth'
            model= init_recognizer(cfg, checkpoint_path, device='cuda:0').eval()
        if dataset_name =='k400':
            cfg = Config.fromfile('configs/recognition/slowfast/slowonly_r50_8x8x1_256e_kinetics400_rgb.py')
            checkpoint_path ='configs/checkpoints/slowonly_r50_256p_8x8x1_256e_kinetics400_rgb_20200820-75851a7d.pth'
            model= init_recognizer(cfg, checkpoint_path, device='cuda:0').eval()
    if model_name =='tsm':
        if dataset_name == 'ucf101':
            cfg = Config.fromfile('configs/recognition/tsm/tsm_k400_pretrained_r50_1x1x16_25e_ucf101_rgb.py')
            checkpoint_path = 'configs/checkpoints/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb_20210630-1fae312b.pth'
            model = init_recognizer(cfg, checkpoint_path, device='cuda:0').eval()
        if dataset_name == 'hmdb51':
            cfg = Config.fromfile('configs/recognition/tsm/tsm_k400_pretrained_r50_1x1x16_25e_hmdb51_rgb.py')
            checkpoint_path = 'configs/checkpoints/tsm_k400_pretrained_r50_1x1x16_25e_hmdb51_rgb_20210630-4785548e.pth'
            model = init_recognizer(cfg, checkpoint_path, device='cuda:0').eval()
        if dataset_name == 'k400':
            cfg = Config.fromfile('configs/recognition/tsm/tsm_r50_1x1x8_100e_kinetics400_rgb.py')
            checkpoint_path = 'configs/checkpoints/tsm_r50_256p_1x1x8_50e_kinetics400_rgb_20200726-020785e2.pth'
            model = init_recognizer(cfg, checkpoint_path, device='cuda:0').eval()
    if model_name =='tsn':
        if dataset_name == 'ucf101':
            cfg = Config.fromfile('configs/recognition/tsn/tsn_r50_1x1x3_75e_ucf101_rgb.py')
            checkpoint_path = 'configs/checkpoints/tsn_r50_1x1x3_75e_ucf101_rgb_20201023-d85ab600.pth'
            model = init_recognizer(cfg, checkpoint_path, device='cuda:0').eval()
        if dataset_name == 'hmdb51':
            cfg = Config.fromfile('configs/recognition/tsn/tsn_r50_1x1x8_50e_hmdb51_kinetics400_rgb.py')
            checkpoint_path = 'configs/checkpoints/tsn_r50_1x1x8_50e_hmdb51_kinetics400_rgb_20201123-7f84701b.pth'
            model = init_recognizer(cfg, checkpoint_path, device='cuda:0').eval()
        if dataset_name == 'k400':
            cfg = Config.fromfile('configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py')
            checkpoint_path = 'configs/checkpoints/tsn_r50_256p_1x1x3_100e_kinetics400_rgb_20200725-22592236.pth'
            model = init_recognizer(cfg, checkpoint_path, device='cuda:0').eval()

    return model


# 根据攻击标识的下坐标id获取对应的攻击图片的label
def get_attacked_targeted_label(model_name, data_name, attack_id):
    df = pd.read_csv('./targeted_exp/attacked_samples-{}-{}.csv'.format(model_name, data_name))
    targeted_label = df[df['attack_id'] == attack_id]['targeted_label'].values.tolist()[0]
    return targeted_label




def get_samples(model_name, data_name):
    if os.path.exists('./untargeted_exp/attacked_samples-{}-{}.pkl'.format(model_name, data_name)):
        with open('./untargeted_exp/attacked_samples-{}-{}.pkl'.format(model_name, data_name), 'rb') as ipt:
            attacked_ids = pickle.load(ipt)
    else:
        print('No pkl files')
        return None
    return attacked_ids

# 获取平均扰动大小
def pertubation(clean,adv):
    loss = torch.nn.L1Loss()
    average_pertubation = loss(clean, adv)
    return average_pertubation


def process_grad(gs, reinforce):
    if reinforce:
        g = torch.tensor(gs)
    else:
        g = torch.sign(gs)
    return g



def speed_up_process(vid,len_limit):
    actions_t = random.sample(range(0, 16), len_limit)
    key_t = torch.zeros(vid.size(0))
    key_t[actions_t] = 1
    neg_numpy = np.array(actions_t)
    actions_t = torch.from_numpy(neg_numpy)
    return actions_t, key_t
