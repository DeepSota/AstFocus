import os
import sys

import numpy as np
import torch
import pickle
import random
import pandas as pd
import torch.nn as nn
from Utils.transformser import Normalize,Compose,TemporalCenterCrop,\
    ToTensor,CenterCrop,ClassLabel,target_Compose,VideoID
from datasets.data_factory import get_validation_set, get_training_lrcn_set
from datasets.c3d_dataset.dataset_c3d import get_test_set   # 获取数据
target_transform = target_Compose([VideoID(), ClassLabel()])

# 获取每个视频的总帧数
def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))
    return value


# 将图片转换到分类器可以使用的向量
def image_to_vector(model_name, x):
    # convert (0-255) image to specify range
    if model_name == 'c3d':
        means = torch.tensor([101.2198, 97.5751, 89.5303], dtype=torch.float32)[:, None, None, None]
        x.add_(means)
        x[x > 255] = 255
        x[x < 0] = 0
        x= x/255
        x=x.permute(1, 0,2,3)   # 帧数，通道，长宽
    elif model_name == 'lrcn':
        # 均值和方差设置
        means = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None, None]
        stds = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None, None]
        x.mul_(stds).add_(means)
        x[x>1.0] = 1.0
        x[x<0.0] = 0.0
        x=x.permute(1,0,2,3)
    elif model_name == 'i3d':
        # 均值和方差设置
        means = torch.tensor([0.39608, 0.38182, 0.35067], dtype=torch.float32)[:, None, None, None]
        stds = torch.tensor([0.15199, 0.14856, 0.15698], dtype=torch.float32)[:, None, None, None]
        x.mul_(stds).add_(means)
        x[x>1.0] = 1.0
        x[x<0.0] = 0.0
        x=x.permute(1,0,2,3)
    return x


# 获取数据集
def generate_dataset(model_name, dataset_name):
    if model_name == 'c3d':
        sys.path.append('./datasets/c3d_dataset')
        test_dataset = get_test_set(dataset_name)
        sys.path.remove('./datasets/c3d_dataset')
    elif model_name == 'lrcn':
        target_transform = target_Compose([VideoID(), ClassLabel()])
        # 均值和方差设置
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # 设置转换函数
        norm_method = Normalize(mean, std)
        validation_transforms = {
            'spatial': Compose([CenterCrop(224),
                                ToTensor(255),
                                norm_method]),
            'temporal': TemporalCenterCrop(16),
            'target': target_transform
        }  # 测试时的数据转换
        test_dataset = get_validation_set(dataset_name, validation_transforms['spatial'],
                                      validation_transforms['temporal'],validation_transforms['target'])
    # elif model_name == 'i3d':
    #     target_transform = target_Compose([VideoID(), ClassLabel()])
    #     # 均值和方差设置
    #     mean = [0.39608, 0.38182, 0.35067]
    #     std = [0.15199, 0.14856, 0.15698]
    #     # 设置转换函数
    #     norm_method = Normalize(mean, std)
    #     validation_transforms = {
    #         'spatial': Compose([CenterCrop(224),
    #                             ToTensor(255),
    #                             norm_method]),
    #         'temporal': TemporalCenterCrop(16),
    #         'target': target_transform
    #     }  # 测试时的数据转换
    #     test_dataset = get_validation_set(dataset_name, validation_transforms['spatial'],
    #                                       validation_transforms['temporal'], validation_transforms['target'])
    return test_dataset


# 获取数据集
def generate_train_dataset(model_name, dataset_name):
    if model_name == 'c3d':
        sys.path.append('./datasets/c3d_dataset')
        from datasets.c3d_dataset.dataset_c3d import get_training_set # 获取数据
        train_dataset = get_training_set(dataset_name)
        sys.path.remove('./datasets/c3d_dataset')
    elif model_name == 'lrcn':
        target_transform = target_Compose([VideoID(), ClassLabel()])
        # 均值和方差设置
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # 设置转换函数
        norm_method = Normalize(mean, std)
        validation_transforms = {
            'spatial': Compose([CenterCrop(224),
                                ToTensor(255),
                                norm_method]),
            'temporal': TemporalCenterCrop(16),
            'target': target_transform
        }  # 测试时的数据转换
        train_dataset = get_training_lrcn_set(dataset_name, validation_transforms['spatial'],
                                      validation_transforms['temporal'],validation_transforms['target'])
    elif model_name == 'i3d':
        target_transform = target_Compose([VideoID(), ClassLabel()])
        # 均值和方差设置
        mean = [0.39608, 0.38182, 0.35067]
        std = [0.15199, 0.14856, 0.15698]
        # 设置转换函数
        norm_method = Normalize(mean, std)
        validation_transforms = {
            'spatial': Compose([CenterCrop(224),
                                ToTensor(255),
                                norm_method]),
            'temporal': TemporalCenterCrop(16),
            'target': target_transform
        }  # 测试时的数据转换
        train_dataset = get_training_lrcn_set(dataset_name, validation_transforms['spatial'],
                                         validation_transforms['temporal'], validation_transforms['target'])
    return train_dataset



# 加载网络模型
def generate_model(model_name, dataset_name):
    assert model_name in ['c3d', 'i3d', 'lrcn']
    if model_name == 'c3d':
        from models.c3d.c3d import generate_model_c3d  # 加载模型
        model = generate_model_c3d(dataset_name)
        model.eval()
    elif model_name == 'lrcn':
        from models.LRCN.LRCN import generate_model_lrcn
        model = generate_model_lrcn(dataset_name)
        model.eval()
    # elif model_name == 'i3d':
    #     from models.I3D.I3D import generate_model_i3d
    #     model = generate_model_i3d(dataset_name)
    #     model.eval()
    return model


# 获取分类器的分类结果的概率和类别
def classify(model,inp,model_name):
    if inp.shape[0] != 1:
        inp = torch.unsqueeze(inp, 0)
    if model_name=='lrcn':
        inp = inp.permute(2, 0, 1, 3, 4)
        inp = inp.cuda()  # GPU化
        with torch.no_grad():
            logits = model.forward(inp)
        logits = torch.mean(logits, dim=1)
        confidence_prob, pre_label = torch.topk(nn.functional.softmax(logits, 1), 1)  # confidence_probs就是值 pre_label 就是label的值
    elif model_name == 'i3d':
        inp = inp.cuda()  # GPU化
        with torch.no_grad():
            logits = model.forward(inp)
        logits = logits.squeeze(dim=2)
        confidence_prob, pre_label = torch.topk(nn.functional.softmax(logits, 1), 1)
    else:
        values, indices = torch.sort(-torch.nn.functional.softmax(model(inp)), dim=1)
        confidence_prob, pre_label = -float(values[:, 0]), int(indices[:, 0])
    return confidence_prob,pre_label


# 根据攻击标识的下坐标id获取对应的攻击图片的label
def get_attacked_targeted_label(model_name, data_name, attack_id):
    df = pd.read_csv('./targeted_exp/attacked_samples-{}-{}.csv'.format(model_name, data_name))
    targeted_label = df[df['attack_id'] == attack_id]['targeted_label'].values.tolist()[0]  # 找到 csv文件里 字典对应的attack_id 里面  对应的 target_label，转化成list只取第一个
    return targeted_label


# 获取用于攻击的测试数据集的index(I3D,LRCN)
def get_attacked_samples(model, test_data, nums_attack, model_name, data_name):
    if os.path.exists('./attacked_samples-{}-{}.pkl'.format(model_name, data_name)):
        # 存在攻击样本
        with open('./attacked_samples-{}-{}.pkl'.format(model_name, data_name), 'rb') as ipt:
            attacked_ids = pickle.load(ipt)
    else:
        # 随机生辰攻击样本
        random.seed(1024)
        idxs = random.sample(range(len(test_data)), len(test_data))  # 随机生成样本index
        attacked_ids = []   # 攻击样本的index
        # 保证当前模型可以正确分类此样本
        for i in idxs:
            clips, label = test_data[i]
            video_id = label[0]
            label = int(label[1])
            _, pre = classify(model, clips, model_name)
            if pre != label:
                pass
            else:
                attacked_ids.append(i)
            if len(attacked_ids) == nums_attack:
                break
        with open('./attacked_samples-{}-{}.pkl'.format(model_name, data_name), 'wb') as opt:
            pickle.dump(attacked_ids, opt)   # 将攻击样本的idx进行保存
    return attacked_ids

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

def random_speed_up(len_frame,len_limit):
    actions_t = random.sample(range(0, 16), len_limit)
    key_t = torch.zeros(len_frame)
    key_t[actions_t] = 1
    neg_numpy = np.array(actions_t)
    actions_t = torch.from_numpy(neg_numpy)
    return actions_t,key_t
