from torch.distributions import Bernoulli

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
from collections import defaultdict
import torch.nn as nn
import torch
import numpy as np
import random
from torchvision import models

# key frame selection by agent
def agent_output(agent,features):
    probs = agent(features[None, :]).flatten().view(-1, 1)        # probability value per frame
    t_probs = probs.data.cpu().squeeze().numpy()
    pidx = np.argsort(t_probs)[::-1].tolist()
    dist = Bernoulli(probs)             # bernoulli function
    return probs, pidx, dist



def finelist(Sidx, key_list, limit_len):
    '''
        Sidx: Sorted index
        key_list: key frames agent selected
        limit_len: The upper limit of key frames
    '''
    masklist = key_list.detach()                       # prevent parameter updates
    key = []
    F = 0
    for i in Sidx:
        if F > limit_len-1:
            break
        if i in masklist:
            key.append(i)
            F = F+1
    return key

def prep_a_net(model_name, shall_pretrain):
    model = getattr(torchvision.models, model_name)(shall_pretrain)
    if "resnet" in model_name:
        model.last_layer_name = 'fc'
    elif "mobilenet_v2" in model_name:
        model.last_layer_name = 'classifier'
    return model

def zero_pad(im, pad_size):
    pad_width = ((0, 0), (pad_size, pad_size), (pad_size, pad_size))
    return np.pad(im, pad_width, mode="constant")

def random_crop(im, size, pad_size=0):
    if pad_size > 0:
        im = zero_pad(im=im, pad_size=pad_size)
    h, w = im.shape[0:]
    if size == h:
        return im
    x1 = np.random.randint(0, w - size)
    x2 = x1+size
    y1 = np.random.randint(0, h - size)
    y2 = y1+size
    return x1, x2, y1, y2

def get_patch(images, action_sequence, patch_size):
    batchsize = 1
    image_size = images.size(2)
    patch_coordinate = torch.floor(action_sequence * (image_size - patch_size)).int().view(-1,2)
    key = []
    for i in range(batchsize):
      x1=(patch_coordinate[i, 0]).item()
      x2=(patch_coordinate[i, 0] + patch_size).item()
      y1=(patch_coordinate[i, 1]).item()
      y2=(patch_coordinate[i, 1] + patch_size).item()
      data = (x1, x2, y1, y2)
      key.append(data)
    return torch.tensor(key)

# sparse perturbations
def sparse_perturbation(perturbation,key):
    MASK = torch.zeros(perturbation.size())  # initialization of the mask
    MASK[key, :, :, :] = 1
    sparse_perturbation = perturbation * (MASK.cuda())
    return sparse_perturbation

class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def data_transforms_cifar(args):
    assert args.dataset in ['cifar10', 'imagenet']
    if args.dataset == 'cifar10':
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD = [0.24703233, 0.24348505, 0.26158768]
    elif args.dataset == 'imagenet':
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

    if args.dataset == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    elif args.dataset == 'imagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        valid_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])


def get_variable(inputs, cuda=False, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    if cuda:
        out = Variable(inputs.cuda(), **kwargs)
    else:
        out = Variable(inputs, **kwargs)
    return out

class PrefetchedWrapper(object):
    def prefetched_loader(loader):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda()
                next_target = next_target.cuda()
                next_input = next_input.float()

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.epoch = 0

    def __iter__(self):
        if (self.dataloader.sampler is not None and
            isinstance(self.dataloader.sampler,
                       torch.utils.data.distributed.DistributedSampler)):

            self.dataloader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(self.dataloader)

# To extract features per frame (512 dimensions)
class feature_extractor(nn.Module):
    def __init__(self):
        super(feature_extractor, self).__init__()
        # ResNet18 extract features from each video frame
        resnet = models.resnet18(pretrained=True)
        resnet.float()
        resnet.cuda()
        resnet.eval()
        module_list = list(resnet.children())
        self.conv5 = nn.Sequential(*module_list[: -2])
        self.pool5 = module_list[-2]

    def forward(self, vid):
        x = vid.clone()
        # IMAGENET data preprocessing
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32,
                            device=x.get_device())[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32,
                           device=x.get_device())[None, :, None, None]
        x = x.sub_(mean).div_(std)     # normalize
        res_conv5 = self.conv5(x)
        res_pool5 = self.pool5(res_conv5)
        res_pool5 = res_pool5.view(res_pool5.size(0), -1)

        return res_pool5


import numpy as np
import math
import torch


def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)


def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    return e ** 2 / 2


def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1
    return act_shape


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
    return img_Hh_Ww_c