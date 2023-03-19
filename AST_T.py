import torch
from torch import autograd, nn
import sys

from ASMARTselection.ppo_t import PPO_t, Memory
import math
sys.path.append("../")
from ASMARTselection.mobilenet import mobilenet_v2
from ASMARTselection.utils import random_crop, get_patch
import torchvision
from PIL.Image import Image
try:
    from PIL import Image
except ImportError:
    import Image

class AST_T(nn.Module):

    def __init__(self, args):
        super(AST_T, self).__init__()
        self.num_segments = args.num_segments
        self.num_class = args.num_classes
        self.glancer = None
        self.focuser = None
        self.classifier = None
        self.input_size = 112
        self.batch_size = 1
        self.patch_size = args.patch_size
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        self.glancer = Glancer(num_classes=self.num_class)
        state_dim = args.feature_map_channels * math.ceil(args.glance_size / 16) * math.ceil(args.glance_size / 16)
        policy_params = {
            'feature_dim': args.feature_map_channels,
            'state_dim': state_dim,
            'action_dim': args.action_dim,
            'hidden_state_dim': args.hidden_state_dim,
            'gpu': args.gpu,
            'gamma': args.gamma,
            'policy_lr': args.policy_rl,
            'policy_conv': args.policy_conv
        }
        self.focuser = Focuser(args.patch_size, args.random_patch, policy_params)
        self.dropout = nn.Dropout(p=args.dropout)
        self.down = torchvision.transforms.Resize((args.patch_size, args.patch_size),interpolation=Image.BILINEAR)


    def forward(self, *argv, **kwargs):
        if kwargs["backbone_pred"]:  # kwargs["backbone_pred"] 看设置的时候是 true or false
            input = kwargs["input"]
            _b, _tc, _h, _w = input.shape
            _t, _c = _tc // 3, 3
            input_2d = input.view(_b * _t, _c, _h, _w)
            if kwargs['glancer']:
                pred = self.glancer.predict(input_2d).view(_b, _t, -1)
            else:
                pred = self.focuser.predict(input_2d).view(_b, _t, -1)
            return pred

        elif kwargs["one_step"]:
            gpu = kwargs["gpu"]
            input = kwargs["input"]
            down_sampled = kwargs["scan"]
            _b, _tc, _h, _w = input.shape
            _t, _c = _tc // 3, 3
            input_2d = input.view(_b, _t, _c, _h, _w)

    def glance(self, input_prime):
        _b, _tc, _h, _w = input_prime.shape  # input (B, T*C, H, W)
        _t, _c = _tc // 3, 3
        downs_2d = input_prime.view(_b * _t, _c, _h, _w)
        global_feat_map, global_feat = self.glancer(downs_2d)
        _, _featc, _feath, _featw = global_feat_map.shape
        return global_feat_map.view(_b, _t, _featc, _feath, _featw), global_feat.view(_b, _t, -1)

    def one_step_act(self, feature,feature_pre, global_feat_map, hx, re_sample, training=True):
        key, HX = self.focuser(feature=feature, feature_pre=feature_pre, global_fea=global_feat_map, hx=hx, re_sample=re_sample, training=training)
        return key, HX




class Glancer(nn.Module):

    def __init__(self, skip=False, num_classes=51):
        super(Glancer, self).__init__()
        self.net = mobilenet_v2(pretrained=True)
        num_ftrs = self.net.last_channel
        self.net.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, num_classes),
        )
        self.skip = skip
    
    def forward(self, input):
        self.float()
        self.cuda()
        self.eval()
        return self.net.get_featmap(input)

    def predict(self, input):
        return self.net(input)
    

class Focuser(nn.Module):
    """
    Local network for focusing
    """
    def __init__(self, size=96, random=True, policy_params: dict = None):
        super(Focuser, self).__init__()
        self.patch_size = size
        self.random = random
        self.patch_sampler = PatchSampler(self.patch_size, self.random)
        # self.policy = None
        self.memory = Memory()

        self.policy_feature_dim = policy_params['feature_dim']
        self.policy_state_dim = policy_params['state_dim']
        self.policy_action_dim = policy_params['action_dim']
        self.policy_hidden_state_dim = policy_params['hidden_state_dim']
        self.policy_conv = policy_params['policy_conv']
        self.gpu = policy_params['gpu']  # for ddp
        self.policy = PPO_t(self.policy_feature_dim, self.policy_state_dim, self.policy_action_dim, self.policy_hidden_state_dim, self.policy_conv, self.gpu, gamma=policy_params['gamma'], lr=policy_params['policy_lr'])

    def forward(self, *argv, **kwargs):
        if self.random:
            standard_action = None
        else:
            action_t, hx = self.policy.select_action(kwargs['feature'], kwargs['global_fea'], kwargs['hx'] , self.memory, kwargs['re_sample'], kwargs['training'])

        key_t = action_t

        return key_t, hx
    

    def random_patching(self, imgs):
        key_random = self.patch_sampler.random_sample(imgs)
        return key_random

    def update(self,  fineT):
        self.policy.update(self.memory,  fineT)
        self.memory.clear_memory()

class PatchSampler(nn.Module):
    """
    Sample patch over the whole image
    """
    def __init__(self, size=96, random=True) -> None:
        super(PatchSampler, self).__init__()
        self.random = random
        self.size = size

    def sample(self, imgs, action = None):
        if self.random:

            key_random = []
            print(self.size)
            for img in imgs:
                key_random.append(random_crop(img, self.size))
            return torch.stack(key_random)
        else:

            assert action != None
            return get_patch(imgs, action, self.size)

    def random_sample(self, imgs):
        # crop at random position
        key_random = []
        for img in imgs:
            key_random.append(random_crop(img, self.size))
        return torch.stack(key_random)

    def forward(self, *argv, **kwargs):
        raise NotImplementedError




