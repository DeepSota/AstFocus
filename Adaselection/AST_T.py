import torch
from torchvision import models
from torch import autograd, nn
import sys
from Adaselection.ppo_t import MAPPO, Memory_T
import math
sys.path.append("../")
from Adaselection.mobilenet import mobilenet_v2
from Adaselection.utils import random_crop, get_patch
try:
    from PIL import Image
except ImportError:
    import Image

class AST_T(nn.Module):

    def __init__(self, args):
        super(AST_T, self).__init__()
        self.num_class = args.num_classes
        self.extractorr = extractor(num_classes=self.num_class)
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
        self.focuser = Focuser(args.patch_size, policy_params)



    def extractor(self, input_prime):
        _b, _tc, _h, _w = input_prime.shape
        _t, _c = _tc // 3, 3
        downs_2d = input_prime.view(_b * _t, _c, _h, _w)
        global_feat_map, global_feat = self.extractorr(downs_2d)
        _, _featc, _feath, _featw = global_feat_map.shape
        return global_feat_map.view(_b, _t, _featc, _feath, _featw), global_feat.view(_b, _t, -1)

    def one_step_act(self, feature, feature_pre, global_feat_map, hx, re_sample, training=True):
        key, HX = self.focuser(feature=feature, feature_pre=feature_pre, global_fea=global_feat_map, hx=hx, re_sample=re_sample, training=training)
        return key, HX




class extractor(nn.Module):
    def __init__(self, skip=False, num_classes=51):
        super(extractor, self).__init__()
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



class feature_extractor(nn.Module):
    def __init__(self):
        super(feature_extractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        resnet.float()
        resnet.cuda()
        resnet.eval()
        module_list = list(resnet.children())
        self.conv5 = nn.Sequential(*module_list[: -2])
        self.pool5 = module_list[-2]

    # 获取图片对应的特征
    def forward(self, vid):
        x = vid.clone()
        # IMAGENET数据预处理
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32,
                            device=x.get_device())[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32,
                           device=x.get_device())[None, :, None, None]
        x = x.sub_(mean).div_(std)     # 标准化
        res_conv5 = self.conv5(x)
        res_pool5 = self.pool5(res_conv5)
        res_pool5 = res_pool5.view(res_pool5.size(0), -1)
        return res_pool5
class Focuser(nn.Module):
    def __init__(self, size=96,  policy_params: dict = None):
        super(Focuser, self).__init__()
        self.patch_size = size
        self.patch_sampler = PatchSampler(self.patch_size)
        self.memory = Memory_T()

        self.policy_feature_dim = policy_params['feature_dim']
        self.policy_state_dim = policy_params['state_dim']
        self.policy_action_dim = policy_params['action_dim']
        self.policy_hidden_state_dim = policy_params['hidden_state_dim']
        self.policy_conv = policy_params['policy_conv']
        self.gpu = policy_params['gpu']
        self.policy = MAPPO(self.policy_feature_dim, self.policy_state_dim, self.policy_action_dim, self.policy_hidden_state_dim, self.policy_conv, self.gpu, gamma=policy_params['gamma'], lr=policy_params['policy_lr'])

    def forward(self, *argv, **kwargs):

        action_t, hx = self.policy.select_action(kwargs['feature'], kwargs['feature_pre'], kwargs['global_fea'], kwargs['hx'], self.memory, kwargs['re_sample'], kwargs['training'])

        key_t = action_t

        return key_t, hx


    def update(self, vid, features,   global_fea, obs_share_s, fineT):
        self.policy.update(self.memory, vid, features, global_fea, obs_share_s, fineT)
        self.memory.clear_memory()

class PatchSampler(nn.Module):

    def __init__(self, size=66) -> None:
        super(PatchSampler, self).__init__()
        self.size = size

    def sample(self, imgs, action = None):
        assert action != None
        return get_patch(imgs, action, self.size)

    def forward(self, *argv, **kwargs):
        raise NotImplementedError




