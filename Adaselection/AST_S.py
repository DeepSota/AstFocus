import torch
from torch import autograd, nn
import sys
sys.path.append("../")
from ops.transforms import *
from Adaselection.mobilenet import mobilenet_v2
from Adaselection.utils import random_crop, get_patch
from Adaselection.ppo_s import MAPPO, Memory_S
import torchvision
from PIL.Image import Image
try:
    from PIL import Image
except ImportError:
    import Image

class AST_S(nn.Module):
    def __init__(self, args):
        super(AST_S, self).__init__()
        self.patch_size = args.patch_size
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
        self.focuser = Focuser(args.patch_size,  policy_params)
        self.dropout = nn.Dropout(p=args.dropout)
        self.down = torchvision.transforms.Resize((args.patch_size, args.patch_size),interpolation=Image.BILINEAR)


    def one_step_act(self, img, feature, feature_pre, global_feat_map, hx, re_sample, training=True):
        key, HX = self.focuser(img=img, feature=feature, feature_pre=feature_pre, global_fea=global_feat_map, hx=hx, re_sample=re_sample, training=training)
        return key, HX


class Focuser(nn.Module):
    def __init__(self, size=66,  policy_params: dict = None):
        super(Focuser, self).__init__()
        self.patch_size = size
        self.patch_sampler = PatchSampler(self.patch_size)
        self.memory = Memory_S()

        assert policy_params != None
        self.standard_actions_set = {
            25: torch.Tensor([
                [0, 0], [0, 1 / 4], [0, 2 / 4], [0, 3 / 4], [0, 4 / 4],
                [1 / 4, 0], [1 / 4, 1 / 4], [1 / 4, 2 / 4], [1 / 4, 3 / 4], [1 / 4, 4 / 4],
                [2 / 4, 0], [2 / 4, 1 / 4], [2 / 4, 2 / 4], [2 / 4, 3 / 4], [2 / 4, 4 / 4],
                [3 / 4, 0], [3 / 4, 1 / 4], [3 / 4, 2 / 4], [3 / 4, 3 / 4], [3 / 4, 4 / 4],
                [4 / 4, 0], [4 / 4, 1 / 4], [4 / 4, 2 / 4], [4 / 4, 3 / 4], [4 / 4, 4 / 4],
            ]).cuda(),
            36: torch.Tensor([
                [0, 0], [0, 1/5], [0, 2/5], [0, 3/5], [0, 4/5], [0, 5/5],
                [1/5, 0], [1/5, 1/5], [1/5, 2/5], [1/5, 3/5], [1/5, 4/5], [1/5, 5/5],
                [2/5, 0], [2/5, 1/5], [2/5, 2/5], [2/5, 3/5], [2/5, 4/5], [2/5, 5/5],
                [3/5, 0], [3/5, 1/5], [3/5, 2/5], [3/5, 3/5], [3/5, 4/5], [3/5, 5/5],
                [4/5, 0], [4/5, 1/5], [4/5, 2/5], [4/5, 3/5], [4/5, 4/5], [4/5, 5/5],
                [5/5, 0], [5/5, 1/5], [5/5, 2/5], [5/5, 3/5], [5/5, 4/5], [5/5, 5/5],
            ]).cuda(),
            49: torch.Tensor([
                [0, 0], [0, 1/6], [0, 2/6], [0, 3/6], [0, 4/6], [0, 5/6], [0, 1],
                [1/6, 0], [1/6, 1/6], [1/6, 2/6], [1/6, 3/6], [1/6, 4/6], [1/6, 5/6], [1/6, 1],
                [2/6, 0], [2/6, 1/6], [2/6, 2/6], [2/6, 3/6], [2/6, 4/6], [2/6, 5/6], [2/6, 1],
                [3/6, 0], [3/6, 1/6], [3/6, 2/6], [3/6, 3/6], [3/6, 4/6], [3/6, 5/6], [3/6, 1],
                [4/6, 0], [4/6, 1/6], [4/6, 2/6], [4/6, 3/6], [4/6, 4/6], [4/6, 5/6], [4/6, 1],
                [5/6, 0], [5/6, 1/6], [5/6, 2/6], [5/6, 3/6], [5/6, 4/6], [5/6, 5/6], [5/6, 1],
                [6/6, 0], [6/6, 1/6], [6/6, 2/6], [6/6, 3/6], [6/6, 4/6], [6/6, 5/6], [6/6, 1],
            ]).cuda(),
            64: torch.Tensor([
                [0, 0], [0, 1/7], [0, 2/7], [0, 3/7], [0, 4/7], [0, 5/7], [0, 6/7], [0, 7/7],
                [1/7, 0], [1/7, 1/7], [1/7, 2/7], [1/7, 3/7], [1/7, 4/7], [1/7, 5/7], [1/7, 6/7], [1/7, 7/7],
                [2/7, 0], [2/7, 1/7], [2/7, 2/7], [2/7, 3/7], [2/7, 4/7], [2/7, 5/7], [2/7, 6/7], [2/7, 7/7],
                [3/7, 0], [3/7, 1/7], [3/7, 2/7], [3/7, 3/7], [3/7, 4/7], [3/7, 5/7], [3/7, 6/7], [3/7, 7/7],
                [4/7, 0], [4/7, 1/7], [4/7, 2/7], [4/7, 3/7], [4/7, 4/7], [4/7, 5/7], [4/7, 6/7], [4/7, 7/7],
                [5/7, 0], [5/7, 1/7], [5/7, 2/7], [5/7, 3/7], [5/7, 4/7], [5/7, 5/7], [5/7, 6/7], [5/7, 7/7],
                [6/7, 0], [6/7, 1/7], [6/7, 2/7], [6/7, 3/7], [6/7, 4/7], [6/7, 5/7], [6/7, 6/7], [6/7, 7/7],
                [7/7, 0], [7/7, 1/7], [7/7, 2/7], [7/7, 3/7], [7/7, 4/7], [7/7, 5/7], [7/7, 6/7], [7/7, 7/7],
            ]).cuda(),
            81: torch.Tensor([
                [0, 0], [0, 1 / 8], [0, 2 / 8], [0, 3 / 8], [0, 4 / 8], [0, 5 / 8], [0, 6 / 8], [0, 7 / 8],
                [0, 8 / 8],
                [1 / 8, 0], [1 / 8, 1 / 8], [1 / 8, 2 / 8], [1 / 8, 3 / 8], [1 / 8, 4 / 8], [1 / 8, 5 / 8],
                [1 / 8, 6 / 8], [1 / 8, 7 / 8], [1 / 8, 8 / 8],
                [2 / 8, 0], [2 / 8, 1 / 8], [2 / 8, 2 / 8], [2 / 8, 3 / 8], [2 / 8, 4 / 8], [2 / 8, 5 / 8],
                [2 / 8, 6 / 8], [2 / 8, 7 / 8], [2 / 8, 8 / 8],
                [3 / 8, 0], [3 / 8, 1 / 8], [3 / 8, 2 / 8], [3 / 8, 3 / 8], [3 / 8, 4 / 8], [3 / 8, 5 / 8],
                [3 / 8, 6 / 8], [3 / 8, 7 / 8], [3 / 8, 8 / 8],
                [4 / 8, 0], [4 / 8, 1 / 8], [4 / 8, 2 / 8], [4 / 8, 3 / 8], [4 / 8, 4 / 8], [4 / 8, 5 / 8],
                [4 / 8, 6 / 8], [4 / 8, 7 / 8], [4 / 8, 8 / 8],
                [5 / 8, 0], [5 / 8, 1 / 8], [5 / 8, 2 / 8], [5 / 8, 3 / 8], [5 / 8, 4 / 8], [5 / 8, 5 / 8],
                [5 / 8, 6 / 8], [5 / 8, 7 / 8], [5 / 8, 8 / 8],
                [6 / 8, 0], [6 / 8, 1 / 8], [6 / 8, 2 / 8], [6 / 8, 3 / 8], [6 / 8, 4 / 8], [6 / 8, 5 / 8],
                [6 / 8, 6 / 8], [6 / 8, 7 / 8], [6 / 8, 8 / 8],
                [7 / 8, 0], [7 / 8, 1 / 8], [7 / 8, 2 / 8], [7 / 8, 3 / 8], [7 / 8, 4 / 8], [7 / 8, 5 / 8],
                [7 / 8, 6 / 8], [7 / 8, 7 / 8], [7 / 8, 8 / 8],
                [8 / 8, 0], [8 / 8, 1 / 8], [8 / 8, 2 / 8], [8 / 8, 3 / 8], [8 / 8, 4 / 8], [8 / 8, 5 / 8],
                [8 / 8, 6 / 8], [8 / 8, 7 / 8], [8 / 8, 8 / 8],
            ]).cuda(),
            100: torch.Tensor([
                [0, 0], [0, 1 / 9], [0, 2 / 9], [0, 3 / 9], [0, 4 / 9], [0, 5 / 9], [0, 6 / 9], [0, 7 / 9],
                [0, 8 / 9], [0, 9 / 9],
                [1 / 9, 0], [1 / 9, 1 / 9], [1 / 9, 2 / 9], [1 / 9, 3 / 9], [1 / 9, 4 / 9], [1 / 9, 5 / 9],
                [1 / 9, 6 / 9], [1 / 9, 7 / 9], [1 / 9, 8 / 9], [1 / 9, 9 / 9],
                [2 / 9, 0], [2 / 9, 1 / 9], [2 / 9, 2 / 9], [2 / 9, 3 / 9], [2 / 9, 4 / 9], [2 / 9, 5 / 9],
                [2 / 9, 6 / 9], [2 / 9, 7 / 9], [2 / 9, 8 / 9], [2 / 9, 9 / 9],
                [3 / 9, 0], [3 / 9, 1 / 9], [3 / 9, 2 / 9], [3 / 9, 3 / 9], [3 / 9, 4 / 9], [3 / 9, 5 / 9],
                [3 / 9, 6 / 9], [3 / 9, 7 / 9], [3 / 9, 8 / 9], [3 / 9, 9 / 9],
                [4 / 9, 0], [4 / 9, 1 / 9], [4 / 9, 2 / 9], [4 / 9, 3 / 9], [4 / 9, 4 / 9], [4 / 9, 5 / 9],
                [4 / 9, 6 / 9], [4 / 9, 7 / 9], [4 / 9, 8 / 9], [4 / 9, 9 / 9],
                [5 / 9, 0], [5 / 9, 1 / 9], [5 / 9, 2 / 9], [5 / 9, 3 / 9], [5 / 9, 4 / 9], [5 / 9, 5 / 9],
                [5 / 9, 6 / 9], [5 / 9, 7 / 9], [5 / 9, 8 / 9], [5 / 9, 9 / 9],
                [6 / 9, 0], [6 / 9, 1 / 9], [6 / 9, 2 / 9], [6 / 9, 3 / 9], [6 / 9, 4 / 9], [6 / 9, 5 / 9],
                [6 / 9, 6 / 9], [6 / 9, 7 / 9], [6 / 9, 8 / 9], [6 / 9, 9 / 9],
                [7 / 9, 0], [7 / 9, 1 / 9], [7 / 9, 2 / 9], [7 / 9, 3 / 9], [7 / 9, 4 / 9], [7 / 9, 5 / 9],
                [7 / 9, 6 / 9], [7 / 9, 7 / 9], [7 / 9, 8 / 9], [7 / 9, 9 / 9],
                [8 / 9, 0], [8 / 9, 1 / 9], [8 / 9, 2 / 9], [8 / 9, 3 / 9], [8 / 9, 4 / 9], [8 / 9, 5 / 9],
                [8 / 9, 6 / 9], [8 / 9, 7 / 9], [8 / 9, 8 / 9], [8 / 9, 9 / 9],
                [9 / 9, 0], [9 / 9, 1 / 9], [9 / 9, 2 / 9], [9 / 9, 3 / 9], [9 / 9, 4 / 9], [9 / 9, 5 / 9],
                [9 / 9, 6 / 9], [9 / 9, 7 / 9], [9 / 9, 8 / 9], [9 / 9, 9 / 9],
            ]).cuda()
        }
        self.policy_feature_dim = policy_params['feature_dim']
        self.policy_state_dim = policy_params['state_dim']
        self.policy_action_dim = policy_params['action_dim']
        self.policy_hidden_state_dim = policy_params['hidden_state_dim']
        self.policy_conv = policy_params['policy_conv']
        self.gpu = policy_params['gpu']  # for ddp
        self.policy = MAPPO(self.policy_feature_dim, self.policy_state_dim, self.policy_action_dim, self.policy_hidden_state_dim, self.policy_conv, self.gpu, gamma=policy_params['gamma'], lr=policy_params['policy_lr'])

    def forward(self, *argv, **kwargs):

        action_s, hx= self.policy.select_action(kwargs['feature'], kwargs['feature_pre'], kwargs['global_fea'],
                                                     kwargs['hx'], self.memory, kwargs['re_sample'],
                                                     kwargs['training'])
        standard_action, patch_size_list = self._get_standard_action(action_s)

        imgs = kwargs['img']
        _b = imgs.shape[0]
        key_s = self.patch_sampler.sample(imgs, standard_action)
        return key_s, hx
    


    def update(self, vid, features, global_fea, obs_share_t, GetFeatures, XX):
        self.policy.update(self.memory, vid, features,  global_fea,  obs_share_t, GetFeatures, XX)
        self.memory.clear_memory()
    
    def _get_standard_action(self, action):
        standard_action = self.standard_actions_set[self.policy_action_dim]
        return standard_action[action], None

class PatchSampler(nn.Module):

    def __init__(self, size=72) -> None:
        super(PatchSampler, self).__init__()
        self.size = size

    def sample(self, imgs, action = None):
        assert action != None
        return get_patch(imgs, action, self.size)

    def forward(self, *argv, **kwargs):
        raise NotImplementedError




