import torch
from torch import  nn
import sys
sys.path.append("../")
from ASMARTselection.mobilenet import mobilenet_v2
from ASMARTselection.utils import random_crop, get_patch
from ASMARTselection.ppo_s import PPO_s, Memory
import torchvision
from PIL.Image import Image
try:
    from PIL import Image
except ImportError:
    import Image

class AST_S(nn.Module):

    def __init__(self, args, state_dim):
        super(AST_S, self).__init__()
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
        self.down = torchvision.transforms.Resize((args.patch_size, args.patch_size), interpolation=Image.BILINEAR)


    def forward(self, *argv, **kwargs):
        if kwargs["backbone_pred"]:
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
            input = kwargs["input"]
            _b, _tc, _h, _w = input.shape
            _t, _c = _tc // 3, 3


    def glance(self, vid):
        _t, _c, _h, _w = vid.shape  # input (T, C, H, W)
        downs_2d = vid.view(_t, _c, _h, _w)
        global_feat_map, _ = self.glancer(downs_2d)
        _, _featc, _feath, _featw = global_feat_map.shape
        return global_feat_map.view(1, _t, _featc, _feath, _featw)


    def one_step_act(self, img, feature, feature_pre, use_pre, resample, training=True):
        key, HX = self.focuser(img=img, feature=feature, feature_pre=feature_pre, use_pre=use_pre, resample = resample, training=training)
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

    def __init__(self, size=96, random=True, policy_params: dict = None):
        super(Focuser, self).__init__()
        self.patch_size = size
        self.random = random
        self.patch_sampler = PatchSampler(self.patch_size, self.random)
        self.memory = Memory()
        if not self.random:
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
            self.policy = PPO_s(self.policy_feature_dim, self.policy_state_dim, self.policy_action_dim, self.policy_hidden_state_dim, self.policy_conv, self.gpu, gamma=policy_params['gamma'], lr=policy_params['policy_lr'])
    
    def forward(self,  **kwargs):
        action_s, hx= self.policy.select_action(kwargs['feature'], kwargs['feature_pre'], self.memory, kwargs['use_pre'] , kwargs['resample'], kwargs['training'])
        standard_action, patch_size_list = self._get_standard_action(action_s)
        imgs = kwargs['img']
        _b = imgs.shape[0]
        key_s = self.patch_sampler.sample(imgs, standard_action)
        return key_s, hx
    

    def random_patching(self, imgs):
        key_random = self.patch_sampler.random_sample(imgs)
        return key_random


    def update(self,  fineS):
        self.policy.update(self.memory,  fineS)
        self.memory.clear_memory()
    
    def _get_standard_action(self, action):
        standard_action = self.standard_actions_set[self.policy_action_dim]
        return standard_action[action], None

class PatchSampler(nn.Module):

    def __init__(self, size=72, random=True) -> None:
        super(PatchSampler, self).__init__()
        self.random = random
        self.size = size

    def sample(self, imgs, action = None):
        if self.random:
            key_random = []
            for img in imgs:
                key_random.append(random_crop(img, self.size))
            return torch.stack(key_random)
        else:
            assert action != None
            return get_patch(imgs, action, self.size)

    def forward(self, *argv, **kwargs):
        raise NotImplementedError




