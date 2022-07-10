import torch
from torch.distributions import Categorical
from Adaselection.modules import build_mlp, PositionalEncoding, proj_masking, SoftAttention, WeightDrop
import torch.nn as nn
from Adaselection.utils import update_linear_schedule, get_gard_norm, huber_loss, mse_loss, get_patch, \
    feature_extractor
from Adaselection.utils_ada import init_hidden
from Adaselection.algorithms.utils.valuenorm import ValueNorm



class Memory_S:
    def __init__(self):

        self.actions_s = []
        self.actions_s_old = []
        self.states_s = []
        self.states_s_old = []
        self.obs_share_old = []
        self.obs_share = []
        self.logprobs_s = []
        self.logprobs_s_old = []
        self.dist_entropy =[]
        self.rewards = []
        self.reward_epochs = []
        self.is_terminals = []
        self.hidden = []

    def clear_memory(self):

        del self.actions_s[:]
        del self.actions_s_old[:]
        del self.states_s[:]
        del self.states_s_old[:]
        del self.obs_share[:]
        del self.obs_share_old[:]
        del self.logprobs_s[:]
        del self.logprobs_s_old[:]
        del self.dist_entropy[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.hidden[:]

class PatchSampler(nn.Module):

    def __init__(self, size=66, random=True) -> None:
        super(PatchSampler, self).__init__()
        self.size = size

    def sample(self, imgs, action = None):
            return get_patch(imgs, action, self.size)

class Actor(nn.Module):
    def __init__(self, args, action_dim, hidden_state_dim=512,  ckpt=None, max_steps=3, tot_frames=16, use_stop=False):
        super(Actor, self).__init__()
        self.patch_size=66
        self.ckpt = ckpt
        self.tot_frames = tot_frames
        self.max_steps = max_steps
        self.embedding_size = 1280  
        self.global_embedding_size = 512
        self.hidden_size = 512  
        self.small_hidden_size = 512
        self.rnn_input_size =  self.small_hidden_size
        self.num_layers = 1
        self.bidirectional = False
        self.use_stop = use_stop

        self.standard_actions_set = {
            25: torch.Tensor([
                [0, 0], [0, 1 / 4], [0, 2 / 4], [0, 3 / 4], [0, 4 / 4],
                [1 / 4, 0], [1 / 4, 1 / 4], [1 / 4, 2 / 4], [1 / 4, 3 / 4], [1 / 4, 4 / 4],
                [2 / 4, 0], [2 / 4, 1 / 4], [2 / 4, 2 / 4], [2 / 4, 3 / 4], [2 / 4, 4 / 4],
                [3 / 4, 0], [3 / 4, 1 / 4], [3 / 4, 2 / 4], [3 / 4, 3 / 4], [3 / 4, 4 / 4],
                [4 / 4, 0], [4 / 4, 1 / 4], [4 / 4, 2 / 4], [4 / 4, 3 / 4], [4 / 4, 4 / 4],
            ]).cuda(),
            36: torch.Tensor([
                [0, 0], [0, 1 / 5], [0, 2 / 5], [0, 3 / 5], [0, 4 / 5], [0, 5 / 5],
                [1 / 5, 0], [1 / 5, 1 / 5], [1 / 5, 2 / 5], [1 / 5, 3 / 5], [1 / 5, 4 / 5], [1 / 5, 5 / 5],
                [2 / 5, 0], [2 / 5, 1 / 5], [2 / 5, 2 / 5], [2 / 5, 3 / 5], [2 / 5, 4 / 5], [2 / 5, 5 / 5],
                [3 / 5, 0], [3 / 5, 1 / 5], [3 / 5, 2 / 5], [3 / 5, 3 / 5], [3 / 5, 4 / 5], [3 / 5, 5 / 5],
                [4 / 5, 0], [4 / 5, 1 / 5], [4 / 5, 2 / 5], [4 / 5, 3 / 5], [4 / 5, 4 / 5], [4 / 5, 5 / 5],
                [5 / 5, 0], [5 / 5, 1 / 5], [5 / 5, 2 / 5], [5 / 5, 3 / 5], [5 / 5, 4 / 5], [5 / 5, 5 / 5],
            ]).cuda(),
            49: torch.Tensor([
                [0, 0], [0, 1 / 6], [0, 2 / 6], [0, 3 / 6], [0, 4 / 6], [0, 5 / 6], [0, 1],
                [1 / 6, 0], [1 / 6, 1 / 6], [1 / 6, 2 / 6], [1 / 6, 3 / 6], [1 / 6, 4 / 6], [1 / 6, 5 / 6], [1 / 6, 1],
                [2 / 6, 0], [2 / 6, 1 / 6], [2 / 6, 2 / 6], [2 / 6, 3 / 6], [2 / 6, 4 / 6], [2 / 6, 5 / 6], [2 / 6, 1],
                [3 / 6, 0], [3 / 6, 1 / 6], [3 / 6, 2 / 6], [3 / 6, 3 / 6], [3 / 6, 4 / 6], [3 / 6, 5 / 6], [3 / 6, 1],
                [4 / 6, 0], [4 / 6, 1 / 6], [4 / 6, 2 / 6], [4 / 6, 3 / 6], [4 / 6, 4 / 6], [4 / 6, 5 / 6], [4 / 6, 1],
                [5 / 6, 0], [5 / 6, 1 / 6], [5 / 6, 2 / 6], [5 / 6, 3 / 6], [5 / 6, 4 / 6], [5 / 6, 5 / 6], [5 / 6, 1],
                [6 / 6, 0], [6 / 6, 1 / 6], [6 / 6, 2 / 6], [6 / 6, 3 / 6], [6 / 6, 4 / 6], [6 / 6, 5 / 6], [6 / 6, 1],
            ]).cuda(),
            64: torch.Tensor([
                [0, 0], [0, 1 / 7], [0, 2 / 7], [0, 3 / 7], [0, 4 / 7], [0, 5 / 7], [0, 6 / 7], [0, 7 / 7],
                [1 / 7, 0], [1 / 7, 1 / 7], [1 / 7, 2 / 7], [1 / 7, 3 / 7], [1 / 7, 4 / 7], [1 / 7, 5 / 7],
                [1 / 7, 6 / 7], [1 / 7, 7 / 7],
                [2 / 7, 0], [2 / 7, 1 / 7], [2 / 7, 2 / 7], [2 / 7, 3 / 7], [2 / 7, 4 / 7], [2 / 7, 5 / 7],
                [2 / 7, 6 / 7], [2 / 7, 7 / 7],
                [3 / 7, 0], [3 / 7, 1 / 7], [3 / 7, 2 / 7], [3 / 7, 3 / 7], [3 / 7, 4 / 7], [3 / 7, 5 / 7],
                [3 / 7, 6 / 7], [3 / 7, 7 / 7],
                [4 / 7, 0], [4 / 7, 1 / 7], [4 / 7, 2 / 7], [4 / 7, 3 / 7], [4 / 7, 4 / 7], [4 / 7, 5 / 7],
                [4 / 7, 6 / 7], [4 / 7, 7 / 7],
                [5 / 7, 0], [5 / 7, 1 / 7], [5 / 7, 2 / 7], [5 / 7, 3 / 7], [5 / 7, 4 / 7], [5 / 7, 5 / 7],
                [5 / 7, 6 / 7], [5 / 7, 7 / 7],
                [6 / 7, 0], [6 / 7, 1 / 7], [6 / 7, 2 / 7], [6 / 7, 3 / 7], [6 / 7, 4 / 7], [6 / 7, 5 / 7],
                [6 / 7, 6 / 7], [6 / 7, 7 / 7],
                [7 / 7, 0], [7 / 7, 1 / 7], [7 / 7, 2 / 7], [7 / 7, 3 / 7], [7 / 7, 4 / 7], [7 / 7, 5 / 7],
                [7 / 7, 6 / 7], [7 / 7, 7 / 7],
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



        proj_kwargs = {
            'input_dim': self.embedding_size,
            'hidden_dims': (384,),
            'use_batchnorm': False,
            'dropout': 0,
        }

        sub_proj_kwargs = {
            'input_dim': self.embedding_size,
            'hidden_dims': (128,),
            'use_batchnorm': False,
            'dropout': 0,
        }

        sub_proj_kwargs_global = {
            'input_dim': self.global_embedding_size,
            'hidden_dims': (256,),
            'use_batchnorm': False,
            'dropout': 0,
        }

        global_proj_kwargs = {
            'input_dim': self.global_embedding_size,  
            'hidden_dims': (self.small_hidden_size,),
            'use_batchnorm': False,
            'dropout': 0,
        }

        hidden_proj_kwargs = {
            'input_dim': self.hidden_size,  
            'hidden_dims': (self.small_hidden_size,),
            'use_batchnorm': False,
            'dropout': 0,
        }
        self.patch_sampler = PatchSampler(self.patch_size)
        self.proj_mlp = build_mlp(**proj_kwargs).cuda()
        self.sub_proj_mlp = build_mlp(**sub_proj_kwargs).cuda()
        self.sub_proj_mlp_global = build_mlp(**sub_proj_kwargs_global).cuda()
        self.global_mlp = build_mlp(**global_proj_kwargs).cuda()
        self.hidden_mlp = build_mlp(**hidden_proj_kwargs).cuda()
        self.getFeatures = feature_extractor()
        self.act_rnn = nn.LSTMCell(input_size=self.rnn_input_size,
                               hidden_size=self.hidden_size, bias=True)



        self.wdrnn = WeightDrop(self.act_rnn, ['weight_hh'], dropout=0.01)
        self.pe = PositionalEncoding(self.global_embedding_size, dropout=0.0, max_len=self.tot_frames)
        self.action_dim = action_dim
        self.actor_temporal = nn.Sequential(
            nn.Linear(512, 49),
            nn.Softmax(dim=-1)
             )

        self.stop = nn.Linear(self.hidden_size, 2)
        self.soft_attn = SoftAttention()


    def forward(self):
        raise NotImplementedError

    def _get_standard_action(self, action):
        self.policy_action_dim = 49
        standard_action = self.standard_actions_set[self.policy_action_dim]
        return standard_action[action], None
    def act(self, feature_cur, feature_pre, global_fea, hx, memory, re_sample, training=True):

        batch_size = 1
        feature_pre = feature_pre.contiguous().cuda()  
        feature_pre = self.sub_proj_mlp(feature_pre)
        feature_cur = feature_cur.contiguous().cuda()  
        feature_cur = self.proj_mlp(feature_cur).view(1, -1)
        state_obs_s = torch.cat([feature_cur, feature_pre.view(1, -1) ], 1)
        h_next, c_next = self.wdrnn(state_obs_s, hx)
        hx = (h_next, c_next)
        prob_s = self.actor_temporal(h_next.view(-1))
        dist_s = Categorical(prob_s)
        dist_entropy = dist_s.entropy().cuda()
        if training:
            action_s = dist_s.sample().view(-1)
            action_s_logprob = dist_s.log_prob(action_s).view(-1)
            memory.states_s_old.append(state_obs_s)
            memory.actions_s_old.append(action_s)
            memory.logprobs_s_old.append(action_s_logprob)
        else:
            action_s = dist_s.sample()

        return action_s, hx


    def re_act(self,vid, features,  global_fea_ori,GetFeatures):
        batch_size = 1
        h_t = init_hidden(batch_size, 512)
        c_t = init_hidden(batch_size, 512)
        hx = (h_t, c_t)
        steps = 1
        step = 0
        feature_pre = torch.ones(1, 1280)
        MASK_t = torch.zeros(vid.size())
        key_t_lists = []
        states_s_list=[]
        actions_s_list=[]
        logprobs_s_list=[]
        dist_entropy_list=[]
        MASK_s = torch.zeros(vid.size())
        for focus_time_step in range(vid.size(0)):
            img = vid[focus_time_step, :, :, :]
            feature_cur_ori = features[focus_time_step, :].clone()
            feature_pre = feature_pre.contiguous().cuda()  
            feature_pre = self.sub_proj_mlp(feature_pre)
            feature_cur = feature_cur_ori.contiguous().cuda()  
            feature_cur = self.proj_mlp(feature_cur).view(1, -1)
            state_obs_s = torch.cat([feature_cur, feature_pre.view(1, -1)], 1)
            h_next, c_next = self.wdrnn(state_obs_s, hx)
            hx = (h_next, c_next)
            prob_s = self.actor_temporal(h_next.view(-1))
            dist_s = Categorical(prob_s)
            dist_entropy = dist_s.entropy().cuda()
            action_s = dist_s.sample().view(-1)
            action_s_logprob = dist_s.log_prob(action_s).view(-1)


            states_s_list.append(state_obs_s)
            actions_s_list.append(action_s)
            logprobs_s_list.append(action_s_logprob)
            dist_entropy_list.append(dist_entropy.unsqueeze(0))
            standard_action, patch_size_list = self._get_standard_action(action_s)
            key_s = self.patch_sampler.sample(img.clone(), standard_action)
            ks = key_s.cpu().numpy()
            for x1, x2, y1, y2 in ks:
                MASK_s[focus_time_step, :, y1:y2, x1:x2] = 1
            feature_s = vid[focus_time_step, :, :, :].clone() * (MASK_s[focus_time_step, :, :, :].clone().cuda())
            _, feature_pre_s = GetFeatures(feature_s.view(1, feature_s.size(0), feature_s.size(1), feature_s.size(2)))
            feature_pre = feature_pre_s.squeeze()
        states_ss = torch.cat(states_s_list, 0)
        logprobs_ss = torch.cat(logprobs_s_list, 0)
        dist_entropys = torch.cat(dist_entropy_list,0)
        return states_ss, logprobs_ss, dist_entropys



class Critic(nn.Module):
    def __init__(self, args, action_dim, hidden_state_dim=512,  ckpt=None, max_steps=3, tot_frames=16, use_stop=False):
        super(Critic, self).__init__()
        self.ckpt = ckpt
        self.tot_frames = tot_frames
        self.max_steps = max_steps
        self.embedding_size = 1280  
        self.cen_embedding_size = 1024
        self.global_embedding_size = 512
        self.hidden_size = 1024  
        self.small_hidden_size = 1024
        self.rnn_input_size =  self.small_hidden_size
        self.num_layers = 1
        self.bidirectional = False
        self.use_stop = use_stop

        proj_kwargs = {
            'input_dim': self.embedding_size,
            'hidden_dims': (512,),
            'use_batchnorm': False,
            'dropout': 0,
        }

        cen_proj_kwargs = {
            'input_dim': self.cen_embedding_size,
            'hidden_dims': (1024,),
            'use_batchnorm': False,
            'dropout': 0,
        }



        global_proj_kwargs = {
            'input_dim': self.global_embedding_size,  
            'hidden_dims': (self.small_hidden_size,),
            'use_batchnorm': False,
            'dropout': 0,
        }

        hidden_proj_kwargs = {
            'input_dim': self.hidden_size,  
            'hidden_dims': (self.small_hidden_size,),
            'use_batchnorm': False,
            'dropout': 0,
        }

        self.proj_mlp = build_mlp(**proj_kwargs).cuda()
        self.cen_proj_mlp = build_mlp(**cen_proj_kwargs).cuda()
        self.global_mlp = build_mlp(**global_proj_kwargs).cuda()
        self.hidden_mlp = build_mlp(**hidden_proj_kwargs).cuda()

        def init(module, weight_init, bias_init, gain=1):
            weight_init(module.weight.data, gain=gain)
            bias_init(module.bias.data)
            return module

        self._use_orthogonal = True
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        self.v_out = init_(nn.Linear(self.hidden_size, 1))
        self.num_layers = 1
        self.bidirectional = False
        self.use_stop = use_stop
        self.pe = PositionalEncoding(self.global_embedding_size, dropout=0.0, max_len=self.tot_frames)
        self.action_dim = action_dim
        self.stop = nn.Linear(self.hidden_size, 2)
        self.soft_attn = SoftAttention()
        self.critic_rnn = nn.LSTMCell(input_size=self.rnn_input_size,
                               hidden_size=1024, bias=True)
        self.wdrnn = WeightDrop(self.critic_rnn, ['weight_hh'], dropout=0.01)

        self.critic_temporal = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
           )

    def evaluate(self, obs_share):
        state_s_value_list = []
        seq_l = obs_share.size(0)
        obs_share = obs_share.view(seq_l, 1, -1)
        batch_size = 1  
        h_t = init_hidden(batch_size, 1024)
        c_t = init_hidden(batch_size, 1024)

        for step in range(seq_l):

            hx = (h_t, c_t)
            act_feature = obs_share[step, :]
            act_feature = self.cen_proj_mlp(act_feature)
            HX, CX = self.wdrnn(act_feature, hx)
            h_t, c_t = HX, CX
            state_s_value = self.critic_temporal(HX)
            state_s_value_list.append(state_s_value)
        state_s_values = torch.cat(state_s_value_list, 0)
        return state_s_values.view(seq_l, batch_size)


class MAPPO(nn.Module):
    def __init__(self, feature_dim, state_dim, action_dim, hidden_state_dim, policy_conv, gpu=0,
                lr=0.0003, betas=(0.9, 0.999), gamma=0.9, K_epochs=1, eps_clip=0.2, value_loss_coef=0.1):
        super(MAPPO, self).__init__()
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.clip_param = 0.2
        self.drop_lamda = 0.002
        self._use_valuenorm = True
        self._use_popart = False
        self._use_huber_loss = True
        self.device = 'cuda:0'
        self.huber_delta = 10.0
        self._use_clipped_value_loss = True
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.value_loss_coef = value_loss_coef
        self.policy_act = Actor(feature_dim, state_dim, action_dim, hidden_state_dim, policy_conv).cuda(gpu)
        self.policy_critic = Critic(feature_dim, state_dim, action_dim, hidden_state_dim, policy_conv).cuda(gpu)
        self.act_optimizer = torch.optim.Adam(self.policy_act.parameters(), lr=lr, betas=betas)
        self.critic_optimizer = torch.optim.Adam(self.policy_critic.parameters(), lr=lr, betas=betas)
        self.policy_old_act = Actor(feature_dim, state_dim, action_dim, hidden_state_dim, policy_conv).cuda(gpu)
        self.policy_old_critic = Critic(feature_dim, state_dim, action_dim, hidden_state_dim, policy_conv).cuda(gpu)

        if self._use_popart:
            self.value_normalizer = self.policy_critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)

        self.policy_old_act.load_state_dict(self.policy_act.state_dict())
        self.policy_old_critic.load_state_dict(self.policy_critic.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, feature, feature_pre, global_fea, hx, memory, re_sample=False, training=True):

        return self.policy_old_act.act(feature, feature_pre, global_fea, hx, memory, re_sample, training)

    def cal_value_loss(self, values, value_preds_batch, return_batch):

        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        value_loss = value_loss.mean()
        return value_loss

    def update(self, memory, vid, features,   global_fea,  obs_share_t, GetFeatures,XX):
        rewards = []
        discounted_reward = 0


        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards).cuda()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)



        reward_epochs = torch.tensor(memory.reward_epochs).cuda()
        reward_epoch = reward_epochs[-1]
        old_states = torch.stack(memory.states_s_old, 0).cuda().detach()
        old_logprobs_s = torch.stack(memory.logprobs_s_old, 0).cuda().detach()
        old_actions_s = torch.stack(memory.actions_s_old, 0).cuda().detach()
        obs_share_old = torch.stack(memory.obs_share_old, 0).cuda().detach()
        value_old = self.policy_old_critic.evaluate(obs_share_old).detach()

        for ii in range(self.K_epochs):
            states_ss, logprobs_s, dist_entropy = self.policy_act.re_act(vid.detach(), features.detach(),   global_fea.detach(), GetFeatures)
            obs_share_s = states_ss.detach()
            a = obs_share_t
            b = obs_share_s.detach()
            obs_share_pre = torch.cat([a, b], 1)
            memory.obs_share.append(obs_share_pre)
            obs_share = torch.stack(memory.obs_share, 0)[-1].cuda().squeeze()
            ratios = torch.exp(logprobs_s - old_logprobs_s.view(-1)).view(-1)
            advantages = rewards - value_old.view(-1)
            
            surr1 = (ratios * advantages).to(torch.float32)
            surr2 = (torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages).to(torch.float32)

            surr1_epoch = (reward_epoch * ratios).to(torch.float32)
            surr2_epoch = (torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * reward_epoch).to(torch.float32)
            epoch_cost = -torch.min(surr1_epoch, surr2_epoch)

            loss1 = -torch.min(surr1, surr2) - 0.01 * dist_entropy.to(torch.float32).view(-1)
            loss = XX*loss1.mean() + (1-XX)*epoch_cost.mean()

            self.act_optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.act_optimizer.step()

            state_values_s = self.policy_critic.evaluate(obs_share.detach())
            value_loss = self.cal_value_loss(state_values_s, value_old, rewards) * self.value_loss_coef
            self.critic_optimizer.zero_grad()
            value_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

        self.policy_old_act.load_state_dict(self.policy_act.state_dict())
        self.policy_old_critic.load_state_dict(self.policy_critic.state_dict())
