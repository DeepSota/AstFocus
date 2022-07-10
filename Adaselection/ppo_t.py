import torch
from torch.distributions import Bernoulli
from Adaselection.modules import build_mlp, PositionalEncoding, proj_masking, SoftAttention, WeightDrop
import torch.nn as nn
from Adaselection.utils import update_linear_schedule, get_gard_norm, huber_loss, mse_loss
from Adaselection.utils_ada import init_hidden
from Adaselection.algorithms.utils.valuenorm import ValueNorm


class Memory_T:
    def __init__(self):
        
        self.actions_t = []
        self.actions_t_old = []
        self.states_t = []
        self.states_t_old = []
        self.obs_share_old = []
        self.obs_share = []
        self.logprobs_t = []
        self.logprobs_t_old = []
        self.dist_entropy =[]
        self.rewards = []
        self.reward_epochs = []
        self.is_terminals = []
        self.hidden = []

    def clear_memory(self):
        
        del self.actions_t[:]
        del self.actions_t_old[:]
        del self.states_t[:]
        del self.states_t_old[:]
        del self.obs_share[:]
        del self.obs_share_old[:]
        del self.logprobs_t[:]
        del self.logprobs_t_old[:]
        del self.dist_entropy[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.hidden[:]


class Actor(nn.Module):
    def __init__(self, args, action_dim, hidden_state_dim=512,  ckpt=None, max_steps=3, tot_frames=16, use_stop=False):
        super(Actor, self).__init__()
        self.ckpt = ckpt
        self.tot_frames = tot_frames
        self.max_steps = max_steps
        self.embedding_size = 1280  
        self.global_embedding_size = 512
        self.hidden_size = 512  
        self.small_hidden_size = 512
        self.rnn_input_size = self.small_hidden_size
        self.num_layers = 1
        self.bidirectional = False
        self.use_stop = use_stop

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
            'hidden_dims': (128,),
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
        self.sub_proj_mlp = build_mlp(**sub_proj_kwargs).cuda()
        self.sub_proj_mlp_global = build_mlp(**sub_proj_kwargs_global).cuda()
        self.global_mlp = build_mlp(**global_proj_kwargs).cuda()
        self.hidden_mlp = build_mlp(**hidden_proj_kwargs).cuda()
        self.act_rnn = nn.LSTMCell(input_size=self.rnn_input_size,
                               hidden_size=self.hidden_size, bias=True)


        self.wdrnn = WeightDrop(self.act_rnn, ['weight_hh'], dropout=0.5)
        self.pe = PositionalEncoding(self.global_embedding_size, dropout=0.0, max_len=self.tot_frames)
        self.action_dim = action_dim
        self.actor_temporal = nn.Sequential(
            nn.Linear(512, 1),
             )

        self.stop = nn.Linear(self.hidden_size, 2)
        self.soft_attn = SoftAttention()


    def forward(self):
        raise NotImplementedError


    def act(self, feature_cur, feature_pre, global_fea, hx, memory, re_sample, training=True):

        batch_size = 1
        global_fea = self.pe(global_fea)
        global_fea = proj_masking(global_fea, self.global_mlp)
        hidden_fea = self.hidden_mlp(hx[0])
        global_fea1, attn = self.soft_attn(hidden_fea, global_fea)
        global_fea = self.sub_proj_mlp_global(global_fea1)
        feature_cur = feature_cur.contiguous().cuda()  
        feature_cur = self.proj_mlp(feature_cur).view(1,-1)
        state_obs_t = torch.cat([feature_cur, global_fea.view(1,-1)], 1)   
        h_next, c_next = self.wdrnn(state_obs_t, hx)
        hx = (h_next, c_next)
        prob_t = self.actor_temporal(h_next.view(-1))
        prob_t = torch.sigmoid(prob_t)
        dist_t = Bernoulli(prob_t)
        dist_entropy = dist_t.entropy().cuda()
        if training:
            action_t = dist_t.sample().view(-1)
            action_t_logprob = dist_t.log_prob(action_t).view(-1)

            if re_sample:
                memory.states_t.append(state_obs_t)
                memory.actions_t.append(action_t)
                memory.logprobs_t.append(action_t_logprob)
                memory.dist_entropy.append(dist_entropy)
            else:
                memory.states_t_old.append(state_obs_t)
                memory.actions_t_old.append(action_t)
                memory.logprobs_t_old.append(action_t_logprob)
        else:
            action_t = dist_t.sample()

        return action_t, hx



    def re_act(self, vid, features,  global_fea_ori):
        batch_size = 1
        h_t = init_hidden(batch_size, 512)
        c_t = init_hidden(batch_size, 512)
        hx = (h_t, c_t)
        states_t_list = []
        actions_t_list = []
        logprobs_t_list = []
        dist_entropy_list = []

        for focus_time_step in range(vid.size(0)):
            feature_cur_ori = features[focus_time_step, :].clone()
            global_fea = self.pe(global_fea_ori.clone())
            global_fea = proj_masking(global_fea, self.global_mlp)
            hidden_fea = self.hidden_mlp(hx[0])
            global_fea1, attn = self.soft_attn(hidden_fea, global_fea)
            global_fea = self.sub_proj_mlp_global(global_fea1)
            
            feature_cur = feature_cur_ori.contiguous().cuda()  
            feature_cur = self.proj_mlp(feature_cur).view(1, -1)
            state_obs_t = torch.cat([feature_cur, global_fea.view(1, -1)], 1)
            h_next, c_next = self.wdrnn(state_obs_t, hx)
            hx = (h_next, c_next)
            prob_t = self.actor_temporal(h_next.view(-1))
            prob_t = torch.sigmoid(prob_t)
            dist_t = Bernoulli(prob_t)
            dist_entropy = dist_t.entropy().cuda()
            action_t = dist_t.sample().view(-1)
            action_t_logprob = dist_t.log_prob(action_t).view(-1)
            states_t_list.append(state_obs_t)
            actions_t_list.append(action_t)
            logprobs_t_list.append(action_t_logprob)
            dist_entropy_list.append(dist_entropy)


        states_ts = torch.cat(states_t_list, 0)
        logprobs_ts = torch.cat(logprobs_t_list, 0)
        dist_entropys = torch.cat(dist_entropy_list, 0)
        return states_ts, logprobs_ts, dist_entropys

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
        self.small_hidden_size =1024
        self.rnn_input_size = self.small_hidden_size

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


        self.pe = PositionalEncoding(self.global_embedding_size, dropout=0.0, max_len=self.tot_frames)
        self.action_dim = action_dim
        self.soft_attn = SoftAttention()

        self.critic_rnn = nn.LSTMCell(input_size=self.rnn_input_size,
                               hidden_size=1024, bias=True)
        self.wdrnn = WeightDrop(self.critic_rnn, ['weight_hh'], dropout=0.5)

        self.critic_temporal = nn.Sequential(
            nn.Linear(1024, 1),
            )

    def evaluate(self, obs_share):
        state_t_value_list = []
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
            state_t_value = self.critic_temporal(HX)
            state_t_value_list.append(state_t_value)

        state_t_values = torch.cat(state_t_value_list, 0)
        return state_t_values.view(seq_l, batch_size)


class MAPPO(nn.Module):
    def __init__(self, feature_dim, state_dim, action_dim, hidden_state_dim, policy_conv, gpu=0,
                lr=0.0001, betas=(0.9, 0.999), gamma=0.9, K_epochs=1, eps_clip=0.2, value_loss_coef=0.1):
        super(MAPPO, self).__init__()
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.clip_param = 0.2
        self._use_valuenorm = True
        self._use_popart = False
        self._use_huber_loss =True
        self.device ='cuda:0'
        self.huber_delta = 10.0
        self._use_clipped_value_loss =True
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

        self.policy_old_act.load_state_dict(self.policy_act .state_dict())
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

    def update(self, memory, vid, features,  global_fea, obs_share_s, XX):
        rewards = []
        discounted_reward = 0



        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards).cuda()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        reward_epochs = torch.tensor(memory.reward_epochs).cuda()
        reward_epoch = reward_epochs[-1]
        old_states = torch.stack(memory.states_t_old, 0).cuda().detach()
        old_logprobs_t = torch.stack(memory.logprobs_t_old, 0).cuda().detach()
        old_actions_t = torch.stack(memory.actions_t_old, 0).cuda().detach()
        obs_share_old = torch.stack(memory.obs_share_old, 0).cuda().detach()
        value_old = self.policy_old_critic.evaluate(obs_share_old).detach()

        for ii in range(self.K_epochs):

            states_ts, logprobs_t, dist_entropy = self.policy_act.re_act(vid, features,   global_fea)
            obs_share_t = states_ts
            a = obs_share_t[:]
            b = obs_share_s[:]

            obs_share_pre = torch.cat([a, b], 1).detach()

            memory.obs_share.append(obs_share_pre)
            obs_share = torch.stack(memory.obs_share, 0).cuda()[-1].squeeze()
            state_values_t = self.policy_critic.evaluate(obs_share)
            value_loss = self.cal_value_loss(state_values_t, value_old, rewards) * self.value_loss_coef
            self.critic_optimizer.zero_grad()
            value_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

            ratios = torch.exp(logprobs_t - old_logprobs_t.view(-1) .detach()).view(-1)
            advantages = rewards - value_old.view(-1)
            surr1 = (ratios * advantages).to(torch.float32)
            surr2 = (torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages).to(torch.float32)

            surr1_epoch = (reward_epoch * ratios).to(torch.float32)
            surr2_epoch = (torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * reward_epoch).to(torch.float32)
            epoch_cost = -torch.min(surr1_epoch, surr2_epoch)

            loss1 = -torch.min(surr1, surr2) - 0.01 * dist_entropy.to(torch.float32).view(-1)
            loss = loss1.mean() + epoch_cost.mean()
            loss = XX * loss1.mean() + (1-XX) * epoch_cost.mean()
            self.act_optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.mean().backward(retain_graph=True)
            self.act_optimizer.step()

        self.policy_old_act.load_state_dict(self.policy_act.state_dict())
        self.policy_old_critic.load_state_dict(self.policy_critic.state_dict())
