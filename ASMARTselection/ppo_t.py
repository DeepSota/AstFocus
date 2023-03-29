from torch.distributions import Bernoulli
from ASMARTselection.modules import *


class Memory:
    def __init__(self):

        self.actions_t = []
        self.states = []
        self.logprobs_t = []
        self.rewards = []
        self.reward_epochs=[]
        self.is_terminals = []
        self.hidden = []
        self.hidden_ht = []
        self.hidden_ct = []

    def clear_memory(self):


        del self.actions_t[:]
        del self.states[:]
        del self.logprobs_t[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.hidden[:]
        del self.hidden_ht[:]
        del self.hidden_ct[:]

    def clear_reward(self):
        del self.rewards[:]


class ActorCritic(nn.Module):
    def __init__(self, feature_dim, state_dim, action_dim, hidden_state_dim=512, policy_conv=True):
        super(ActorCritic, self).__init__()

        self.tot_frames = 16
        self.beta =1.1
        self.max_prob = torch.tensor([[0.99]])
        self.embedding_size = 1280
        self.global_embedding_size = 1280

        self.hidden_size = 512
        self.small_hidden_size = 512

        self.rnn_input_size = 512 + self.small_hidden_size
        self.num_layers = 1
        self.vector_dim = 16
        proj_kwargs = {
            'input_dim': self.embedding_size,
            'hidden_dims': (512,),
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
        self.global_mlp = build_mlp(**global_proj_kwargs).cuda()
        self.hidden_mlp = build_mlp(**hidden_proj_kwargs).cuda()

        self.rnn = nn.LSTMCell(input_size=self.rnn_input_size,
                               hidden_size=self.hidden_size, bias=True)

        self.wdrnn = WeightDrop(self.rnn, ['weight_hh'], dropout=0.5)

        self.pe = PositionalEncoding(self.global_embedding_size, dropout=0.0, max_len=self.tot_frames)
        self.action_dim = action_dim


        self.state_encoder = nn.Sequential(
            nn.Conv2d(feature_dim, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_state_dim, hidden_state_dim),
            nn.ReLU()
        )

        self.actor_temporal = nn.Sequential(
            nn.Linear(hidden_state_dim, 1),
             )

        self.critic_temporal = nn.Sequential(
            nn.Linear(self.vector_dim, 1))

        self.hidden_state_dim = hidden_state_dim
        self.action_dim = action_dim
        self.policy_conv = policy_conv
        self.feature_dim = feature_dim
        self.feature_ratio = int(math.sqrt(state_dim/feature_dim))
        self.soft_attn = SoftAttention()
        self.emb = torch.nn.Embedding(self.vector_dim, self.vector_dim)

    def forward(self):
        raise NotImplementedError

    def act(self, feature, global_fea , hx, memory_t, restart_batch=False, training=True):

        global_fea = self.pe(global_fea)
        global_fea = proj_masking(global_fea, self.global_mlp).squeeze().unsqueeze(0)


        hidden_fea = self.hidden_mlp(hx[0])
        global_fea, attn = self.soft_attn(hidden_fea, global_fea)

        feature = feature.contiguous().cuda()
        feature1 = self.proj_mlp(feature)
        features = torch.cat([global_fea, feature1], 1)
        ht,ct=hx
        memory_t.hidden_ct.append(ht)
        memory_t.hidden_ht.append(ct)
        h_next, c_next = self.wdrnn(features, hx)
        hx = (h_next, c_next)
        prob_t = torch.sigmoid(self.actor_temporal(h_next))*self.beta
        prob_t = torch.min(prob_t, self.max_prob.cuda())
        dist_t = Bernoulli(prob_t)


        if training:

            memory_t.states.append(features)
            action_t = dist_t.sample().view(-1)
            action_t_logprob = dist_t.log_prob(action_t).view(-1)
            memory_t.actions_t.append(action_t)
            memory_t.logprobs_t.append(action_t_logprob)
        else:

            action_t = dist_t.sample()

        return action_t, hx

    def evaluate(self, state, hx, action):
        seq_l = state.size(0)
        batch_size=1
        state = state.view(seq_l,  -1)
        state, hidden = self.wdrnn(state, hx)
        state = state.view(seq_l, -1)

        action_probs = torch.sigmoid(self.actor_temporal(state)).view(-1)
        dist = Bernoulli(action_probs)
        action_logprobs = dist.log_prob(torch.squeeze(action.view(seq_l * batch_size, -1))).cuda()
        dist_entropy = dist.entropy().cuda()
        action = torch.tensor(action).to(torch.int64)
        state_critc = self.emb(action)
        state_value = self.critic_temporal(state_critc)

        return action_logprobs.view(seq_l, batch_size), \
               state_value.view(seq_l, batch_size), \
               dist_entropy.view(seq_l, batch_size)

class PPO_t(nn.Module):
    def __init__(self, feature_dim, state_dim, action_dim, hidden_state_dim, policy_conv, gpu=0,
                lr=0.0003, betas=(0.9, 0.999), gamma=0.99, K_epochs=10, eps_clip=0.2):
        super(PPO_t, self).__init__()
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy_t = ActorCritic(feature_dim, state_dim, action_dim, hidden_state_dim, policy_conv).cuda(gpu)

        self.optimizer = torch.optim.Adam(self.policy_t.parameters(), lr=lr, betas=betas)

        self.policy_old_t = ActorCritic(feature_dim, state_dim, action_dim, hidden_state_dim, policy_conv).cuda(gpu)
        self.policy_old_t.load_state_dict(self.policy_t.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, global_fea, hx, memory_t, restart_batch=False, training=True):
        return self.policy_old_t.act(state, global_fea,hx, memory_t, restart_batch, training)

    def update(self, memory_t,fineT):
        rewards = []
        discounted_reward = 0

        for reward in reversed(memory_t.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards).cuda()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        reward_epochs = torch.tensor(memory_t.reward_epochs).cuda()
        reward_epoch = reward_epochs[-1]
        old_states = torch.stack(memory_t.states, 0).cuda().detach()
        old_logprobs_t= torch.stack(memory_t.logprobs_t, 0).cuda().detach()
        old_actions_t = torch.stack(memory_t.actions_t, 0).cuda().detach()
        old_hidden_ht = torch.stack(memory_t.hidden_ht, 0).cuda().detach().squeeze()
        old_hidden_ct = torch.stack(memory_t.hidden_ct, 0).cuda().detach().squeeze()
        old_hx = old_hidden_ht, old_hidden_ct
        for _ in range(self.K_epochs):
            logprobs_t, state_values_t, dist_entropy_t = self.policy_t.evaluate(old_states, old_hx, old_actions_t)

            ratios = torch.exp(logprobs_t - old_logprobs_t .detach()).view(-1)
            advantages = rewards - state_values_t.view(-1).detach()
            surr1 = (ratios * advantages).to(torch.float32)
            surr2 = (torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages).to(torch.float32)

            surr1_epoch = (reward_epoch * ratios).to(torch.float32)
            surr2_epoch = (torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * reward_epoch).to(torch.float32)
            epoch_cost = -torch.min(surr1_epoch, surr2_epoch)

            loss1 = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values_t.to(torch.float32).view(-1), rewards.to(torch.float32)) - 0.01 * dist_entropy_t.to(torch.float32).view(-1)
            loss = fineT*loss1.mean() + (1-fineT)*epoch_cost.mean()
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


        self.policy_old_t.load_state_dict(self.policy_t.state_dict())