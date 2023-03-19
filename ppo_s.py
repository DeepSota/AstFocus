import math
from torch.distributions import Categorical
from ASMARTselection.utils_ada import init_hidden
import torch
import torch.nn as nn

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.reward_epochs = []
        self.is_terminals = []
        self.hidden = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.hidden[:]

    def clear_step_memory(self):
        self.actions[:] = self.actions[:-1]
        self.states[:] = self.states[:-1]
        self.logprobs[:] = self.logprobs[:-1]
        self.hidden[:] = self.hidden[:-1]

    def exchange_step_memory(self):
        self.actions.pop(-2)
        self.states.pop(-2)
        self.logprobs.pop(-2)
        self.hidden.pop(-2)

    def show_len(self):
        return len(self.logprobs)

class ActorCritic(nn.Module):
    def __init__(self, feature_dim, state_dim, action_dim,  hidden_state_dim=512, policy_conv=True):
        super(ActorCritic, self).__init__()

        if policy_conv:

            self.state_encoder = nn.Sequential(
                nn.Conv2d(feature_dim, 32, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(state_dim, hidden_state_dim),
                nn.ReLU()
            )

        else:
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, 2048),
                nn.ReLU(),
                nn.Linear(2048, hidden_state_dim),
                nn.ReLU()
            )

        self.lstm = nn.LSTM(hidden_state_dim, hidden_state_dim, batch_first=False)
        self.actor = nn.Sequential(
            nn.Linear(hidden_state_dim, action_dim),
            nn.Softmax(dim=-1))

        self.critic = nn.Sequential(
            nn.Linear(hidden_state_dim, 1))

        self.hidden_state_dim = hidden_state_dim
        self.action_dim = action_dim
        self.policy_conv = policy_conv
        self.feature_dim = feature_dim
        self.feature_ratio = int(math.sqrt(state_dim/feature_dim))

    def forward(self):
        raise NotImplementedError

    def act(self, state_ini, feature_pre, memory, restart_batch=False, training=True):
        if restart_batch:
            del memory.hidden[:]
            batch_size=1
            h_t = init_hidden(batch_size, 512)
            c_t = init_hidden(batch_size, 512)
            hx = (h_t, c_t)
            memory.hidden.append(hx)

        state = state_ini.unsqueeze(0)
        state_encoder = state
        state = self.state_encoder(state_encoder)
        state, hidden_output = self.lstm(state.view(state.size(0), state.size(1)), memory.hidden[-1])
        memory.hidden.append(hidden_output)
        state = state[0]
        action_probs = self.actor(state)
        dist = Categorical(action_probs)



        if training:
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            memory.states.append(state_ini)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
        else:
            action = action_probs.max(1)[1]

        return action, hidden_output

    def evaluate(self, state, action):
        seq_l = state.size(0)
        batch_size = 1
        state = self.state_encoder(state)
        state = state.view(seq_l, batch_size, -1)
        state, hidden = self.lstm(state)
        state = state.view(seq_l * batch_size, -1)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(torch.squeeze(action.view(seq_l * batch_size, -1))).cuda()
        dist_entropy = dist.entropy().cuda()
        state_value = self.critic(state)

        return action_logprobs.view(seq_l, batch_size), \
               state_value.view(seq_l, batch_size), \
               dist_entropy.view(seq_l, batch_size)


class PPO_s(nn.Module):
    def __init__(self, feature_dim, state_dim, action_dim,  hidden_state_dim, policy_conv, gpu=0,
                lr=0.0003, betas=(0.9, 0.999), gamma=0.99, K_epochs = 10, eps_clip=0.2, decay = 0.1):
        super(PPO_s, self).__init__()
        self.lr = lr
        self.betas = betas

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.decay = decay
        self.policy = ActorCritic(feature_dim, state_dim, action_dim, hidden_state_dim, policy_conv).cuda(gpu)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(feature_dim, state_dim, action_dim,  hidden_state_dim, policy_conv).cuda(gpu)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, feature_pre, memory, restart_batch=False, training=True):
        return self.policy_old.act(state, feature_pre, memory, restart_batch, training)

    def update(self, memory, sigma_s):
        rewards = []
        discounted_reward = 0

        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards).cuda()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        reward_epochs = torch.tensor(memory.reward_epochs).cuda()
        reward_epoch = reward_epochs[-1]
        old_states = torch.stack(memory.states, 0).cuda().detach()
        old_actions = torch.stack(memory.actions, 0).cuda().detach()
        old_logprobs = torch.stack(memory.logprobs, 0).cuda().detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs.view(-1) - old_logprobs.detach()).view(-1)
            advantages = rewards - self.decay *state_values.view(-1).detach()
            surr1 = (ratios * advantages).to(torch.float32)
            surr2 = (torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages).to(torch.float32)
            surr1_epoch = (reward_epoch * ratios).to(torch.float32)
            surr2_epoch = (torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * reward_epoch).to(torch.float32)
            epoch_cost = -torch.min(surr1_epoch, surr2_epoch)

            loss1 = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values.to(torch.float32).view(-1), rewards.to(torch.float32)) - 0.01 * dist_entropy.to(torch.float32).view(-1)
            loss_s = sigma_s*loss1.mean() + (1-sigma_s)*epoch_cost.mean()
            loss_total = loss_s

            loss1.requires_grad_(True)
            epoch_cost.requires_grad_(True)
            loss_total.requires_grad_(True)
            self.optimizer.zero_grad()
            loss_total.backward()
            for x in self.optimizer.param_groups[0]['params']:
                x.grad
            self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict())
        return ratios