import math
import sys
import torch
import torch.nn.functional as F
import numpy as np


def intrinsicreward(seq, actions, ignore_far_sim=True, temp_dist_thre=3, use_gpu=False):

    _seq = seq.detach()
    _actions = actions.detach()
    pick_idxs = _actions.squeeze().nonzero().squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1
    if num_picks == 0:
        reward = torch.tensor(0.)
        if use_gpu: reward = reward.cuda()
        return reward
    _seq = _seq.squeeze()
    n = _seq.size(0)
    if num_picks == 1:
        reward_div = torch.tensor(0.)
        if use_gpu: reward_div = reward_div.cuda()
    else:
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
        dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t())
        dissim_submat = dissim_mat[pick_idxs,:][:,pick_idxs]
        if ignore_far_sim:
            pick_mat = pick_idxs.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
        reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.))
    if num_picks == 1:
        reward_rep = torch.tensor(0.)
    else:
        dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_mat = dist_mat + dist_mat.t()
        dist_mat = dist_mat[:, pick_idxs]
        dist_mat = dist_mat.min(1, keepdim=True)[0]

        dist_mat = torch.pow(dist_mat, 1/2)
        reward_rep = torch.exp(-dist_mat.mean())
    reward = reward_div + reward_rep
    return reward



def attackreward(res,iter_num,vid,adv_vid):
    def pertubation(clean, adv):
        loss = torch.nn.L1Loss()
        average_pertubation = loss(clean, adv)
        return average_pertubation
    R = 0.0
    if (res):
        if (iter_num>15000):
            P = pertubation(vid, adv_vid).cpu().numpy()
            R = 0.999*np.exp(-(P/0.05))
            P = P.cpu()
        else:
            P = pertubation(vid, adv_vid).cpu().numpy()
            R = np.exp(-(P/0.05))
    else:
        R = -1
    return R


def untargeted_reward(vid_model,adv_vid,confidence,l):

    _, _, logits = vid_model(adv_vid[None, :])
    adv_confidence =(F.softmax(logits.detach(), 1)).view(-1)[l]
    reward1 = confidence.cuda()-adv_confidence.view(-1)
    return reward1*10, adv_confidence

def untargeted_reward0(vid_model,adv_vid,confidence,l):

    _, _, logits = vid_model(adv_vid[None, :])
    adv_confidence =(F.softmax(logits.detach(), 1)).view(-1)[l]
    reward1 = confidence.cuda()-adv_confidence.view(-1)
    return reward1*10

def target_reward_advantage(vid_model,adv_vid,label,target_class,pre_confidence,pre_hunxiao_confidence):

    _, _, logits = vid_model(adv_vid[None, :])
    adv_confidence = (F.softmax(logits.detach(), 1)).view(-1)[label].flatten().cpu().numpy()
    hunxiao_confidence = (F.softmax(logits.detach(), 1)).view(-1)[target_class]
    rh2 = torch.exp(hunxiao_confidence.cpu())
    ra2 = torch.exp(torch.tensor(adv_confidence))
    rh1 = torch.exp(pre_hunxiao_confidence)
    ra1 = torch.exp(pre_confidence)
    reward1 = (rh2-ra2)-(rh1-ra1)
    sq = torch.norm((rh1-ra1))*2
    reward = reward1 / sq
    r_attack = (rh2-ra2)/sq

    return reward, adv_confidence, hunxiao_confidence,r_attack

def target_reward_advantage( hunxiao_confidence, adv_confidence, pre_confidence,pre_hunxiao_confidence):

    rh2 = hunxiao_confidence.cpu()
    ra2 = torch.tensor(adv_confidence)
    rh1 = pre_hunxiao_confidence
    ra1 = pre_confidence
    vx2 = torch.exp(rh2-ra2)
    vx1 = torch.exp(rh1-ra1)
    reward = (vx2-vx1)/vx1
    r_attack = vx1
    return reward, adv_confidence, hunxiao_confidence, r_attack

def reward_advantage(logits,label,pre_confidence,pre_hunxiao_confidence,rein):

    adv_confidence = (F.softmax(logits.detach(), 1)).view(-1)[label].flatten().cpu().numpy()
    hunxiao_confidence = (F.softmax(logits.detach(), 1)).view(-1).sort()[0][-2]
    rh2 = torch.exp(hunxiao_confidence.cpu())
    ra2 = torch.exp(torch.tensor(adv_confidence))
    rh1 = torch.exp(pre_hunxiao_confidence)
    ra1 = torch.exp(pre_confidence)
    if rein:
        reward1 = (rh2 - ra2) - (rh1 - ra1)
        sq = torch.norm((rh1 - ra1)) * 2
        reward = reward1 / sq
        r_attack = (rh2 - ra2) / sq
    else:
        vx2 = torch.exp(rh2-ra2)
        vx1 = torch.exp(rh1-ra1)
        reward = (vx2-vx1)/vx1
        r_attack = vx1
    return reward, adv_confidence, hunxiao_confidence, r_attack

def sparse_closeto_dense_reward(model,adv_vid,actions_t,mask,rectified_directions,cur_lr):

    mask_list_t = actions_t
    MASK = torch.zeros(adv_vid.size())
    b = mask.numpy()
    c = b[mask_list_t]
    i = 0
    for x1, x2, y1, y2 in c:
        key = mask_list_t[i]
        x1 = torch.tensor(x1).to(torch.int64)
        x2 = torch.tensor(x2).to(torch.int64)
        y1 = torch.tensor(y1).to(torch.int64)
        y2 = torch.tensor(y2).to(torch.int64)
        MASK[key, :, y1:y2, x1:x2] = rectified_directions[i, :, :, :]
        i = i + 1

    adv_vid += cur_lr * MASK.cuda()
    ref_adv_vid = adv_vid+cur_lr * rectified_directions
    MASK = torch.zeros(adv_vid.size())
    MASK[key, :, :, :] = 1
    proposed_adv_vid = adv_vid+cur_lr * rectified_directions * (MASK.cuda())
    logits0: object
    _,_,logits0 = model(ref_adv_vid[None,:])
    _,_,logits1 = model(proposed_adv_vid[None,:])
    loss0 = -torch.max(logits0, 1)[0]
    loss1 = -torch.max(logits1,1)[0]
    reward = torch.exp(-torch.abs(loss0-loss1))
    return reward

def  confidence_eval(vid_model,adv_vid,label):
    _, _, logits = vid_model(adv_vid[None, :])
    adv_confidence = (F.softmax(logits.detach(), 1)).view(-1)[label]
    reward1 =  adv_confidence
    return reward1.flatten()


def div_frame(seq, actions, ignore_far_sim=True, temp_dist_thre=3, use_gpu=False):

    _seq = seq.detach()
    _actions = actions.detach()
    pick_idxs = _actions.squeeze().nonzero().squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1
    if num_picks == 0:
        reward = torch.tensor(0.)
        if use_gpu: reward = reward.cuda()
        return reward
    _seq = _seq.squeeze()
    n = _seq.size(0)
    if num_picks == 1:
        reward_div = torch.tensor(0.)
        if use_gpu: reward_div = reward_div.cuda()
    else:
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
        dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t())
        dissim_submat = dissim_mat[pick_idxs,:][:,pick_idxs]
        if ignore_far_sim:
            pick_mat = pick_idxs.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
        reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.))
    reward = reward_div
    return reward



def rep_frame(feature_t, features_pool):
    reward_rep = 0
    for ii in range(0, 16):
        reward_rep_t = math.exp(-1/16*torch.norm((features_pool[ii]- feature_t), p=2))
        reward_rep += reward_rep_t
    reward = reward_rep*1/16
    return reward

def rep_frame_mean(feature_pool,len_limit):
    feature_mean = torch.mean(feature_pool)
    rep_list = []
    for ii in range(0, 16):
        rep_list.append(math.exp(-1/16*torch.norm((feature_pool[ii] - feature_mean), p=2)))
    rep_id = np.argsort(rep_list)[-len_limit-2]

    reward_rep_list = []
    for ii in range(0, 16):
        reward_rep_list.append(math.exp(-1/16*torch.norm((feature_pool[ii] - feature_pool[rep_id]), p=2)))
    rep_mean = np.mean(reward_rep_list)-0.001
    return rep_mean

def sparse_reward(actions_t, T, sparse_exp):
    if sparse_exp:
       reward = math.exp(-1 / 16 * math.fabs(len(actions_t) - T))
    else:
       reward = ((len(actions_t) - T) / 16) ** 2
    return reward


