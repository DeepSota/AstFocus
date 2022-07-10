import sys
import torch
import torch.nn.functional as F
import numpy as np

# -------------------------计算奖励值（基于帧间不相似性以及帧的代表性）-------------------------
def intrinsicreward(seq, actions, ignore_far_sim=True, temp_dist_thre=3, use_gpu=False):
    """
    计算差异性奖励值和表示能力奖励值
    输入:
        seq: 特征序列形状为(1, seq_len, dim)
        actions: 二进制动作序列形状为(1, seq_len, 1)
        ignore_far_sim (bool): 是否考虑时序距离相似性（默认为True）
        temp_dist_thre (int): 阈值用于时域距离相似性（默认为20）
        use_gpu (bool): 是否使用GPU
    """
    _seq = seq.detach()          # 将seq变量从torch变量中分离，不参与参数更新
    _actions = actions.detach()  # 将actions变量从torch变量中分离，不参与参数更新
    pick_idxs = _actions.squeeze().nonzero().squeeze()                 # actions中非零元素的下坐标
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1    # 选择的帧的数量
    # 没有帧被选中，返回的奖励为0
    if num_picks == 0:
        reward = torch.tensor(0.)
        if use_gpu: reward = reward.cuda()
        return reward
    _seq = _seq.squeeze()   # 从数组的形状中删除单维度条目，即把shape中为1的维度删掉
    n = _seq.size(0)        # 序列seq的长度
    # --------------------------------计算差异性奖励值------------------------------
    if num_picks == 1:
        # 只有一帧被选中
        reward_div = torch.tensor(0.)
        if use_gpu: reward_div = reward_div.cuda()
    else:
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)     # 标准化
        dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t())  # 不相似性矩阵（完整帧序列）
        dissim_submat = dissim_mat[pick_idxs,:][:,pick_idxs]        # 选出的核心帧的不相似性矩阵
        if ignore_far_sim:
            # 考虑时域距离
            pick_mat = pick_idxs.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
        reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.))   # 平均不相似性
    # ------------------------------表示能力奖励值---------------------------------
    if num_picks == 1:
        reward_rep = torch.tensor(0.)  # 只有一帧
    else:
        dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)    # 将平方和扩展为n×n的矩阵
        dist_mat = dist_mat + dist_mat.t()
        # dist_mat =dist_mat-2*dist_mat*dist_mat  # addmm_ ：inputs   = 1 × inputs - 2 ×（inputs @ inputs_t）
        dist_mat = dist_mat[:, pick_idxs]                                       # 距离矩阵
        dist_mat = dist_mat.min(1, keepdim=True)[0]                            # 最小距离kmedoid

        dist_mat = torch.pow(dist_mat, 1/2)
        reward_rep = torch.exp(-dist_mat.mean())                               # 最小距离的指数平方  奖励像素值大的点
    # 将两种奖励值结合
    reward = reward_div + reward_rep
    # 返回当前actions对应的奖励值
    return reward


# 有目标攻击函数产的reward
def attackreward(res,iter_num,vid,adv_vid):
    # 获取平均扰动大小
    def pertubation(clean, adv):
        loss = torch.nn.L1Loss()
        average_pertubation = loss(clean, adv)
        return average_pertubation
    R = 0.0
    if (res):
        if (iter_num>15000):
            P = pertubation(vid, adv_vid).cpu().numpy()       # 平均每个像素的扰动量
            R = 0.999*np.exp(-(P/0.05))     # 越小的扰动量对应越大的reward
            P = P.cpu()
        else:
            P = pertubation(vid, adv_vid).cpu().numpy()        # 平均每个像素的扰动量
            R = np.exp(-(P/0.05))           # 越小的扰动量对应越大的reward
    else:
        R = -1
    # 返回攻击奖励以及平均扰动量的大小
    return R

# reward function
def untargeted_reward(vid_model,adv_vid,confidence,l):
    '''
    Reward values are drived from：
            Sparse attack is as close as possible to the effect of dense attack
    '''
    # Get the logits and calculate the corresponding loss function

    _, _, logits = vid_model(adv_vid[None, :])
    adv_confidence =(F.softmax(logits.detach(), 1)).view(-1)[l]
    reward1 = confidence.cuda()-adv_confidence.view(-1)
    return reward1*10, adv_confidence

def untargeted_reward0(vid_model,adv_vid,confidence,l):
    '''
    Reward values are drived from：
            Sparse attack is as close as possible to the effect of dense attack
    '''
    # Get the logits and calculate the corresponding loss function

    _, _, logits = vid_model(adv_vid[None, :])
    adv_confidence =(F.softmax(logits.detach(), 1)).view(-1)[l]
    reward1 = confidence.cuda()-adv_confidence.view(-1)
    return reward1*10
def target_reward_advantage(vid_model,adv_vid,label,target_class,pre_confidence,pre_hunxiao_confidence):
    '''
    Reward values are drived from：
            Sparse attack is as close as possible to the effect of dense attack
    '''
    # Get the logits and calculate the corresponding loss function

    _, _, logits = vid_model(adv_vid[None, :])
    adv_confidence = (F.softmax(logits.detach(), 1)).view(-1)[label].flatten().cpu().numpy()
    hunxiao_confidence = (F.softmax(logits.detach(), 1)).view(-1)[target_class]

    reward0 = (pre_confidence - torch.tensor(adv_confidence))/ pre_confidence
    # r00 = 1-torch.tensor(adv_confidence)
    # reward1 = (pre_confidence - torch.tensor(adv_confidence)) + (hunxiao_confidence.cpu() - pre_hunxiao_confidence)
    # r1 = ((hunxiao_confidence.cpu()-torch.tensor(adv_confidence))-(pre_hunxiao_confidence-pre_confidence))/torch.norm(pre_hunxiao_confidence-pre_confidence)
    rh2 = torch.exp(hunxiao_confidence.cpu())
    ra2 = torch.exp(torch.tensor(adv_confidence))
    rh1 = torch.exp(pre_hunxiao_confidence)
    ra1 = torch.exp(pre_confidence)
    reward1 = (rh2-ra2)-(rh1-ra1)
    sq = torch.norm((rh1-ra1))*2
    reward = reward1 / sq
    r_attack = (rh2-ra2)/sq

    return reward, adv_confidence, hunxiao_confidence,r_attack

def reward_advantage(vid_model,adv_vid,label,pre_confidence,pre_hunxiao_confidence):
    '''
    Reward values are drived from：
            Sparse attack is as close as possible to the effect of dense attack
    '''
    # Get the logits and calculate the corresponding loss function

    _, _, logits = vid_model(adv_vid[None, :])
    adv_confidence = (F.softmax(logits.detach(), 1)).view(-1)[label].flatten().cpu().numpy()
    hunxiao_confidence = (F.softmax(logits.detach(), 1)).view(-1).sort()[0][-2]

    reward0 = (pre_confidence - torch.tensor(adv_confidence))/ pre_confidence
    # r00 = 1-torch.tensor(adv_confidence)
    # reward1 = (pre_confidence - torch.tensor(adv_confidence)) + (hunxiao_confidence.cpu() - pre_hunxiao_confidence)
    # r1 = ((hunxiao_confidence.cpu()-torch.tensor(adv_confidence))-(pre_hunxiao_confidence-pre_confidence))/torch.norm(pre_hunxiao_confidence-pre_confidence)
    rh2 = torch.exp(hunxiao_confidence.cpu())
    ra2 = torch.exp(torch.tensor(adv_confidence))
    rh1 = torch.exp(pre_hunxiao_confidence)
    ra1 = torch.exp(pre_confidence)
    reward1 = (rh2-ra2)-(rh1-ra1)
    sq = torch.norm((rh1-ra1))*2
    reward = reward1 / sq
    r_attack = (rh2-ra2)/sq

    return reward, adv_confidence, hunxiao_confidence,r_attack
def sparse_closeto_dense_reward(model,adv_vid,actions_t,mask,rectified_directions,cur_lr):
    '''
    Reward values are drived from：
            Sparse attack is as close as possible to the effect of dense attack
    '''
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
    ref_adv_vid = adv_vid+cur_lr * rectified_directions     # dense attack
    MASK = torch.zeros(adv_vid.size())                      # Initialization of the mask
    MASK[key, :, :, :] = 1                               # The key mask is assigned to 1
    proposed_adv_vid = adv_vid+cur_lr * rectified_directions * (MASK.cuda())   # Sparse attack

    # Get the logits and calculate the corresponding loss function
    logits0: object
    _,_,logits0 = model(ref_adv_vid[None,:])
    _,_,logits1 = model(proposed_adv_vid[None,:])
    loss0 = -torch.max(logits0, 1)[0] #max（，1）之后由两部分组成 值和索引 所以要用max（）只取出值
    loss1 = -torch.max(logits1,1)[0]
    # print(loss0,loss1)
    reward = torch.exp(-torch.abs(loss0-loss1))
    return reward

def  confidence_eval(vid_model,adv_vid,label):
    _, _, logits = vid_model(adv_vid[None, :])
    adv_confidence = (F.softmax(logits.detach(), 1)).view(-1)[label]
    reward1 =  adv_confidence  # torch.exp(hunxiao_confidence - adv_confidence)
    return reward1.flatten()


def div_frame(seq, actions, ignore_far_sim=True, temp_dist_thre=3, use_gpu=False):
    """
    计算差异性奖励值和表示能力奖励值
    输入:
        seq: 特征序列形状为(1, seq_len, dim)
        actions: 二进制动作序列形状为(1, seq_len, 1)
        ignore_far_sim (bool): 是否考虑时序距离相似性（默认为True）
        temp_dist_thre (int): 阈值用于时域距离相似性（默认为20）
        use_gpu (bool): 是否使用GPU
    """
    _seq = seq.detach()          # 将seq变量从torch变量中分离，不参与参数更新
    _actions = actions.detach()  # 将actions变量从torch变量中分离，不参与参数更新
    pick_idxs = _actions.squeeze().nonzero().squeeze()                 # actions中非零元素的下坐标  #nonzero 返回非零元素下标
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1    # 选择的帧的数量
    # 没有帧被选中，返回的奖励为0
    if num_picks == 0:
        reward = torch.tensor(0.)
        if use_gpu: reward = reward.cuda()
        return reward
    _seq = _seq.squeeze()   # 从数组的形状中删除单维度条目，即把shape中为1的维度删掉
    n = _seq.size(0)        # 序列seq的长度
    # --------------------------------计算差异性奖励值------------------------------
    if num_picks == 1:
        # 只有一帧被选中
        reward_div = torch.tensor(0.)
        if use_gpu: reward_div = reward_div.cuda()
    else:
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)     # 标准化
        dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t())  # 不相似性矩阵（完整帧序列）
        dissim_submat = dissim_mat[pick_idxs,:][:,pick_idxs]        # 选出的核心帧的不相似性矩阵
        if ignore_far_sim:
            # 考虑时域距离
            pick_mat = pick_idxs.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1. # 差别三帧以上的 本来差别就不小了 直接定为1
        reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.))   # 平均不相似性   奖励不相似 惩罚选的帧数过多
    # 将两种奖励值结合
    reward = reward_div
    # 返回当前actions对应的奖励值
    return reward



def rep_frame(seq, actions, ignore_far_sim=True, temp_dist_thre=3, use_gpu=False):
    """
    计算差异性奖励值和表示能力奖励值
    输入:
        seq: 特征序列形状为(1, seq_len, dim)
        actions: 二进制动作序列形状为(1, seq_len, 1)
        ignore_far_sim (bool): 是否考虑时序距离相似性（默认为True）
        temp_dist_thre (int): 阈值用于时域距离相似性（默认为20）
        use_gpu (bool): 是否使用GPU
    """
    _seq = seq.detach()          # 将seq变量从torch变量中分离，不参与参数更新
    _actions = actions  # 将actions变量从torch变量中分离，不参与参数更新
    pick_idxs = _actions                 # actions中非零元素的下坐标  #nonzero 返回非零元素下标
    num_picks = 1    # 选择的帧的数量
    # 没有帧被选中，返回的奖励为0
    if num_picks == 0:
        reward = torch.tensor(0.)
        if use_gpu: reward = reward.cuda()
        return reward
    _seq = _seq.squeeze()   # 从数组的形状中删除单维度条目，即把shape中为1的维度删掉
    n = _seq.size(0)        # 序列seq的长度
    # ------------------------------表示能力奖励值---------------------------------

    dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)    # 将平方和扩展为n×n的矩阵
    dist_mat = dist_mat + dist_mat.t()
    # dist_mat =dist_mat-2*dist_mat*dist_mat  # addmm_ ：inputs   = 1 × inputs - 2 ×（inputs @ inputs_t）
    dist_mat = dist_mat[:, pick_idxs]                                       # 距离矩阵
    dist_mat = dist_mat.min(0, keepdim=True)[0]                            # 最小距离kmedoid

    dist_mat = torch.pow(dist_mat, 1/5)
    reward_rep = torch.exp(-dist_mat.mean())                               # 最小距离的指数平方  奖励像素值大的点
    # 将两种奖励值结合
    reward = reward_rep
    # 返回当前actions对应的奖励值
    return reward





# 有目标攻击函数产的reward
def attackreward(res,iter_num,vid,adv_vid):
    # 获取平均扰动大小
    def pertubation(clean, adv):
        loss = torch.nn.L1Loss()
        average_pertubation = loss(clean, adv)
        return average_pertubation
    R = 0.0
    if (res):
        if (iter_num>15000):
            P = pertubation(vid,adv_vid)       # 平均每个像素的扰动量
            R = 0.999*torch.exp(-(P/0.05))     # 越小的扰动量对应越大的reward
            P = P.cpu()
        else:
            P = pertubation(vid,adv_vid)       # 平均每个像素的扰动量
            R = torch.exp(-(P/0.05))           # 越小的扰动量对应越大的reward   还有个query次数
    else:
        R = -1
    # 返回攻击奖励以及平均扰动量的大小
    return R