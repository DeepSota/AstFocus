import torch
import collections
from Adaselection.AST_S import AST_S
from Adaselection.AST_T import AST_T, feature_extractor
from Adaselection.Reward import intrinsicreward, attackreward, untargeted_reward, reward_advantage, \
    sparse_closeto_dense_reward, target_reward_advantage
from Adaselection.utils import agent_output, finelist, sparse_perturbation
from Adaselection.utils_ada import init_hidden
from Adaselection.Reward import rep_frame, div_frame
from Utils.utils import pertubation, random_speed_up
from Utils.edgebox import *
import Utils.edgebox as eb
import torch.nn.functional as F


# 二进制搜索，获得梯度方向最大步长的上界
def fine_grained_binary_search(vid_model, theta, initial_lbd,image_ori,targeted):
    lbd = initial_lbd
    while vid_model((image_ori + lbd * theta)[None,:])[1] != targeted:
        lbd *= 1.05
        if lbd > 1:
            return False, lbd
    num_intervals = 100
    lambdas = np.linspace(0.0, lbd.cpu(), num_intervals)[1:]
    lbd_hi = lbd
    lbd_hi_index = 0
    for i, lbd in enumerate(lambdas):
        if vid_model((image_ori + lbd * theta)[None,:])[1] == targeted:
            lbd_hi = lbd
            lbd_hi_index = i
            break
    lbd_lo = lambdas[lbd_hi_index - 1]
    while (lbd_hi - lbd_lo) > 1e-7:
        lbd_mid = (lbd_lo + lbd_hi) / 2.0
        if vid_model((image_ori + lbd_mid * theta)[None,:])[1] == targeted:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return True,lbd_hi




# 对于有目标攻击初始化对抗样本
def initialize_from_train_dataset_baseline(vid_model, image_ori, image_adv, targeted, MASK):
    theta = (image_adv - image_ori) * MASK      # 初始总共扰动量
    initial_lbd = torch.norm(theta)
    theta = theta / initial_lbd                 # 初始的扰动方向
    I,lbd = fine_grained_binary_search(vid_model, theta, initial_lbd,image_ori,targeted)
    theta, g2 = theta, lbd                      # 更新theta和g_theta,theta表示方向，g_theta表示移动步长
    new_image = image_ori + theta *lbd
    new_image = torch.clamp(new_image, 0., 1.)    # 裁剪
    return I, new_image



def nes_patch_frame(model, vid, patch_size, mask_list, actions_t, n, ori_class,  sub_num, sigma=1e-3):
    with torch.no_grad():
        len_frame = len(actions_t)
        grads = torch.zeros((len_frame, 3, patch_size, patch_size), device='cuda')  # Partitioned gradient initialization

        count_in = 0
        batch_loss = []  # loss per batch
        batch_noise = []  # noise per batch
        batch_idx = []  # category per batch
        assert n % sub_num == 0 and sub_num % 2 == 0
        for _ in range(n // sub_num):
            adv_vid_rs = vid.repeat((sub_num,) + (1,) * len(vid.size()))  # sample initialization
            noise_list = torch.randn((sub_num // 2,) + grads.size(), device='cuda') * sigma  # noise initialization
            all_noise = torch.cat([noise_list, -noise_list], 0)  # noise

            label = ori_class.flatten()


            mask_list_t = actions_t.view(-1)                       # 选出的帧序列号
            MASK = torch.zeros(adv_vid_rs.size())
            b = mask_list.numpy()                         # 选出的patch的位置 (16,4)

            c = b[mask_list_t]                            # 只用选出的帧里所含的patch
            i = 0
            try:
                for x1, x2, y1, y2 in c:
                    key = mask_list_t[i]                      # 存储着选出的帧 的序号
                    x1 = torch.tensor(x1).to(torch.int64)
                    x2 = torch.tensor(x2).to(torch.int64)
                    y1 = torch.tensor(y1).to(torch.int64)
                    y2 = torch.tensor(y2).to(torch.int64)
                    MASK[:, key, :, y1:y2, x1:x2] = all_noise[:, i, :, :, :]   # 按照选出的 帧索引 在对应的patch上添加噪声
                    i = i+1
            except:
                print('c', c)


            adv_vid_rs += MASK.cuda()  # add perturbations to input
            del MASK # release gpu resources

            top_val, top_idx, logits = model(adv_vid_rs)
            top_idx = top_idx.view(-1)
            top_val = top_val.view(-1)
            ground_preds = F.softmax(logits.detach(), 1).view(sub_num, -1)[:, label].flatten()
            hunxiao_probs = torch.zeros(sub_num).cuda()
            top2_val = F.softmax(logits.detach(), 1).view(sub_num, -1).sort(1)[0][:, -2]
            hunxiao_probs[top_idx == label] = top2_val[top_idx == label]
            hunxiao_probs[top_idx != label] = top_val[top_idx != label]

            loss = hunxiao_probs - ground_preds
            batch_loss.append(loss)
            batch_idx.append(top_idx)
            batch_noise.append(all_noise)

        # concat operations
        batch_noise = torch.cat(batch_noise, 0)
        batch_loss = torch.cat(batch_loss, 0)
        batch_idx = torch.cat(batch_idx)

        # sorting loss
        good_idx = torch.sum(batch_idx == ori_class, 1).byte()                               # valid sample
        changed_loss = torch.where(good_idx, batch_loss, torch.tensor(1000., device='cuda'))    # penalty
        loss_order = torch.zeros(changed_loss.size(0), device='cuda')              # 48个0 size（48,1）
        sort_index = changed_loss.sort()[1]                                        # sort loss to obtain coordinates
        loss_order[sort_index] = torch.arange(0, changed_loss.size(0), device='cuda', dtype=torch.float)  # changed_loss.size(0)=48
        # 就是 从小到大共48个数， 按0-47赋值
        available_number = torch.sum(good_idx).item()                              # the number of valid samples
        count_in += available_number                                               # accumulative count
        unavailable_number = n - available_number                                               # invalid number
        unavailable_weight = torch.sum(torch.where(good_idx, torch.tensor(0., device='cuda'),
                                                    loss_order)) / unavailable_number if unavailable_number else torch.tensor(
            0., device='cuda')
        rank_weight = torch.where(good_idx, loss_order, unavailable_weight) / (n - 1)           # weighted gradient
        grads += torch.sum(batch_noise / sigma * (rank_weight.view((-1,) + (1,) * (len(batch_noise.size()) - 1))),0)


        return grads

def nes_target(model, vid, patch_size, mask_list, actions_t, n, ori_class, target_class, sub_num, sigma=1e-3):
    with torch.no_grad():

        len_frame = len(actions_t)
        grads = torch.zeros((len_frame, 3, patch_size, patch_size), device='cuda')  # Partitioned gradient initialization
        loss_total =0
        count_in = 0
        batch_loss = []  # loss per batch
        batch_noise = []  # noise per batch
        batch_idx = []  # category per batch
        assert n % sub_num == 0 and sub_num % 2 == 0
        for _ in range(n // sub_num):
            adv_vid_rs = vid.repeat((sub_num,) + (1,) * len(vid.size()))  # sample initialization
            noise_list = torch.randn((sub_num // 2,) + grads.size(), device='cuda') * sigma  # noise initialization
            all_noise = torch.cat([noise_list, -noise_list], 0)  # noise

            label = ori_class.flatten()
            mask_list_t = actions_t.view(-1)                       # 选出的帧序列号
            MASK = torch.zeros(adv_vid_rs.size())
            b = mask_list.numpy()                         # 选出的patch的位置 (16,4)
            c = b[mask_list_t]                            # 只用选出的帧里所含的patch
            i = 0
            try:
                for x1, x2, y1, y2 in c:
                    key = mask_list_t[i]                      # 存储着选出的帧 的序号
                    x1 = torch.tensor(x1).to(torch.int64)
                    x2 = torch.tensor(x2).to(torch.int64)
                    y1 = torch.tensor(y1).to(torch.int64)
                    y2 = torch.tensor(y2).to(torch.int64)
                    MASK[:, key, :, y1:y2, x1:x2] = all_noise[:, i, :, :, :]   # 按照选出的 帧索引 在对应的patch上添加噪声
                    i = i+1
            except:
                print('c', c)


            adv_vid_rs += MASK.cuda()  # add perturbations to input
            del MASK # release gpu resources

            top_val, top_idx, logits = model(adv_vid_rs)
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor(target_class, dtype=torch.long,
                                                                          device='cuda').repeat(sub_num),
                                                     reduction='none')

            batch_loss.append(-loss)
            batch_noise.append(all_noise)

        # concat operations
        batch_noise = torch.cat(batch_noise, 0)
        batch_loss = torch.cat(batch_loss, 0)

        valid_loss = batch_loss                        # total loss
        loss_total += torch.mean(valid_loss).item()    # total loss mean 就是一个总的均值
        count_in += valid_loss.size(0)                 # count_in=48
        noise_select = batch_noise                     # noise
        grads += torch.sum(noise_select / sigma * (valid_loss.view((-1,) + (1,) * (len(noise_select.size()) - 1))),0) # sum 48个和一
        l = torch.mean(batch_loss)
        return l, grads

# 块空间采样
# untargeted attack gradient estimation (NES)
def nes_patch(model, vid, patch_size, mask_list,  n,  target, rank_transform, sub_num,   sigma=1e-3):
    # n is the sample number for NES, sub_num is also the sample number and used to
    # prevent GPU resource insufficiency if n is too large
    with torch.no_grad():
        grads = torch.zeros(16, 3, patch_size, patch_size, device='cuda')   # Partitioned gradient initialization
        count_in = 0                                         # record the effective number
        loss_total = 0                                       # loss per NES
        batch_loss = []                                      # loss per batch
        batch_noise = []                                     # noise per batch
        batch_idx = []                                       # category per batch
        assert n % sub_num == 0 and sub_num % 2 == 0
        for _ in range(n // sub_num):
            adv_vid_rs = vid.repeat((sub_num,) + (1,) * len(vid.size())).clone()             # sample initialization
            noise_list = torch.randn((sub_num // 2,) + grads.size(), device='cuda') * sigma   # noise initialization
            all_noise = torch.cat([noise_list, -noise_list], 0)                            # noise
            perturbation_sample = all_noise   # produce perturbations


            MASK = torch.zeros(adv_vid_rs.size())
            b = mask_list.numpy()
            i = 0
            for x1, x2, y1, y2 in b:
                MASK[:, i, :, y1:y2, x1:x2] = all_noise[:, i, :, :, :]  # tensor 要取里面的值
                i += 1



            label =target.flatten()

            adv_vid_rs += MASK.cuda()                                              # add perturbations to input
            del MASK                                                        # release gpu resources
            top_val, top_idx, logits = model(adv_vid_rs)                                    # classification results


            top_idx = top_idx.view(-1)
            top_val = top_val.view(-1)
            ground_preds = F.softmax(logits.detach(), 1).view(sub_num, -1)[:, label].flatten()
            hunxiao_probs = torch.zeros(sub_num).cuda()
            top2_val = F.softmax(logits.detach(), 1).view(sub_num, -1).sort(1)[0][:, -2]
            hunxiao_probs[top_idx == label] = top2_val[top_idx == label]
            hunxiao_probs[top_idx != label] = top_val[top_idx != label]


            loss = hunxiao_probs -ground_preds
            batch_loss.append(loss)
            batch_idx.append(top_idx)
            batch_noise.append(all_noise)
        # concat operations
        batch_noise = torch.cat(batch_noise, 0)
        batch_idx = torch.cat(batch_idx)
        batch_loss = torch.cat(batch_loss)
        # sorting loss
        if rank_transform:
            good_idx = torch.sum(batch_idx == target, 1).byte()                               # valid sample
            changed_loss = torch.where(good_idx, batch_loss, torch.tensor(1000., device='cuda'))    # penalty
            loss_order = torch.zeros(changed_loss.size(0), device='cuda')
            sort_index = changed_loss.sort()[1]                                        # sort loss to obtain coordinates
            loss_order[sort_index] = torch.arange(0, changed_loss.size(0), device='cuda', dtype=torch.float)  # changed_loss.size(0)=48
            # 就是 从小到大共48个数， 按0-47赋值
            available_number = torch.sum(good_idx).item()                              # the number of valid samples
            count_in += available_number                                               # accumulative count
            unavailable_number = n - available_number                                               # invalid number
            unavailable_weight = torch.sum(torch.where(good_idx, torch.tensor(0., device='cuda'),
                                                        loss_order)) / unavailable_number if unavailable_number else torch.tensor(
                0., device='cuda')
            rank_weight = torch.where(good_idx, loss_order, unavailable_weight) / (n - 1)           # weighted gradient
            grads += torch.sum(batch_noise / sigma * (rank_weight.view((-1,) + (1,) * (len(batch_noise.size()) - 1))),0)
        else:
            idxs = (batch_idx == target).nonzero()   # valid samples
            valid_idxs = idxs[:, 0]                        # coordinates
            valid_loss = torch.index_select(batch_loss, 0, valid_idxs)     # valid loss
            loss_total += torch.mean(valid_loss).item()    # average loss
            count_in += valid_loss.size(0)                 # count
            noise_select = torch.index_select(batch_noise, 0, valid_idxs)  # valid noise
            grads += torch.sum(noise_select / sigma * (valid_loss.view((-1,) + (1,) * (len(noise_select.size()) - 1))),0)
        if count_in == 0:
            return None, None
        # return estimated gradient and loss
        return loss_total / count_in, grads

# 全空间采样
def sim_rectification_vector(model, vid,  n,  target_class,
                             rank_transform, sub_num, sigma=1e-3):
    # n is the sample number for NES, sub_num is also the sample number and used to
    # prevent GPU resource insufficiency if n is too large
    with torch.no_grad():
        grads = torch.zeros(vid.size(), device='cuda')   # Partitioned gradient initialization
        count_in = 0                                         # record the effective number
        loss_total = 0                                       # loss per NES
        batch_loss = []                                      # loss per batch
        batch_noise = []                                     # noise per batch
        batch_idx = []                                       # category per batch
        assert n % sub_num == 0 and sub_num % 2 == 0
        for _ in range(n // sub_num):
            adv_vid_rs = vid.repeat((sub_num,) + (1,) * len(vid.size()))             # sample initialization
            noise_list = torch.randn((sub_num // 2,) + grads.size(), device='cuda') * sigma   # noise initialization
            all_noise = torch.cat([noise_list, -noise_list], 0)                            # noise
            perturbation_sample = all_noise   # produce perturbations

            adv_vid_rs += perturbation_sample                                               # add perturbations to input
            del perturbation_sample                                                         # release gpu resources
            top_val, top_idx, logits = model(adv_vid_rs)                                    # classification results



            label = target_class.flatten()
            top_idx = top_idx.view(-1)
            top_val = top_val.view(-1)
            ground_preds = F.softmax(logits.detach(), 1).view(sub_num, -1)[:, label].flatten()
            hunxiao_probs = torch.zeros(sub_num).cuda()
            top2_val = F.softmax(logits.detach(), 1).view(sub_num, -1).sort(1)[0][:, -2]
            hunxiao_probs[top_idx == label] = top2_val[top_idx == label]
            hunxiao_probs[top_idx != label] = top_val[top_idx != label]


            loss = hunxiao_probs -ground_preds
            batch_loss.append(loss)
            batch_idx.append(top_idx)
            batch_noise.append(all_noise)
        # concat operations
        batch_noise = torch.cat(batch_noise, 0)
        batch_idx = torch.cat(batch_idx)
        batch_loss = torch.cat(batch_loss)
        # sorting loss
        if rank_transform:
            good_idx = torch.sum(batch_idx == target_class, 1).byte()                               # valid sample
            changed_loss = torch.where(good_idx, batch_loss, torch.tensor(1000., device='cuda'))    # penalty
            loss_order = torch.zeros(changed_loss.size(0), device='cuda')              # 48个0 size（48,1）
            sort_index = changed_loss.sort()[1]                                        # sort loss to obtain coordinates
            loss_order[sort_index] = torch.arange(0, changed_loss.size(0), device='cuda', dtype=torch.float)  # changed_loss.size(0)=48
            # 就是 从小到大共48个数， 按0-47赋值
            available_number = torch.sum(good_idx).item()                              # the number of valid samples
            count_in += available_number                                               # accumulative count
            unavailable_number = n - available_number                                               # invalid number

            unavailable_weight = torch.sum(torch.where(good_idx, torch.tensor(0., device='cuda'),
                                                        loss_order)) / unavailable_number if unavailable_number else torch.tensor(
                0., device='cuda')

            # unavailable_weight = torch.sum(torch.where(good_idx, torch.tensor(0., device='cuda'),
            #                                             loss_order)) if unavailable_number else torch.tensor(
            #     0., device='cuda')

            rank_weight = torch.where(good_idx, loss_order, unavailable_weight) / (n - 1)           # weighted gradient
            grads += torch.sum(batch_noise / sigma * (rank_weight.view((-1,) + (1,) * (len(batch_noise.size()) - 1))),0)
        else:
            idxs = (batch_idx == target_class).nonzero()   # valid samples
            valid_idxs = idxs[:, 0]                        # coordinates
            valid_loss = torch.index_select(batch_loss, 0, valid_idxs)     # valid loss
            loss_total += torch.mean(valid_loss).item()    # average loss
            count_in += valid_loss.size(0)                 # count
            noise_select = torch.index_select(batch_noise, 0, valid_idxs)  # valid noise
            grads += torch.sum(noise_select / sigma * (valid_loss.view((-1,) + (1,) * (len(noise_select.size()) - 1))),0)
        if count_in == 0:
            return None, None
        # return estimated gradient and loss
        return loss_total / count_in, grads


# 输入的视频应该是tensor，大小格式为[num_frames, c, w, h]. 输入应该归一化到[0, 1]
def untargeted_video_attack(vid_model, vid, x0, ori_class, args,  eps=0.05,
                        max_lr=0.01, min_lr=0.001, effect_num=6, sample_per_draw=60,  sub_num_sample=2):

    # ----------------------------------------------------初始化---------------------------------------------------------
    max_iter = args.max_iter
    num_iter = 0
    last_p = []
    epis_reward_epochs = []
    baselines_s = 0

    im = x0[7, :, :, :].permute(1, 2, 0).numpy().astype(np.float32)
    fd = eb.image_2_foundation(im)
    nes_num = 0
    len_limit = 8
    len_up_band = 10
    AST_t = AST_T(args)
    AST_s = AST_S(args)
    GetFeatures = AST_t.extractor  # 特征提取器
    use_adap_lr = False
    reinforce_power = False
    if reinforce_power == False:
       max_lr =0.03
       min_lr =2.4e-3
       use_adap_lr = True
    cur_lr = max_lr
    getfeatures = feature_extractor()  # 特征提取器
    len_frame = vid.size(0)


    try:
        AST_t.cuda()
    except:
        pass

    adv_vid = vid
    top_val, top_idx, logits = vid_model(adv_vid[None, :])
    pre_confidence = top_val.view(-1).cpu()
    pre_hunxiao_confidence = (F.softmax(logits.detach(), 1)).view(-1).sort()[0][-2].cpu()
    num_iter += 1

    if args.model_name == 'lrcn' :
        perturbation = (torch.rand_like(vid) * 2 - 1) * eps                                # initial perturbations
        key_list = random.sample(range(0, 16), len_limit)               # generate a len_limit length frame sequence
        perturbation = sparse_perturbation(perturbation, key_list)
        adv_vid = torch.clamp(vid.clone() + perturbation, 0., 1.)
    while num_iter < max_iter:
        # ---------------------------------------------------select-----------------------------------------------------
        _, features = GetFeatures(vid)
        features = features.squeeze()
        global_fea = getfeatures(vid)
        batch_size = 1
        h_s = init_hidden(batch_size, 512)
        c_s = init_hidden(batch_size, 512)
        hx_s = (h_s, c_s)
        step = 0
        feature_pre_s = torch.ones(1, 1280)
        MASK_s = torch.zeros(vid.size())
        key_s_lists = []

        # patch
        for focus_time_step in range(vid.size(0)):
            img = vid[focus_time_step, :, :, :]
            feature = features[focus_time_step, :].clone()
            key_s, HX_s = AST_s.one_step_act(img, feature, feature_pre_s, global_fea, hx_s, re_sample=False, training=True)
            ks = key_s.cpu().numpy()
            hx_s = HX_s
            for x1, x2, y1, y2 in ks:
                MASK_s[focus_time_step, :, y1:y2, x1:x2] = 1
            feature_s = vid[focus_time_step, : ,:,:].clone() * (MASK_s[focus_time_step,:,:,:].clone().cuda())
            _, feature_pre_s = GetFeatures(feature_s.view(1, feature_s.size(0), feature_s.size(1), feature_s.size(2)))
            feature_pre_s = feature_pre_s.squeeze()
            obj_score = eb.get_objectness(fd, x1, y1, x2, y2)[0]
            reward_step = obj_score
            AST_s.focuser.memory.rewards.append(reward_step)

            key_s_lists.append(key_s)
            step += 1
        mask_s = torch.cat(key_s_lists, 0).view(16, -1).cpu()

        step = 0
        NONRE =True

        # frame
        while True:
            AST_t.focuser.memory.clear_memory()
            h_t = init_hidden(batch_size, 512)
            c_t = init_hidden(batch_size, 512)
            hx = (h_t, c_t)
            feature_pre = torch.ones(1, 1280)
            key_t_lists = []

            for focus_time_step in range(vid.size(0)):
                feature = features[focus_time_step, :].clone()
                key_t, HX = AST_t.one_step_act(feature, feature_pre, global_fea, hx, re_sample=False, training=True)
                hx = HX
                if key_t:
                   feature_pre = feature.clone()
                   R_rep = rep_frame(features, focus_time_step).cpu()
                else:
                   feature_pre = torch.ones(1, 1280)
                   R_rep = 0


                AST_t.focuser.memory.rewards.append(R_rep)
                key_t_lists.append(key_t)

            actions_t = torch.cat(key_t_lists, 0).cpu().nonzero()
            ll = len(actions_t)
            step += 1
            if  ll >= 2 and ll <= len_up_band :
                break
            if step > 10:
                actions_t, key_t = random_speed_up(len_frame, len_limit)
                NONRE = False
                break



        if NONRE:
           mask_t = torch.cat(key_t_lists, 0).view(-1, 1).cpu()
        else:
           mask_t = key_t.view(-1, 1).cpu()
        mask = mask_s * mask_t


        # ---------------------------------------------------nes--------------------------------------------------------
        # nes梯度估计
        patch_size = args.patch_size
        # patch+frame
        gs = nes_patch_frame(vid_model, adv_vid, patch_size, mask, actions_t,  sample_per_draw, ori_class, sub_num_sample, sigma=1e-3)

        num_iter += sample_per_draw        # 累计查询次数
        nes_num += 1


        # ---------------------------------------------------噪声攻击-----------------------------------------------------
        proposed_adv_vid = adv_vid
        if reinforce_power:
           g = torch.sign(gs)
        else:
           g = torch.tensor(gs)


        # 帧块空间采样
        mask_list_t = actions_t.view(-1)
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
            MASK[key, :, y1:y2, x1:x2] = g[i, :, :, :]
            i = i + 1

        proposed_adv_vid += cur_lr * MASK.cuda()

        # 生成样本的裁剪
        bottom_bounded_adv = torch.where((vid - eps) > proposed_adv_vid, vid - eps,
                                         proposed_adv_vid)
        bounded_adv = torch.where((vid + eps) < bottom_bounded_adv, vid + eps, bottom_bounded_adv)
        clip_frame = torch.clamp(bounded_adv, 0., 1.)
        adv_vid = clip_frame.clone()
        top_val, top_idx, _ = vid_model(adv_vid[None, :])  # 模型预测结果
        num_iter += 1  # 迭代次数加一
        if ori_class != top_idx[0][0]:
            return True, num_iter, adv_vid, nes_num

        # ---------------------------------------------------agent更新---------------------------------------------------
        # patch_agent更新
        reward_epoch0, adv_confidence, hunxiao_confidence, r_attack = reward_advantage(vid_model, adv_vid, ori_class, pre_confidence, pre_hunxiao_confidence)
        pre_confidence = torch.tensor(adv_confidence).cpu()
        pre_hunxiao_confidence = torch.tensor(hunxiao_confidence).cpu()
        epis_reward_epochs.append(reward_epoch0.numpy())
        baselines_s = 0.9 * baselines_s + 0.1 * np.mean(epis_reward_epochs)
        reward_epoch = reward_epoch0 - baselines_s

        for cc in range(0, 16):
            obs_share_old = torch.cat([AST_t.focuser.memory.states_t_old[cc], AST_s.focuser.memory.states_s_old[cc]],1)
            AST_s.focuser.memory.obs_share_old.append(obs_share_old)
            AST_t.focuser.memory.obs_share_old.append(obs_share_old)

        T = 8
        lamda = 0.8
        R_frame = ((len(actions_t) - T) / 16) ** 2
        reward_epoch_t = (1-lamda)*reward_epoch-lamda*R_frame

        AST_s.focuser.memory.reward_epochs.append(torch.tensor(reward_epoch).cuda())
        AST_t.focuser.memory.reward_epochs.append(torch.tensor(reward_epoch_t).cuda())  # 时间上的 reward 不加

        obs_share_s = torch.stack(AST_s.focuser.memory.states_s_old, 0).cuda().squeeze().clone()
        obs_share_t = torch.stack(AST_t.focuser.memory.states_t_old, 0).cuda().squeeze().clone()


        fineS = 0.2
        fineT = 0.8
        AST_s.focuser.update(vid, features,   global_fea,  obs_share_t, GetFeatures, fineS)
        AST_t.focuser.update(vid, features,   global_fea,  obs_share_s, fineT)

        # --------------------------------------------------动态调整学习率-------------------------------------------------
        # 退火最大学习率
        if use_adap_lr:
            last_p.append(float(top_val))  # 记录预测概率值
            last_p = last_p[-20:]  # 取最新20个概率预测
            if last_p[-1] <= last_p[0] and len(last_p) == effect_num:
                # 如果有攻击效果，则镜像退火操作
                if cur_lr > min_lr:
                    # print("[log] Annealing max_lr")
                    cur_lr = max(cur_lr / 2., min_lr)
                last_p = []
            del top_val
            del top_idx
    return False, 0, adv_vid, nes_num


def targeted_video_attack(XX, vid_model, vid,   x0, ori_class, target_class, args,  starting_eps=1., eps=0.05,
                        max_lr = 0.03, min_lr=0.0024, sample_per_draw=60,  sub_num_sample=2, max_iter=30000):

    # ----------------------------------------------------初始化---------------------------------------------------------
    print(XX)
    ori_class = ori_class.view(-1)
    target_class = target_class.view(-1)
    num_iter = 0
    cur_lr = max_lr
    epis_reward_epochs = []
    baselines_s = 0
    len_frame= 16
    im = x0[7, :, :, :].permute(1, 2, 0).numpy().astype(np.float32)
    fd = eb.image_2_foundation(im)
    nes_num = 0
    len_limit = 8
    alpha = len_limit / 16.0
    target_prob_list =[]
    last_loss = []
    AST_t = AST_T(args)
    AST_s = AST_S(args)
    GetFeatures = AST_t.extractor
    getfeatures = feature_extractor()
    model_name = args.model_name
    dataset_name = args.dataset_name
    try:
        AST_t.cuda()
    except:
        pass

    cur_eps = starting_eps
    top_val, top_idx, logits = vid_model(vid[None, :])
    pre_confidence = top_val.view(-1).cpu()
    pre_confidence1 = pre_confidence.clone()
    pre_hunxiao_confidence = (F.softmax(logits.detach(), 1)).view(-1)[target_class].cpu()
    adv_vid = vid.clone()
    pre_g = 0.0

    while num_iter < max_iter:
        # ---------------------------------------------------select-----------------------------------------------------


        _, features = GetFeatures(vid[:, :, :, :])
        features = features.squeeze()
        global_fea = getfeatures(vid)

        batch_size = 1
        h_s = init_hidden(batch_size, 512)
        c_s = init_hidden(batch_size, 512)
        hx_s = (h_s, c_s)
        step = 0
        feature_pre_s = torch.ones(1, 1280)
        MASK_s = torch.zeros(vid.size())
        key_s_lists = []

        # patch
        for focus_time_step in range(vid.size(0)):
            img = vid[focus_time_step, :, :, :]
            feature = features[focus_time_step, :].clone()
            key_s, HX_s = AST_s.one_step_act(img, feature, feature_pre_s, global_fea, hx_s, re_sample=False, training=True)
            ks = key_s.cpu().numpy()
            hx_s = HX_s
            for x1, x2, y1, y2 in ks:
                MASK_s[focus_time_step, :, y1:y2, x1:x2] = 1
            feature_s = vid[focus_time_step, : ,:,:].clone() * (MASK_s[focus_time_step,:,:,:].clone().cuda())
            _, feature_pre_s = GetFeatures(feature_s.view(1, feature_s.size(0), feature_s.size(1), feature_s.size(2)))
            feature_pre_s = feature_pre_s.squeeze()
            obj_score = eb.get_objectness(fd, x1, y1, x2, y2)[0]
            reward_step = obj_score
            AST_s.focuser.memory.rewards.append(reward_step)

            key_s_lists.append(key_s)
            step += 1
        mask_s = torch.cat(key_s_lists, 0).view(16, -1).cpu()

        step = 0
        NONRE =True

        # frame
        while True:
            AST_t.focuser.memory.clear_memory()
            h_t = init_hidden(batch_size, 512)
            c_t = init_hidden(batch_size, 512)
            hx = (h_t, c_t)
            feature_pre = torch.ones(1, 1280)
            MASK_t = torch.zeros(vid.size())
            key_t_lists = []

            for focus_time_step in range(vid.size(0)):
                feature = features[focus_time_step, :].clone()
                key_t, HX = AST_t.one_step_act(feature, feature_pre, global_fea, hx, re_sample=False, training=True)
                kt = key_t.cpu().numpy()
                hx = HX
                # MASK = torch.zeros(vid.size())
                if key_t:
                   feature_pre = feature.clone()
                   R_rep = rep_frame(features, focus_time_step).cpu()
                else:
                   feature_pre = torch.ones(1, 1280)
                   R_rep = 0
                AST_t.focuser.memory.rewards.append(R_rep)
                key_t_lists.append(key_t)

            actions_t = torch.cat(key_t_lists, 0).cpu().nonzero()
            ll = len(actions_t)
            step += 1
            if  ll >= 2:
                break
            if step > 10:
                actions_t, key_t = random_speed_up(len_frame, len_limit)
                NONRE = False
                break
        if NONRE:
           mask_t = torch.cat(key_t_lists, 0).view(-1, 1).cpu()
        else:
           mask_t = key_t.view(-1, 1).cpu()
        mask = mask_s * mask_t


        # ---------------------------------------------------nes--------------------------------------------------------
        # nes梯度估计
        patch_size = args.patch_size

        # patch+frame
        l, g = nes_target(vid_model, adv_vid, patch_size, mask, actions_t,  sample_per_draw, top_idx, target_class, sub_num_sample, sigma=1e-3)

        num_iter += sample_per_draw        # 累计查询次数
        nes_num += 1
        if l is None and g is None:
            print('nes sim fails, try again....')
            continue

        last_loss.append(l)                                # record loss
        last_loss = last_loss[-5:]                         # Take the latest five loss value
        if last_loss[-1] > last_loss[0]:
            if cur_lr > min_lr:
                cur_lr = max(cur_lr/2.,min_lr)
            last_loss = []

        # ---------------------------------------------------噪声攻击-----------------------------------------------------
        g = torch.tensor(g)
        # 帧块空间采样
        mask_list_t = actions_t.view(-1)
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
            MASK[key, :, y1:y2, x1:x2] = g[i, :, :, :]
            i = i + 1
        g = MASK.clone()
        g = 0.1*pre_g + 0.9*g
        pre_g = g

        proposed_adv_vid = adv_vid.clone()
        proposed_adv_vid += cur_lr * MASK.sign().cuda()

        # 生成样本的裁剪
        bottom_bounded_adv = torch.where((vid - eps) > proposed_adv_vid, vid - eps,
                                         proposed_adv_vid)
        bounded_adv = torch.where((vid + eps) < bottom_bounded_adv, vid + eps, bottom_bounded_adv)
        adv_vid= torch.clamp( bounded_adv, 0., 1.)


        num_iter += 1
        top_val, top_idx, logi = vid_model(adv_vid[None, :])
        target_prob = (F.softmax(logi.detach(), 1)).view(-1)[target_class].cpu()
        AP = pertubation(vid, adv_vid)
        print('top_idx: {}, top_val: {}, target_prob: {}'.format(top_idx.item(), top_val.item(), target_prob.item()))

        target_prob_list.append(target_prob)
        target_prob_list = target_prob_list[-5:]
        # if target_prob_list[-1] >=target_prob_list[0]:
        #     min_lr = min_lr/2

        if target_class == top_idx[0][0]:
            # 提前终止
            print('early stop at iterartion {}'.format(num_iter))
            return True, num_iter, adv_vid
        # ---------------------------------------------------agent更新---------------------------------------------------
        reward_epoch0, adv_confidence, hunxiao_confidence, r_attack = target_reward_advantage(vid_model, adv_vid, ori_class,target_class,pre_confidence, pre_hunxiao_confidence)
        pre_confidence = torch.tensor(adv_confidence).cpu()
        pre_hunxiao_confidence = torch.tensor(hunxiao_confidence).cpu()
        epis_reward_epochs.append(reward_epoch0.numpy())
        baselines_s = 0.9 * baselines_s + 0.1 * np.mean(epis_reward_epochs)
        reward_epoch = reward_epoch0 - baselines_s

        for cc in range(0, 16):
            obs_share_old = torch.cat([AST_t.focuser.memory.states_t_old[cc], AST_s.focuser.memory.states_s_old[cc]],1)
            AST_s.focuser.memory.obs_share_old.append(obs_share_old)
            AST_t.focuser.memory.obs_share_old.append(obs_share_old)

        T = 8
        lamda = 0.8
        R_frame = ((len(actions_t) - T) / 16) ** 2
        reward_epoch_t = (1-lamda)*reward_epoch-lamda*R_frame

        AST_s.focuser.memory.reward_epochs.append(torch.tensor(reward_epoch).cuda())
        AST_t.focuser.memory.reward_epochs.append(torch.tensor(reward_epoch_t).cuda())  # 时间上的 reward 不加

        obs_share_s = torch.stack(AST_s.focuser.memory.states_s_old, 0).cuda().squeeze().clone()
        obs_share_t = torch.stack(AST_t.focuser.memory.states_t_old, 0).cuda().squeeze().clone()


        fineS = 0.2
        fineT = 0.8
        AST_s.focuser.update(vid, features,   global_fea,  obs_share_t, GetFeatures, fineS)
        AST_t.focuser.update(vid, features,   global_fea,  obs_share_s, fineT)
    adv_vid = torch.max(torch.min(adv_vid, vid + 0.02), vid - 0.02)
    return False, cur_eps, adv_vid




class Tattack_base(object):
    def __init__(self, model,   vid, tar_vid, ori_label, target_label, model_name,
                 dataset_name, del_frame=True, bound=True, bound_threshold=3):

        self.model = model                    # 预训练的模型
        self.x0 = vid                         # 攻击的图片
        self.xi = tar_vid.cuda()
        self.y0 = ori_label                          # 攻击的label
        self.targeted = target_label.view(-1,1)              # 有目标攻击标识位
        self.model_name = model_name          # 模型名称
        self.dataset_name = dataset_name      # 数据集名称
        # 稀疏性设置参数
        self.del_frame = del_frame
        self.bound = bound
        self.bound_threshold = bound_threshold
        self.samples = 1
        self.query_counts = 0                 # quey次数
        self.opt_counts = 0                   # opt次数
        self.initialize_paras()
        self.initial_lr= 0.02
        self.idx = 50

    # 部分参数的初始化
    def initialize_paras(self):
        if self.model_name == 'c3d':
            # batch_size, num_channels, seq_len, height, width
            self.seq_len = self.x0.size()[0]      # 序列长度
            self.seq_axis = 1                     # 序列长度对应的维度
        elif self.model_name == 'lrcn':
            # batch_size, seq_len, height, width, num_channels
            self.seq_len = self.x0.size()[0]
            self.seq_axis = 0
        self.image_ori = self.x0.clone()  # 原始图片
        self.MASK = torch.ones(self.x0.size()).cuda()
        self.best_MASK = torch.ones(self.x0.size()).cuda()

    def classify(self, inp):
        if inp.shape[0] != 1:
            inp = torch.unsqueeze(inp, 0)
        # inp = inp.permute(2, 0, 1, 3, 4)
        # logits = self.model(inp)
        # logits = torch.mean(logits, dim=1)
        # values, indices = torch.sort(-torch.nn.functional.softmax(logits), dim=1)
        # confidence_prob, pre_label = -float(values[:, 0]), int(indices[:, 0])
        confidence_prob, pre_label, _ =self.model(inp)

        self.query_counts += 1    # 梯度估计时查询次数
        return confidence_prob, pre_label

    def fine_grained_binary_search(self, theta, initial_lbd, this_mask):
        lbd = initial_lbd

        while self.classify(self.image_ori + lbd * this_mask * theta)[1] != self.targeted:
            lbd *= 1.05
            if lbd > 300:
                return float('inf')
        num_intervals = 100
        lambdas = np.linspace(0.0, lbd.cpu(), num_intervals)[1:]
        lbd_hi = lbd
        lbd_hi_index = 0
        for i, lbd in enumerate(lambdas):
            if self.classify(self.image_ori + lbd * this_mask * theta)[1] == self.targeted:
                lbd_hi = lbd
                lbd_hi_index = i
                break
        lbd_lo = lambdas[lbd_hi_index - 1]
        while (lbd_hi - lbd_lo) > 1e-7:
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            if self.classify(self.image_ori + lbd_mid * this_mask * theta)[1] == self.targeted:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid

        return lbd_hi



    def cal_confi(self, vid_model, image_ori, lbd_hi, theta, ori_class, taget_class):
        top_val, top_idx, logits = vid_model((image_ori + lbd_hi * theta)[None, :])
        ori_confi = F.softmax(logits, 1)[ori_class]
        return ori_confi



    def get_bounding_value(self, frame_indices, noise_vector, image_adv):
        '''平均扰动要与tmp_p进行对比，如果小于，减少帧，如果大于，减少扰动
            知道该初始方向上的最佳对抗样本。'''
        bound_mask = self.frames_to_mask(frame_indices)
        tmp_vector = noise_vector * bound_mask + self.x0
        this_prob, this_pre = self.classify(tmp_vector)
        if this_pre == self.targeted:
            theta = (image_adv-self.image_ori) * bound_mask
            initial_lbd = torch.norm(theta)
            theta = theta /initial_lbd
            lbd = self.fine_grained_binary_search(theta, initial_lbd, bound_mask)
            if lbd == float('inf'):
                return None
            tmp_image_noise = theta * lbd* bound_mask
            all_nums = int(torch.sum(bound_mask.reshape(-1)).item())
            valid_indices = torch.argsort(-bound_mask.reshape(-1))[:all_nums]
            tmp_p = torch.mean(torch.abs(tmp_image_noise.reshape(-1)[valid_indices]))
            return (frame_indices, initial_lbd, lbd, theta,tmp_p, this_pre)
        else:
            return None

    def initialize_from_train_dataset_del_frame_bound(self):
        outer_best_p = float('inf')
        best_theta, g_theta = None, float('inf')
        self.best_MASK = None
        self.best_frame_indices = None
        for i in range(0, 1):

            noise_vector = self.xi - self.x0
            vector_adv = noise_vector * self.MASK + self.x0
            this_prob, this_pre = self.classify(vector_adv)
            if this_pre == self.targeted:
                image_adv = vector_adv.clone()
                del_frame_sequences = self.loop_del_frame_sort_sequence(noise_vector)
                print('this del_frame_sequence', del_frame_sequences)
                if not del_frame_sequences:
                    continue
                begin_frames = [k for k in range(self.seq_len)]
                re = self.get_bounding_value(begin_frames, noise_vector, image_adv)
                if not re:
                    continue
                frame_indices, initial_lbd, lbd, theta, tmp_p, this_pre = re
                # 定义变量进行筛选
                inner_frames = frame_indices
                inner_p = tmp_p
                inner_lbd = lbd
                inner_theta = theta

                for del_frame in del_frame_sequences:
                    print('del frame, cycle', del_frame)
                    tmp_frames = [i for i in inner_frames if i != del_frame]
                    re = self.get_bounding_value(tmp_frames, noise_vector, image_adv)
                    if re:
                        frame_indices, initial_lbd, lbd, theta, tmp_p, this_pre = re
                        # 如果大于bound_threshold，那么按照减少扰动的方向移动
                        if inner_p >= self.bound_threshold:
                            if tmp_p < inner_p:
                                inner_frames = tmp_frames
                                inner_p = tmp_p
                                inner_lbd = lbd
                                inner_theta = theta
                        # 如果小于bound_threshod，那么按照较少帧数的方向移动
                        else:
                            if len(tmp_frames) < len(inner_frames):
                                inner_frames = tmp_frames
                                inner_p = tmp_p
                                inner_lbd = lbd
                                inner_theta = theta
                    else:
                        continue
                if inner_p < outer_best_p:
                    best_theta, g_theta = inner_theta, inner_lbd
                    outer_best_p = inner_p
                    self.best_frame_indices = inner_frames
                    best_mask = self.frames_to_mask(inner_frames)
                    self.adv_vid_ini = self.image_ori + g_theta * best_theta * best_mask
        self.best_MASK = best_mask
        self.MASK = best_mask
        self.best_theta, self.g_theta = best_theta, g_theta
        self.theta, self.g2 = best_theta, g_theta
        return self.adv_vid_ini
    def loop_del_frame_sort_sequence(self, noise_vector, mode='target'):
            all_frames = [i for i in range(16)]
            score_dict = {}
            for i in all_frames:
                tmp_frames = [_ for _ in all_frames if _!=i]
                tmp_mask = self.frames_to_mask(tmp_frames)
                tmp_vector_adv = noise_vector * tmp_mask + self.x0
                this_prob, this_pre = self.classify(tmp_vector_adv)
                if mode == 'target':
                    if this_pre != self.targeted:
                        pass
                    else:
                        score_dict[i] = this_prob
                elif mode == 'untarget':
                    if this_pre != self.y0:
                        score_dict[i] = this_prob
                    else:
                        pass
            if score_dict:
                sorted_items = sorted(score_dict.items(), key=lambda x:-x[1])
                sorted_frames = []
                for item in sorted_items:
                    sorted_frames.append(item[0])
                return sorted_frames
            else:
                return None

    # 获取视频帧对应的MASK
    def frames_to_mask(self, frame_indices):
        mask = torch.zeros(self.x0.size())
        if self.model_name == 'c3d':
            mask[frame_indices, :, :, :] = 1
        else:
            mask[frame_indices, :, :, :] = 1
        return mask.cuda() * self.MASK