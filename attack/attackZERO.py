import torch
from ASMARTselection.AST_S import  AST_S
from ASMARTselection.AST_T import  AST_T
from ASMARTselection.Reward import reward_advantage, \
    target_reward_advantage, rep_frame, sparse_reward
from ASMARTselection.utils_ada import init_hidden

from ASMARTselection.edgebox import *
from ASMARTselection import edgebox as eb
import torch.nn.functional as F


from Utils.utils import speed_up_process, process_grad


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
    return True, lbd_hi



def initialize_from_train_dataset_baseline(vid_model, image_ori, image_adv, targeted, MASK):
    theta = (image_adv - image_ori) * MASK
    initial_lbd = torch.norm(theta)
    theta = theta / initial_lbd
    I,lbd = fine_grained_binary_search(vid_model, theta, initial_lbd,image_ori,targeted)
    theta, g2 = theta, lbd
    new_image = image_ori + theta *lbd
    new_image = torch.clamp(new_image, 0., 1.)
    return I, new_image


def nes_patch_frame(model, vid, patch_size, mask_list, actions_t, n, target, sub_num, sigma=1e-3):
    with torch.no_grad():

        len_frame = len(actions_t)
        grads = torch.zeros((len_frame, 3, patch_size, patch_size), device='cuda')

        count_in = 0
        batch_loss = []
        batch_noise = []
        batch_idx = []
        assert n % sub_num == 0 and sub_num % 2 == 0
        for _ in range(n // sub_num):
            adv_vid_rs = vid.repeat((sub_num,) + (1,) * len(vid.size()))
            noise_list = torch.randn((sub_num // 2,) + grads.size(), device='cuda') * sigma
            all_noise = torch.cat([noise_list, -noise_list], 0)

            label = target.flatten()


            mask_list_t = actions_t.view(-1)
            MASK = torch.zeros(adv_vid_rs.size())
            b = mask_list.numpy()
            c = b[mask_list_t]
            i = 0
            try:
                for x1, x2, y1, y2 in c:
                    key = mask_list_t[i]
                    x1 = torch.tensor(x1).to(torch.int64)
                    x2 = torch.tensor(x2).to(torch.int64)
                    y1 = torch.tensor(y1).to(torch.int64)
                    y2 = torch.tensor(y2).to(torch.int64)
                    MASK[:, key, :, y1:y2, x1:x2] = all_noise[:, i, :, :, :]
                    i = i+1
            except:
                print('c', c)


            adv_vid_rs += MASK.cuda()
            del MASK

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


        batch_noise = torch.cat(batch_noise, 0)
        batch_loss = torch.cat(batch_loss, 0)
        batch_idx = torch.cat(batch_idx)


        good_idx = torch.sum(batch_idx == target, 1).byte()
        changed_loss = torch.where(good_idx, batch_loss, torch.tensor(1000., device='cuda'))
        loss_order = torch.zeros(changed_loss.size(0), device='cuda')
        sort_index = changed_loss.sort()[1]
        loss_order[sort_index] = torch.arange(0, changed_loss.size(0), device='cuda', dtype=torch.float)
        available_number = torch.sum(good_idx).item()
        count_in += available_number
        unavailable_number = n - available_number
        unavailable_weight = torch.sum(torch.where(good_idx, torch.tensor(0., device='cuda'),
                                                    loss_order)) / unavailable_number if unavailable_number else torch.tensor(
            0., device='cuda')
        rank_weight = torch.where(good_idx, loss_order, unavailable_weight) / (n - 1)
        grads += torch.sum(batch_noise / sigma * (rank_weight.view((-1,) + (1,) * (len(batch_noise.size()) - 1))),0)

        return grads



def nes_target(model, vid, patch_size, mask_list, actions_t, n, ori_class, target_class, sub_num, w1, sigma=1e-6):
    with torch.no_grad():

        len_frame = len(actions_t)
        grads = torch.zeros((len_frame, 3, patch_size, patch_size), device='cuda')
        loss_total = 0
        count_in = 0
        batch_loss = []
        batch_noise = []
        batch_idx = []
        assert n % sub_num == 0 and sub_num % 2 == 0
        for _ in range(n // sub_num):
            adv_vid_rs = vid.repeat((sub_num,) + (1,) * len(vid.size()))
            noise_list = torch.randn((sub_num // 2,) + grads.size(), device='cuda') * sigma
            all_noise = torch.cat([noise_list, -noise_list], 0)

            label = ori_class.flatten()
            mask_list_t = actions_t.view(-1)
            MASK = torch.zeros(adv_vid_rs.size())
            b = mask_list.numpy()
            c = b[mask_list_t]
            i = 0
            try:
                for x1, x2, y1, y2 in c:
                    key = mask_list_t[i]
                    x1 = torch.tensor(x1).to(torch.int64)
                    x2 = torch.tensor(x2).to(torch.int64)
                    y1 = torch.tensor(y1).to(torch.int64)
                    y2 = torch.tensor(y2).to(torch.int64)
                    MASK[:, key, :, y1:y2, x1:x2] = all_noise[:, i, :, :, :]
                    i = i+1
            except:
                print('c', c)


            adv_vid_rs += MASK.cuda()
            del MASK
            top_val, top_idx, logits = model(adv_vid_rs)
            top_idx = top_idx.view(-1)

            loss_ori = torch.nn.functional.cross_entropy(logits, torch.tensor(ori_class.view(-1), dtype=torch.long,
                                                                              device='cuda').repeat(sub_num),
                                                         reduction='none')

            loss_tar = torch.nn.functional.cross_entropy(logits, torch.tensor(target_class.view(-1), dtype=torch.long, device='cuda').repeat(sub_num),
                                                     reduction='none')


            loss =  w1*loss_ori -(1-w1)*loss_tar
            batch_loss.append(loss)
            batch_idx.append(top_idx)
            batch_noise.append(all_noise)

        batch_noise = torch.cat(batch_noise, 0)
        batch_loss = torch.cat(batch_loss, 0)
        batch_idx = torch.cat(batch_idx)
        valid_loss = batch_loss
        loss_total += torch.mean(valid_loss).item()
        count_in += valid_loss.size(0)
        noise_select = batch_noise
        grads += torch.sum(noise_select / sigma * (valid_loss.view((-1,) + (1,) * (len(noise_select.size()) - 1))),0)

        l = torch.mean(batch_loss)
        return l, grads




def  untargeted_video_attack( vid_model, vid, x0, ori_class,  args,  eps=0.05,
                        max_lr=0.01, min_lr= 1e-3, effect_num=20, sample_per_draw=60,  sub_num_sample=2, max_iter=15000):

    # ----------------------------------------------------初始化---------------------------------------------------------

    len_limit = 10
    num_iter = 0
    T = 10
    lamda = 0.2
    fineS = 0.4
    fineT = 0.6
    cur_lr = max_lr
    last_p = []
    last_score = []
    score_list = []
    epis_reward_epochs = []
    baselines_s = 0
    im = x0[7, :, :, :].permute(1, 2, 0).cpu().numpy().astype(np.float32)
    fd = eb.image_2_foundation(im)
    nes_num = 0
    use_pre = False
    rein =True
    sparse_exp = False
    AST_t = AST_T(args)
    GetFeatures = AST_t.glance  # 特征提取器
    features, features_pool = GetFeatures(vid[:, :, :, :])
    features = features.squeeze()
    batch_size = 1
    h_s = init_hidden(batch_size, 512)
    c_s = init_hidden(batch_size, 512)
    hx_s = (h_s, c_s)
    step = 0
    MASK_s = torch.zeros(vid.size())

    state_dim = features.size(2)**2*32
    AST_s = AST_S(args, state_dim)
    AST_s.focuser.memory.hidden.append(hx_s)
    record_num = int(max_iter*(1/125))
    feature_pre = torch.ones(1, 1280)

    try:
        AST_t.cuda()
    except:
        pass
    confidence_lsit = []
    adv_vid = vid.clone()
    top_val, top_idx, logits = vid_model(adv_vid[None, :])
    num_iter += 1
    pre_confidence = top_val.view(-1).cpu()
    confidence_lsit.append(pre_confidence.cuda())
    pre_hunxiao_confidence = (F.softmax(logits.detach(), 1)).view(-1).sort()[0][-2].cpu()

    while num_iter < max_iter:
        # ---------------------------------------------------select-----------------------------------------------------

        # patch
        key_s_lists = []
        for focus_time_step in range(vid.size(0)):
            img = vid[focus_time_step, :, :, :]
            feature_s = features[focus_time_step, :].clone()
            key_s, HX_s = AST_s.one_step_act(img, feature_s, feature_pre, use_pre, resample=True, training=True)
            ks = key_s.cpu().numpy()
            for x1, x2, y1, y2 in ks:
                MASK_s[focus_time_step, :, y1:y2, x1:x2] = 1
            obj_score = eb.get_objectness(fd, x1, y1, x2, y2)[0]
            reward_step = obj_score
            AST_s.focuser.memory.rewards.append(reward_step)

            key_s_lists.append(key_s)
            step += 1
            if use_pre:
                key_patch = torch.zeros(vid.size())
                key_patch[:, :, y1:y2, x1:x2] =vid[:, :, y1:y2, x1:x2].clone()
                _, features_pre = GetFeatures(key_patch.cuda())
                feature_pre = features_pre[focus_time_step]
        mask_s = torch.cat(key_s_lists, 0).view(16, -1).cpu()


        # frame
        step = 0
        while True:
            AST_t.focuser.memory.clear_memory()
            h_t = init_hidden(batch_size, 512)
            c_t = init_hidden(batch_size, 512)
            hx = (h_t, c_t)


            feature_pre = torch.ones(1, 1280)
            key_t_lists = []
            R_rep_list = []
            for focus_time_step in range(vid.size(0)):
                feature_t = features_pool[focus_time_step, :].clone()
                key_t, HX = AST_t.one_step_act(feature_t, feature_pre, features_pool, hx, re_sample=False, training=True)
                hx = HX

                if key_t:
                   feature_pre = feature_t.clone()
                   R_rep = rep_frame(feature_t, features_pool)

                else:
                   feature_pre = torch.ones(1, 1280)
                   R_rep = 0
                key_t_lists.append(key_t)
                AST_t.focuser.memory.rewards.append(R_rep)
                R_rep_list.append(R_rep)

            actions_t = torch.cat(key_t_lists, 0).cpu().nonzero()
            ll = len(actions_t)
            step += 1
            if ll >= 2 and ll<=len_limit:
                mask_t = torch.cat(key_t_lists, 0).view(-1, 1).cpu()
                break

            if step > 2:
                    actions_t, key_t = speed_up_process(vid, len_limit)
                    mask_t = key_t.view(-1, 1).cpu()
                    break

        mask = mask_s * mask_t

        # ---------------------------------------------------nes--------------------------------------------------------
        patch_size = args.patch_size
        # patch+frame
        gs = nes_patch_frame(vid_model, adv_vid, patch_size, mask, actions_t,  sample_per_draw, ori_class, sub_num_sample, sigma=1e-3)

        num_iter += sample_per_draw
        nes_num += 1


        # ---------------------------------------------------噪声攻击-----------------------------------------------------
        proposed_adv_vid = adv_vid.clone()
        g = process_grad(gs, rein)

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
        top_val, top_idx, logits = vid_model(clip_frame[None, :])
        num_iter += 1

        if ori_class != top_idx[0][0]:
            adv_vid = clip_frame.clone()
            return True, num_iter, adv_vid


        if num_iter<600:
           if confidence_lsit[-1] <= top_val:
               cur_lr = max(cur_lr /1.5, min_lr)


        confidence_lsit.append(top_val)
        adv_vid = clip_frame

        # ---------------------------------------------------agent更新---------------------------------------------------
        # patch_agent更新
        reward_epoch0, adv_confidence, hunxiao_confidence, r_attack = reward_advantage(logits, ori_class,
                                                                                       pre_confidence,
                                                                                       pre_hunxiao_confidence, rein)
        pre_confidence = torch.tensor(adv_confidence).cpu()
        pre_hunxiao_confidence = torch.tensor(hunxiao_confidence).cpu()

        epis_reward_epochs.append(reward_epoch0.numpy())
        baselines_s = 0.9 * baselines_s + 0.1 * np.mean(epis_reward_epochs)
        reward_epoch = reward_epoch0 - baselines_s


        R_frame =  sparse_reward(actions_t, T, sparse_exp)
        reward_epoch_t = (1 - lamda) * reward_epoch - lamda * R_frame

        AST_s.focuser.memory.reward_epochs.append(torch.tensor(reward_epoch).cuda())
        AST_t.focuser.memory.reward_epochs.append(torch.tensor(reward_epoch_t).cuda())


        AST_s.focuser.update(fineS)
        AST_t.focuser.update(fineT)

        # --------------------------------------------------动态调整学习率-------------------------------------------------

        pre_score = top_val[0]
        del top_val
        del top_idx

        last_score.append(float(pre_score))
        last_score = last_score[-record_num:]
        if last_score[-1] >= last_score[0] and len(last_score) == record_num:
            return False, num_iter, adv_vid
        if len(last_score) > record_num:
            if last_score[-1] >= last_score[-(record_num/2)] and len(last_score) == record_num:
                min_lr = min_lr/5

        score_list.append(float(pre_score))
        score_list = score_list[-5:]
        if last_score[-1] >= last_score[0] and len(last_score) == 5:
            cur_lr = max(cur_lr / 2., min_lr)

        last_p.append(float(pre_score))
        last_p = last_p[-effect_num:]
        if last_p[-1] <= last_p[0] and len(last_p) == effect_num:
            if cur_lr > min_lr:
                cur_lr = max(cur_lr / 2., min_lr)
            last_p = []
        del clip_frame
        continue

    return False,  num_iter, adv_vid

def  targeted_video_attack( vid_model, vid, x0, ori_class, target_label, model_name, args,  eps=0.05,
                        max_lr=0.03, min_lr= 2.4e-3,  sample_per_draw=60,  sub_num_sample=2, max_iter=30000):

    # ----------------------------------------------------初始化---------------------------------------------------------
    len_limit = 10
    num_iter = 0
    T = 8
    lamda = 0.2
    fineS = 0.1
    fineT = 0.8
    cur_lr = max_lr
    epis_reward_epochs = []
    baselines_s = 0
    im = x0[7, :, :, :].permute(1, 2, 0).cpu().numpy().astype(np.float32)
    fd = eb.image_2_foundation(im)
    nes_num = 0
    AST_t = AST_T(args)
    GetFeatures = AST_t.glance
    features, features_pool = GetFeatures(vid[:, :, :, :])
    features = features.squeeze()
    batch_size = 1
    h_s = init_hidden(batch_size, 512)
    c_s = init_hidden(batch_size, 512)
    hx_s = (h_s, c_s)
    state_dim = features.size(2)**2*32
    AST_s = AST_S(args, state_dim)
    AST_s.focuser.memory.hidden.append(hx_s)
    last_loss = []
    target_prob_list = []
    sparse_exp = False
    use_pre = False
    try:
        AST_t.cuda()
    except:
        pass
    confidence_lsit = []
    adv_vid = vid.clone()
    top_val, top_idx, logits = vid_model(adv_vid[None, :])
    num_iter += 1


    if model_name == 'c3d':
        pre_confidence = (F.softmax(logits.detach(), 1)).view(-1)[ori_class].cpu()
        pre_hunxiao_confidence = (F.softmax(logits.detach(), 1)).view(-1)[target_label].cpu()
    elif model_name == 'tsm':
        pre_confidence = (logits.detach()).view(-1)[ori_class].cpu()
        pre_hunxiao_confidence = (logits.detach()).view(-1)[target_label].cpu()
    elif model_name == 'tsn':
        pre_confidence = (F.softmax(logits.detach(), 1)).view(-1)[ori_class].cpu()
        pre_hunxiao_confidence = (F.softmax(logits.detach(), 1)).view(-1)[target_label].cpu()
    else:
        pre_confidence = (logits.detach()).view(-1)[ori_class].cpu()
        pre_hunxiao_confidence = (logits.detach()).view(-1)[target_label].cpu()

    feature_pre = torch.ones(1, 1280)
    confidence_lsit.append(pre_confidence.cuda())
    while num_iter < max_iter:
        # ---------------------------------------------------select-----------------------------------------------------



        step = 0
        MASK_s = torch.zeros(vid.size())
        key_s_lists = []
        for focus_time_step in range(vid.size(0)):
            img = vid[focus_time_step, :, :, :]
            feature_s = features[focus_time_step, :].clone()
            key_s, HX_s = AST_s.one_step_act(img, feature_s, feature_pre, use_pre, resample=True, training=True)
            ks = key_s.cpu().numpy()
            for x1, x2, y1, y2 in ks:
                MASK_s[focus_time_step, :, y1:y2, x1:x2] = 1
            obj_score = eb.get_objectness(fd, x1, y1, x2, y2)[0]
            reward_step = obj_score
            AST_s.focuser.memory.rewards.append(reward_step)

            key_s_lists.append(key_s)
            step += 1
            if use_pre:
                key_patch = torch.zeros(vid.size())
                key_patch[:, :, y1:y2, x1:x2] =vid[:, :, y1:y2, x1:x2].clone()
                _, features_pre = GetFeatures(key_patch.cuda())
                feature_pre = features_pre[focus_time_step]
        mask_s = torch.cat(key_s_lists, 0).view(16, -1).cpu()


        # frame
        step = 0
        while True:
            AST_t.focuser.memory.clear_memory()
            h_t = init_hidden(batch_size, 512)
            c_t = init_hidden(batch_size, 512)
            hx = (h_t, c_t)


            feature_pre = torch.ones(1, 1280)
            key_t_lists = []
            R_rep_list = []
            for focus_time_step in range(vid.size(0)):
                feature_t = features_pool[focus_time_step, :].clone()
                key_t, HX = AST_t.one_step_act(feature_t, feature_pre, features_pool, hx, re_sample=False, training=True)
                hx = HX

                if key_t:
                   feature_pre = feature_t.clone()
                   R_rep = rep_frame(feature_t, features_pool)

                else:
                   feature_pre = torch.ones(1, 1280)
                   R_rep = 0
                key_t_lists.append(key_t)
                AST_t.focuser.memory.rewards.append(R_rep)
                R_rep_list.append(R_rep)

            actions_t = torch.cat(key_t_lists, 0).cpu().nonzero()
            ll = len(actions_t)
            step += 1
            if ll >= 2 and ll<=len_limit:
                mask_t = torch.cat(key_t_lists, 0).view(-1, 1).cpu()
                break

            if step > 2:
                    actions_t, key_t = speed_up_process(vid, len_limit)
                    mask_t = key_t.view(-1, 1).cpu()
                    break

        mask = mask_s * mask_t

        # ---------------------------------------------------nes--------------------------------------------------------
        # nes梯度估计
        patch_size = args.patch_size

        # patch+frame
        l, gs = nes_target(vid_model, adv_vid, patch_size, mask, actions_t, sample_per_draw, ori_class, target_label,
                          sub_num_sample, w1=0.1, sigma=1e-3)

        num_iter += sample_per_draw  # 累计查询次数
        nes_num += 1
        if l is None and g is None:
            print('nes sim fails, try again....')
            continue

        last_loss.append(l)
        last_loss = last_loss[-5:]
        if last_loss[-1] > last_loss[0]:
            if cur_lr > min_lr:
                cur_lr = max(cur_lr / 2., min_lr)
            last_loss = []
        # ---------------------------------------------------噪声攻击-----------------------------------------------------
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

        proposed_adv_vid = adv_vid.clone()
        proposed_adv_vid += cur_lr * MASK.sign().cuda()

        # 生成样本的裁剪
        bottom_bounded_adv = torch.where((vid - eps) > proposed_adv_vid, vid - eps,
                                         proposed_adv_vid)
        bounded_adv = torch.where((vid + eps) < bottom_bounded_adv, vid + eps, bottom_bounded_adv)
        clip_frame = torch.clamp( bounded_adv, 0., 1.)
        top_val, top_idx, logits = vid_model(clip_frame[None, :])
        num_iter += 1
        adv_vid = clip_frame.clone()

        if model_name=='c3d':
            target_prob = (F.softmax(logits.detach(), 1)).view(-1)[target_label].cpu()
            adv_confidence =(F.softmax(logits.detach(), 1)).view(-1)[ori_class].cpu()
        elif model_name=='tsm':
            target_prob = (logits.detach()).view(-1)[target_label].cpu()
            adv_confidence = (logits.detach()).view(-1)[ori_class].cpu()
        elif model_name=='tsn':
            target_prob = (F.softmax(logits.detach(), 1)).view(-1)[target_label].cpu()
            adv_confidence = (F.softmax(logits.detach(), 1)).view(-1)[ori_class].cpu()
        else:
            target_prob = (logits.detach()).view(-1)[target_label].cpu()
            adv_confidence = (logits.detach()).view(-1)[ori_class].cpu()
        target_prob_list.append(target_prob)
        target_prob_list = target_prob_list[-5:]


        if  top_idx[0][0] == target_label:
            return True, num_iter, adv_vid

        if num_iter > max_iter*1/3:
            if  adv_confidence.item() > target_prob.item()*6:
                return False, num_iter, adv_vid
        if num_iter > max_iter*1/2:
            if  adv_confidence.item() > target_prob.item() * 4:
                return False, num_iter, adv_vid
        if num_iter > max_iter*2/3:
            if  adv_confidence.item() > target_prob.item() * 2:
                return False, num_iter, adv_vid
        #---------------------------------------------------agent更新---------------------------------------------------
        # patch_agent更新
        reward_epoch0, adv_confidence, hunxiao_confidence, r_attack = target_reward_advantage(target_prob,adv_confidence,
                                                                                       pre_confidence,
                                                                                       pre_hunxiao_confidence)

        pre_confidence = torch.tensor(adv_confidence).cpu()
        pre_hunxiao_confidence = torch.tensor(hunxiao_confidence).cpu()
        epis_reward_epochs.append(reward_epoch0.numpy())
        baselines_s = 0.9 * baselines_s + 0.1 * np.mean(epis_reward_epochs)
        reward_epoch = reward_epoch0 - baselines_s


        R_frame = sparse_reward(actions_t, T, sparse_exp)
        reward_epoch_t = (1 - lamda) * reward_epoch - lamda * R_frame

        AST_s.focuser.memory.reward_epochs.append(torch.tensor(reward_epoch).cuda())
        AST_t.focuser.memory.reward_epochs.append(torch.tensor(reward_epoch_t).cuda())


        AST_s.focuser.update(fineS)
        AST_t.focuser.update(fineT)

    return False,  num_iter, adv_vid

