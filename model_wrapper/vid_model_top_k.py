import torch
import torch.nn as nn
import numpy as np

class SLOWFAST_K_Model():
    def __init__(self, model):
        self.k = 1
        self.model = model

    def set_k(self, k):
        self.k = k

    def get_top_k(self, vid, k):
        vid_t = vid.clone().squeeze()
        if len(vid_t.size()) ==5:
            vid_t = vid_t.transpose(2,1)
            vid_t = vid_t.unsqueeze(1)
        else:
            vid_t = vid_t.transpose(1,0)
            vid_t =vid_t[None,None, :]
        vid_t = vid_t.cuda()
        with torch.no_grad():
            logits = self.model(vid_t, return_loss=False)
            logits = torch.tensor(logits)
        ls =torch.sum(logits)
        if ls==1:
            top_val, top_idx = torch.topk(logits, k)
        else:
            top_val, top_idx = torch.topk(nn.functional.softmax(logits, 1), k)
        return top_val.cuda(), top_idx.cuda(), logits.cuda()

    def __call__(self, vid):
        return self.get_top_k(vid, self.k)


class TSN_K_Model():
    def __init__(self, model):
        self.k = 1
        self.model = model

    def set_k(self, k):
        self.k = k

    def get_top_k(self, vid, k):
        vid_t = vid.clone().squeeze()
        if len(vid_t.size()) ==5:
            vid_t = vid_t
        else:
            vid_t = vid_t[None, :]
        vid_t = vid_t.cuda()
        with torch.no_grad():
            logits = self.model(vid_t, return_loss=False)
            logits = torch.tensor(logits)
        ls = torch.sum(logits)
        if ls==1:
            top_val, top_idx = torch.topk(logits, k)
        else:
            top_val, top_idx = torch.topk(nn.functional.softmax(logits, 1), k)
        return top_val.cuda(), top_idx.cuda(), logits.cuda()

    def __call__(self, vid):
        return self.get_top_k(vid, self.k)

class TSM_K_Model():
    def __init__(self, model):
        self.k = 1
        self.model = model

    def set_k(self, k):
        self.k = k

    def get_top_k(self, vid, k):
        vid_t = vid.clone().squeeze()
        if len(vid_t.size()) ==5:
            vid_t = vid_t
        else:
            vid_t =vid_t[None, :]
        vid_t = vid_t.cuda()
        with torch.no_grad():
            logits = self.model(vid_t, return_loss=False)
            logits = torch.tensor(logits)
        ls = torch.sum(logits)
        if ls==1:
            top_val, top_idx = torch.topk(logits, k)
        else:
            top_val, top_idx = torch.topk(nn.functional.softmax(logits, 1), k)
        return top_val.cuda(), top_idx.cuda(), logits.cuda()

    def __call__(self, vid):
        return self.get_top_k(vid, self.k)

class X_K_Model():
    def __init__(self, model):
        self.k = 1
        self.model = model

    def set_k(self, k):
        self.k = k

    def get_top_k(self, vid, k):
        vid_t = vid.clone().squeeze()
        if len(vid_t.size()) ==5:
            vid_t = np.transpose(vid_t.cpu().numpy(), (0, 2, 1, 3, 4))
            vid_t = torch.tensor(vid_t)
            vid_t = vid_t.unsqueeze(1)
        else:
            vid_t = np.transpose(vid_t.cpu().numpy(),(1,0,2,3))
            vid_t = torch.tensor(vid_t)
            vid_t =vid_t[None,None, :]
        vid_t = vid_t.cuda()
        with torch.no_grad():
            logits = self.model(vid_t, return_loss=False)
            logits = torch.tensor(logits)
        top_val, top_idx = torch.topk(nn.functional.softmax(logits, 1), k)
        return top_val.cuda(), top_idx.cuda(), logits.cuda()

    def __call__(self, vid):
        return self.get_top_k(vid, self.k)





class Oud_K_Model():
    def __init__(self, model):
        self.k = 1
        self.model = model

    def set_k(self, k):
        self.k = k

    def preprocess(self, vid):
        vid_t = vid.clone()
        mean = torch.tensor([101.2198, 97.5751, 89.5303], dtype=torch.float32,
                            device=vid.get_device())[None, None, :, None, None]
        vid_t = vid_t * 255
        vid_t[vid_t > 255] = 255
        vid_t[vid_t < 0] = 0
        vid_t.sub_(mean)
        vid_t = vid_t.permute(0, 2, 1, 3, 4)
        return vid_t

    def get_top_k(self, vid, k):
        with torch.no_grad():

            logits = self.model(self.preprocess(vid))
        top_val, top_idx = torch.topk(nn.functional.softmax(logits, 1), k)
        return top_val, top_idx, logits

    def __call__(self, vid):
        return self.get_top_k(vid, self.k)

# C3D 预测结果的K个结果
class C3D_K_Model():
    def __init__(self, model):
        self.k = 1
        self.model = model

    def set_k(self, k):
        self.k = k

    def preprocess(self, vid):
        vid_t = vid.clone()
        mean = torch.tensor([101.2198, 97.5751, 89.5303], dtype=torch.float32,
                            device=vid.get_device())[None, None, :, None, None]
        vid_t = vid_t * 255
        vid_t[vid_t > 255] = 255
        vid_t[vid_t < 0] = 0
        vid_t.sub_(mean)
        vid_t = vid_t.permute(0, 2, 1, 3, 4)
        return vid_t

    def get_top_k(self, vid, k):
        with torch.no_grad():
            logits = self.model(self.preprocess(vid))
        top_val, top_idx = torch.topk(nn.functional.softmax(logits, 1), k)
        return top_val, top_idx, logits

    def __call__(self, vid):
        return self.get_top_k(vid, self.k)



class C3D_K_Model_k400():
    def __init__(self, model):
        self.k = 1
        self.model = model

    def set_k(self, k):
        self.k = k

    def preprocess(self, vid):
        vid_t = vid.clone()
        mean = torch.tensor( [114.7748, 107.7354, 99.4750], dtype=torch.float32,
                            device=vid.get_device())[None, None, :, None, None]
        vid_t = vid_t * 255
        vid_t[vid_t > 255] = 255
        vid_t[vid_t < 0] = 0
        vid_t.sub_(mean)
        vid_t = vid_t.permute(0, 2, 1, 3, 4)
        return vid_t

    def get_top_k(self, vid, k):
        with torch.no_grad():
            logits = self.model(self.preprocess(vid))
        top_val, top_idx = torch.topk(nn.functional.softmax(logits, 1), k)
        return top_val, top_idx, logits

    def __call__(self, vid):
        return self.get_top_k(vid, self.k)