import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from functools import wraps


def build_mlp(input_dim, hidden_dims, output_dim=None,
              use_batchnorm=False, dropout=0):
    layers = []
    D = input_dim
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(input_dim))
    if hidden_dims:
        for dim in hidden_dims:
            layers.append(nn.Linear(D, dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            D = dim
    if output_dim:
        layers.append(nn.Linear(D, output_dim))
    return nn.Sequential(*layers)


class LayerNormalization(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out


class SoftDotAttention(nn.Module):
    """Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, input_dim, output_dim):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        # TODO: why not bias?
        self.linear_in = nn.Linear(input_dim, output_dim, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, h, context, mask=None):
        """Propagate h through the network.

        h: batch x dim (concat(img, action))
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        """
        # TODO: rely a single FC layer to map (img, action) to instruction is too heavy
        # we probably need attention over attention or Transformer
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len

        # TODO: attention usually tries to find the correlation
        # the correlation of past (image features, actions) and words
        # should be "avoid" not "focused on"
        # attn = -attn

        if mask is not None:
            # -Inf masking prior to the softmax
            # attn.data.masked_fill_(mask, -float('inf'))
            attn.data.masked_fill_((mask == 0).data, -float('inf'))
        attn = self.softmax(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        # h_tilde = torch.cat((weighted_context, h), 1)
        # h_tilde = self.tanh(self.linear_out(h_tilde))
        return weighted_context, attn


class SoftAttention(nn.Module):
    """Soft-Attention without learnable parameters
    """

    def __init__(self):
        super(SoftAttention, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, h, context, mask=None):
        """Propagate h through the network.

        h: batch x dim (concat(img, action))
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        """
        # Get attention
        attn = torch.bmm(context, h.unsqueeze(2)).squeeze(2)  # batch x seq_len

        if mask is not None:
            attn.data.masked_fill_((mask == 0).data, -float('inf'))
        attn = self.softmax(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim

        return weighted_context, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, attn_mask=None):
        _, len_q, _ = q.size()
        _, len_k, _ = k.size()

        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).expand_as(attn)
            # TODO: use detach()?
            attn.data.masked_fill_((attn_mask == 0).data, -float('inf'))

        attn_weight = self.softmax(attn.view(-1, len_k)).view(-1, len_q, len_k)

        attn_weight = self.dropout(attn_weight)
        output = torch.bmm(attn_weight, v)
        return output, attn_weight


class PositionalEncoding(nn.Module):
    """Implement the PE function to introduce the concept of relative position"""

    def __init__(self, d_model, dropout, max_len=80):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) *-(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # dim 2i
        pe[:, 1::2] = torch.cos(position * div_term)  # dim 2i + 1
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # register_buffer 类似 nn.Parameter，load module 的时候会被加载，但是 optim.step 不会被更新

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)].cuda(), requires_grad=False)
        # return self.dropout(x)
        return x

class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1) # position-wise
        self.layer_norm = LayerNormalization(d_hid)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return self.layer_norm(output + residual)


class SelfAttnLayer(nn.Module):
    """A sub-layer for performing self-attention"""

    def __init__(self, ctx_dim, img_dim):
        super(SelfAttnLayer, self).__init__()
        self.ctx_attn = ScaledDotProductAttention(ctx_dim)
        self.img_attn = ScaledDotProductAttention(img_dim)

        self.lang_fc = nn.Linear(ctx_dim, ctx_dim)
        self.img_fc = nn.Linear(img_dim, img_dim)

        # self.lang_norm = nn.BatchNorm1d(opts.rnn_hidden_size)
        # self.img_norm = nn.BatchNorm1d(opts.img_fc_dim[-1])
        self.lang_norm = LayerNormalization(ctx_dim)
        self.img_norm = LayerNormalization(img_dim)

        self.lang_pos_ffn = PositionwiseFeedForward(ctx_dim, ctx_dim)
        self.img_pos_ffn = PositionwiseFeedForward(img_dim, img_dim)

    def forward(self, ctx, ctx_mask, img_feat, img_feat_mask):
        batchsize = ctx.size(0)

        weighted_ctx, _ = self.ctx_attn(ctx, ctx, ctx, attn_mask=ctx_mask)
        weighted_img, _ = self.img_attn(img_feat, img_feat, img_feat, attn_mask=img_feat_mask)

        # FC + short cut connection
        weighted_ctx = self.lang_fc(weighted_ctx.view(-1, weighted_ctx.size(2))) + ctx.view(-1, weighted_ctx.size(2))
        weighted_img = self.img_fc(weighted_img.view(-1, weighted_img.size(2))) + img_feat.view(-1, weighted_img.size(2))

        # BN or LN
        weighted_ctx = self.lang_norm(weighted_ctx)
        weighted_img = self.img_norm(weighted_img)

        weighted_ctx = weighted_ctx.view(batchsize, -1, weighted_ctx.size(1))
        weighted_img = weighted_img.view(batchsize, -1, weighted_img.size(1))

        ctx_output = self.lang_pos_ffn(weighted_ctx)
        img_output = self.img_pos_ffn(weighted_img)

        return ctx_output, img_output


class CtxSelfAttnLayer(nn.Module):
    """A sub-layer for performing self-attention"""

    def __init__(self, rnn_hidden_size):
        super(CtxSelfAttnLayer, self).__init__()
        self.ctx_attn = ScaledDotProductAttention(rnn_hidden_size)

        self.lang_fc = nn.Linear(rnn_hidden_size, rnn_hidden_size)

        self.lang_norm = LayerNormalization(rnn_hidden_size)

        self.lang_pos_ffn = PositionwiseFeedForward(rnn_hidden_size, rnn_hidden_size)

    def forward(self, ctx, ctx_mask):
        batchsize = ctx.size(0)

        weighted_ctx, _ = self.ctx_attn(ctx, ctx, ctx, attn_mask=ctx_mask)

        # FC + short cut connection
        weighted_ctx = self.lang_fc(weighted_ctx.view(-1, weighted_ctx.size(2))) + ctx.view(-1, weighted_ctx.size(2))

        # LN
        weighted_ctx = self.lang_norm(weighted_ctx)

        weighted_ctx = weighted_ctx.view(batchsize, -1, weighted_ctx.size(1))

        ctx_output = self.lang_pos_ffn(weighted_ctx)

        return ctx_output


def proj_masking(feat, projector, mask=None):
    """Universal projector and masking"""
    proj_feat = projector(feat.view(-1, feat.size(2)))
    proj_feat = proj_feat.view(feat.size(0), feat.size(1), -1)
    if mask is not None:
        return proj_feat * mask.unsqueeze(2).expand_as(proj_feat)
    else:
        return proj_feat


class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            # print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)

