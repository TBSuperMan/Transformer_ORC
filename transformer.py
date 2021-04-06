import torch.nn as nn
import torch
import torch.nn.functional as F
import copy, math
from torchvision import models

def clone(module, k):
    return nn.ModuleList([copy.deepcopy(module) for i in range(k)])

class Transformer(nn.Module):
    def __init__(self, c_model, c_feature, c_feedforw, layer_name, dropout, voc_len):
        super(Transformer, self).__init__()
        self.featureExtractor = FeatureExtractor(c_model, c_feature, layer_name)
        positionEncoder = PositionEncoder(c_model, dropout)
        self.img_pos_enc = copy.deepcopy(positionEncoder)
        self.tgt_pos_enc = copy.deepcopy(positionEncoder)
        self.self_attention = SelfAttention(c_model)
        self.pos_wise = PositionWise(c_model, c_feedforw)
        self.encoder = Encoder(EncoderLayer(copy.deepcopy(self.self_attention),
                                            copy.deepcopy(self.pos_wise),
                                            c_model, dropout), 4)
        self.decoder = Decoder(DecoderLayer(copy.deepcopy(self.self_attention),
                                            copy.deepcopy(self.self_attention),
                                            copy.deepcopy(self.pos_wise), c_model, dropout), 4)
        self.tgt_embedding = TargetEmbedding(c_model, voc_len)
        self.out_softmax = OutProcess(c_model, voc_len)

    def forward(self, x, src_mask, tgt, tgt_mask):
        # print('trans file x size:', x.size())
        # print('trans file src_mask size:', src_mask.size())
        # print('trans file tgt_mask size:', tgt_mask.size())
        x = self.featureExtractor(x)#提取特征  batch * (h/16 * w/16) * c_model
        # print("x:",x.shape)
        x = self.img_pos_enc(x)#加入位置编码，同时设置一定的dropout

        x = self.encoder(x, src_mask)

        tgt = self.tgt_embedding(tgt.long())
        tgt = self.tgt_pos_enc(tgt)

        x = self.decoder(tgt, x, src_mask, tgt_mask)

        return self.out_softmax(x)


class EncoderLayer(nn.Module):
    def __init__(self, self_atte, pos_wise, c_model, dropout):
        super(EncoderLayer, self).__init__()
        self.sublayers = clone(SubLayer(c_model, dropout), 2)
        self.attention = self_atte
        self.pos_wise = pos_wise
        self.c_model = c_model

    def forward(self, x, mask):
        x = self.sublayers[0](x, lambda x: self.attention(x, x, x, mask))
        x = self.sublayers[1](x, self.pos_wise)
        return x

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.c_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, self_atte, src_atte, pos_wise, c_model, dropout):
        super(DecoderLayer, self).__init__()
        self.sublayers = clone(SubLayer(c_model, dropout), 3)
        self.self_atte = self_atte
        self.src_atte = src_atte
        self.pos_wise = pos_wise
        self.c_model = c_model

    def forward(self, x, memory, src_mask, tag_mask):
        m = memory
        x = self.sublayers[0](x, lambda x: self.self_atte(x, x, x, tag_mask))
        x = self.sublayers[1](x, lambda x: self.src_atte(x, m, m, src_mask))
        x = self.sublayers[2](x, self.pos_wise)
        return x

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.c_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class OutProcess(nn.Module):
    def __init__(self, c_model, voc_len):
        super(OutProcess, self).__init__()
        self.linear = nn.Linear(c_model, voc_len)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, c_model, h=8, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.h = h
        self.d_k = c_model // h
        self.c_model = c_model
        self.linears = clone(nn.Linear(c_model, c_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask):
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears,(query, key, value))]
        # print("quary ",query.shape)
        #query:batch * h * len * d_k
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(self.d_k)
        #scores : batch * h * len * len
        # print("scores ",scores.shape)
        # print("mask ",mask.shape)
        # print(mask)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) #？？ mask全1
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        res = torch.matmul(p_attn, value).transpose(1, 2)
        res = res.contiguous().view(batch_size, -1, self.c_model)
        return self.linears[-1](res)

class PositionWise(nn.Module):
    def __init__(self, c_model, c_ff, dropout=0.1):
        super(PositionWise, self).__init__()
        self.linear1 = nn.Linear(c_model, c_ff)
        self.linear2 = nn.Linear(c_ff, c_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class SubLayer(nn.Module):
    def __init__(self, c_model, dropout):
        super(SubLayer, self).__init__()
        self.norm = LayerNorm(c_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))# ??????顺序问题

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class PositionEncoder(nn.Module):
    def __init__(self, c_model, dropout, max_len=5000):
        super(PositionEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = nn.Parameter(data=torch.randn(1, max_len, c_model))
        self.register_parameter('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class FeatureExtractor(nn.Module):
    def __init__(self, c_model, c_feature, layer_name):
        super(FeatureExtractor, self).__init__()
        self.resnet = getattr(models, 'resnet101')(pretrained=False)
        self.layer_name = layer_name
        self.linear = nn.Linear(c_feature, c_model)

    def forward(self, x):
        for name, module in self.resnet._modules.items():
            x = module(x)
            if name == self.layer_name:
                batch = x.size(0)
                channel = x.size(1)
                return self.linear(x.view(batch, channel, -1).permute(0, 2, 1))
        return None

class TargetEmbedding(nn.Module):
    def __init__(self, c_model, voc_len):
        super(TargetEmbedding, self).__init__()
        self.embedding = nn.Embedding(voc_len, c_model)
        self.c_model = c_model

    def forward(self, tag):
        return self.embedding(tag) / math.sqrt(self.c_model)


