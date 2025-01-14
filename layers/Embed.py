import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


def compared_version(ver1, ver2):
    """
    :param ver1
    :param ver2
    :return: ver1< = >ver2 False/True
    """
    list1 = str(ver1).split(".")
    list2 = str(ver2).split(".")

    for i in range(len(list1)) if len(list1) < len(list2) else range(len(list2)):
        if int(list1[i]) == int(list2[i]):
            pass
        elif int(list1[i]) < int(list2[i]):
            return -1
        else:
            return 1

    if len(list1) == len(list2):
        return True
    elif len(list1) < len(list2):
        return False
    else:
        return True


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1), :]  # [1 x L x d_model]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        # padding = 1 if compared_version(torch.__version__, '1.5.0') else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=1, padding=0, padding_mode='replicate', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type, freq):
        super(TemporalEmbedding, self).__init__()

        minute_size = 60
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        # year_size = 5

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        # if freq % datetime.timedelta(hours=1) > datetime.timedelta(0):
        #     self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        # self.day_embed = Embed(day_size, d_model)
        # self.month_embed = Embed(month_size, d_model)
        # self.year_embed = Embed(year_size, d_model)

    def forward(self, x):
        x = x.long()

        # minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        # day_x = self.day_embed(x[:, :, 1])
        # month_x = self.month_embed(x[:, :, 0])
        # year_x = x[:,:,0] - 2022
        # year_x = self.year_embed(year_x)

        return hour_x + weekday_x  # + day_x + month_x + minute_x  # + year_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type, freq):
        raise NotImplementedError("This module is not adapted for the specific project for which Autoformer is used.")
        super(TimeFeatureEmbedding, self).__init__()

        # freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        # d_inp = freq_map[freq]
        d_inp = 5
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class NodeEmbeddingEmbedding(nn.Module):
    def __init__(self, d_model, ne_dimensions, features, c_in):
        super().__init__()
        if features == "S":
            self.embed = nn.Linear(ne_dimensions, d_model)
        else:
            self.embed = nn.Linear(ne_dimensions * c_in, d_model)

    def forward(self, x):
        return self.embed(x)


class RPE(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, pos0):
        return self.pe[:, pos0: (x.size(1) + pos0), :]  # [1 x L x d_model]


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type, freq, dropout, ne_dimensions, features):
        super(DataEmbedding, self).__init__()
        self.data_embedding_wo_pos = DataEmbedding_wo_pos(c_in=c_in, d_model=d_model, embed_type=embed_type, freq=freq,
                                                          dropout=0, ne_dimensions=ne_dimensions, features=features)
        self.position_embedding = RPE(d_model)  # PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark, node_embedding=None, pos0=0):
        x = self.data_embedding_wo_pos(x, x_mark, node_embedding) + self.position_embedding(x, pos0)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type, freq, dropout, ne_dimensions, features):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        if ne_dimensions is not None:
            self.node_embedding_embedding = NodeEmbeddingEmbedding(d_model=d_model, ne_dimensions=ne_dimensions,
                                                                   features=features, c_in=c_in)
        else:
            self.node_embedding_embedding = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark, node_embedding=None):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        if node_embedding is not None:
            x += self.node_embedding_embedding(node_embedding.reshape(x.shape[0], 1, -1))
        return self.dropout(x)
