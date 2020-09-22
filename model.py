#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2020/9/14 下午2:15
# @Author : Pengyu Yang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MyModel(nn.Module):

    def __init__(self, args, embed, nlayers=3, dropout=0.1):
        super(MyModel, self).__init__()

        self.cuda = args.cuda
        self.nhead = args.n_head  # 多头注意力模型的头数
        self.hidden = args.word_emb_dim  # 编码器和解码器输入的大小
        self.doc_max_timesteps = args.doc_max_timesteps # 输入文档的最大句子数目
        self.outputSize = args.vocab_size  # 词汇表大小(用于预测每个单词输出的概率)
        self.sent_max_len = args.sent_max_len  # 输入句子的最大长度

        self.embed = embed  # encoder和decoder输入的embedding
        self.pos_encoder = PositionalEncoding(self.hidden, dropout, self.sent_max_len )  # 输入的位置编码

        self.pos_decoder = PositionalEncoding(self.hidden, dropout, self.sent_max_len*self.doc_max_timesteps)  # 输出的位置编码

        self.transformer = nn.Transformer(d_model=self.hidden, nhead=self.nhead, num_encoder_layers=nlayers,
                                          num_decoder_layers=nlayers, dim_feedforward=self.hidden, dropout=dropout, activation="gelu")

        self.transformerEncoderLayer = nn.TransformerEncoderLayer(d_model=self.hidden, nhead = self.nhead, dim_feedforward=self.hidden, dropout=0.1, activation="gelu")

        self.transformerEncoder = nn.TransformerEncoder(self.transformerEncoderLayer, num_layers=6)

        self.src_mask = None  # 输入序列的mask
        self.trg_mask = None  # 输出序列的mask
        self.memory_mask = None # encoder输出序列的mask

        self.fc_out_tran = nn.Linear(self.hidden, self.outputSize)

        # self.fc_out = nn.Linear(2*self.hidden, 2)
        self.fc_out = nn.Linear(self.hidden, 2)

        # word level BiLSTM
        self.word_RNN = nn.LSTM(
            input_size=args.word_emb_dim,
            hidden_size=self.hidden,
            batch_first=True,
            bidirectional=True
        )


    def generate_square_subsequent_mask(self, sz):
        """
        为序列生成mask
        :param sz:int:序列长度
        :return:mask:Tensor:mask Tensor
        """
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, source, target):
        # 交换batch size和sequence len
        src = source.transpose(1, 0)
        trg = target.repeat(self.doc_max_timesteps, 1).transpose(1, 0)

        # 解码器masked以防止当前位置Attend到后续位置
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)

        # Transformer根据每个句子生成文档
        src = self.embed(src)
        src = self.pos_encoder(src)

        trg = self.embed(trg)
        trg = self.pos_decoder(trg)

        gen_document = self.transformer(src, trg, tgt_mask=self.trg_mask).transpose(1, 0)

        classifier = F.softmax(self.fc_out_tran(gen_document), dim=2).max(dim=2)[1]

        # TransformerEncoder对生成文档编码
        doc_word_embed = self.embed(classifier)

        # word_hidden = self.word_RNN(word_embed)[0]
        doc_word_hidden = self.transformerEncoder(doc_word_embed)

        doc = torch.sum(doc_word_hidden, dim = 1)/(self.sent_max_len*self.doc_max_timesteps)

        output = self.fc_out(doc)

        return output
