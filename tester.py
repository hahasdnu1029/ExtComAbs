#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2020/9/15 下午5:06
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
import torch.nn.functional as F
import os
import time
import datetime
from tools.utils import eval_label
from tools.logger import *



class TestPipLine():
    def __init__(self, model_gen, model_class, m, test_dir, limited):
        """
            :param model: the model
            :param m: the number of sentence to select
            :param test_dir: for saving decode files
            :param limited: for limited Recall evaluation
        """
        self.model_gen = model_gen
        self.model_class = model_class
        self.limited = limited
        self.m = m
        self.test_dir = test_dir
        self.extracts = []

        self.batch_number = 0
        self.running_loss = 0
        self.example_num = 0
        self.total_sentence_num = 0

        self._hyps = []
        self._refer = []

    def evaluation(self, G, index, valset):
        pass

    def getMetric(self):
        pass

    def SaveDecodeFile(self):
        import datetime
        nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')  # 现在
        log_dir = os.path.join(self.test_dir, nowTime)
        with open(log_dir, "wb") as resfile:
            for i in range(self.rougePairNum):
                resfile.write(b"[Reference]\t")
                resfile.write(self._refer[i].encode('utf-8'))
                resfile.write(b"\n")
                resfile.write(b"[Hypothesis]\t")
                resfile.write(self._hyps[i].encode('utf-8'))
                resfile.write(b"\n")
                resfile.write(b"\n")
                resfile.write(b"\n")

    @property
    def running_avg_loss(self):
        return self.running_loss / self.batch_number

    @property
    def rougePairNum(self):
        return len(self._hyps)

    @property
    def hyps(self):
        if self.limited:
            hlist = []
            for i in range(self.rougePairNum):
                k = len(self._refer[i].split(" "))
                lh = " ".join(self._hyps[i].split(" ")[:k])
                hlist.append(lh)
            return hlist
        else:
            return self._hyps

    @property
    def refer(self):
        return self._refer

    @property
    def extractLabel(self):
        return self.extracts


class SLTester(TestPipLine):
    def __init__(self, model_gen, model_class, m, test_dir=None, limited=False, blocking_win=3):
        super().__init__(model_gen, model_class, m, test_dir, limited)
        self.pred, self.true, self.match, self.match_true = 0, 0, 0, 0
        self._F = 0
        self.criterion = torch.nn.CrossEntropyLoss()
        self.blocking_win = blocking_win

    def evaluation(self, features_in, features_out, labels, index, dataset, now_time, vocab,blocking=False):
        """
        :param features_in:
        :param features_out:
        :param label:
        :param index:
        :param dataset:
        :param blocking:
        :return:
        """
        self.batch_number += 1

        outputs = self.model_class.forward(self.model_gen, features_in, features_out)

        loss_outputs = torch.transpose(outputs, 2, 1)

        loss = self.criterion(loss_outputs, labels)

        self.running_loss += float(loss.data)

        labels = labels.cpu()

        for i in range(outputs.shape[0]):

            example = dataset.get_example(index[i])
            sent_max_number = example.doc_max_timesteps
            original_article_sents = example.original_article_sents
            refer = example.original_abstract

            softmax_value = F.softmax(outputs[i], dim=1).max(dim=1)[1]

            print(softmax_value)

            if blocking:
                pred_idx = self.ngram_blocking(original_article_sents, softmax_value, self.blocking_win,
                                               min(self.m, sent_max_number))
            else:
                topk, pred_idx = torch.topk(softmax_value, min(self.m, sent_max_number))
                print(topk)
                print(pred_idx)

            prediction = torch.zeros(sent_max_number).long()
            prediction[pred_idx] = 1

            self.extracts.append(pred_idx.tolist())

            self.pred += prediction.sum()
            self.true += labels[i].sum()

            self.match_true += ((prediction == labels[i]) & (prediction == 1)).sum()
            self.match += (prediction == labels[i]).sum()
            self.total_sentence_num += sent_max_number
            self.example_num += 1

            hyps = "\n".join(original_article_sents[id] for id in pred_idx if id < sent_max_number)

            path1 = os.path.join("save/eval","hyper")
            path2 = os.path.join("save/eval", "refer")
            if not os.path.exists(path1):
                os.mkdir(path1)
            if not os.path.exists(path2):
                os.mkdir(path2)
            hyper_txt = os.path.join(path1, "hyper"+now_time+".txt")
            refer_txt = os.path.join(path2, "refer"+now_time+".txt")
            h_f = open(hyper_txt, "a+")
            h_f.write(hyps)
            h_f.write('\n')
            h_f.write('\n')
            r_f = open(refer_txt, "a+")
            r_f.write(refer)
            r_f.write('\n')
            r_f.write('\n')
            self._hyps.append(hyps)
            self._refer.append(refer)

    def getMetric(self):
        logger.info("[INFO] Validset match_true %d, pred %d, true %d, total %d, match %d",
                    self.match_true, self.pred, self.true, self.total_sentence_num, self.match)
        self._accu, self._precision, self._recall, self._F = eval_label(
            self.match_true, self.pred, self.true, self.total_sentence_num, self.match)
        logger.info(
            "[INFO] The size of totalset is %d, sent_number is %d, accu is %f, precision is %f, recall is %f, F is %f",
            self.example_num, self.total_sentence_num, self._accu, self._precision, self._recall, self._F)

    def ngram_blocking(self, sents, p_sent, n_win, k):
        """
        :param p_sent: [sent_num, 1]
        :param n_win: int, n_win=2,3,4...
        :return:
        """
        ngram_list = []
        _, sorted_idx = p_sent.sort(descending=True)
        print(sorted_idx)
        S = []
        for idx in sorted_idx:
            sent = sents[idx]
            pieces = sent.split()
            overlap_flag = 0
            sent_ngram = []
            for i in range(len(pieces) - n_win):
                ngram = " ".join(pieces[i: (i + n_win)])
                if ngram in ngram_list:
                    overlap_flag = 1
                    break
                else:
                    sent_ngram.append(ngram)
            if overlap_flag == 0:
                S.append(idx)
                ngram_list.extend(sent_ngram)
                if len(S) >= k:
                    break
        S = torch.LongTensor(S)
        return S

    @property
    def labelMetric(self):
        return self._F


