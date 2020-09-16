#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2020/9/15 下午3:09
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

import time
import json
import torch.utils.data
from tools.logger import *


class Example(object):
    """Class representing a train/val/test example for single-document extractive summarization."""

    def __init__(self, article_sents, abstract_sents, vocab, sent_max_len, doc_max_timesteps, label):
        """ Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.

        :param article_sents: list(strings) for single document or list(list(string)) for multi-document; one per article sentence. each token is separated by a single space.
        :param abstract_sents: list(strings); one per abstract sentence. In each sentence, each token is separated by a single space.
        :param vocab: Vocabulary object
        :param sent_max_len: int, max length of each sentence
        :param label: list, the No of selected sentence, e.g. [1,3,5]
        """

        self.sent_max_len = sent_max_len
        self.enc_sent_len = []
        self.enc_sent_input = []
        self.enc_sent_input_pad = []
        self.enc_sent_input_pad_together = []

        # Store the original strings
        self.original_article_sents = article_sents
        self.original_abstract = "\n".join(abstract_sents)

        if(len(article_sents) > doc_max_timesteps ):
            article_sents = article_sents[:doc_max_timesteps]
        else:
            for i in range(doc_max_timesteps - len(article_sents)):
                article_sents.append('[PAD]')

        # Process the sentences
        for sent in article_sents:
            article_words = sent.split()
            self.enc_sent_len.append(len(article_words))  # store the length before padding
            self.enc_sent_input.append([vocab.word2id(w.lower()) for w in
                                        article_words])  # list of word ids; OOVs are represented by the id for UNK token
        self._pad_encoder_input(vocab.word2id('[PAD]'))

        # Process the article
        for i in range(len(self.enc_sent_input_pad)):
            self.enc_sent_input_pad_together.extend(self.enc_sent_input_pad[i])

        # Store the label
        self.label = [0 for i in range(sent_max_len)]
        for i in label:
            if(i < sent_max_len):
                self.label[i] = 1

    def _pad_encoder_input(self, pad_id):
        """
        :param pad_id: int; token pad id
        :return:
        """
        max_len = self.sent_max_len
        for i in range(len(self.enc_sent_input)):
            article_words = self.enc_sent_input[i].copy()
            if len(article_words) > max_len:
                article_words = article_words[:max_len]
            if len(article_words) < max_len:
                article_words.extend([pad_id] * (max_len - len(article_words)))
            self.enc_sent_input_pad.append(article_words)


class ExampleSet(torch.utils.data.Dataset):
    """ Constructor: Dataset of example(object) for single document summarization"""

    def __init__(self, data_path, vocab, doc_max_timesteps, sent_max_len):
        """ Initializes the ExampleSet with the path of data

        :param data_path: string; the path of data
        :param vocab: object;
        :param doc_max_timesteps: int; the maximum sentence number of a document, each example should pad sentences to this length
        :param sent_max_len: int; the maximum token number of a sentence, each sentence should pad tokens to this length
        :param filter_word_path: str; file path, the file must contain one word for each line and the tfidf value must go from low to high (the format can refer to script/lowTFIDFWords.py)
        :param w2s_path: str; file path, each line in the file contain a json format data (which can refer to the format can refer to script/calw2sTFIDF.py)
        """

        self.vocab = vocab
        self.sent_max_len = sent_max_len
        self.doc_max_timesteps = doc_max_timesteps

        logger.info("[INFO] Start reading %s", self.__class__.__name__)
        start = time.time()
        self.example_list = readJson(data_path)
        logger.info("[INFO] Finish reading %s. Total time is %f, Total size is %d", self.__class__.__name__,
                    time.time() - start, len(self.example_list))
        self.size = len(self.example_list)

    def get_example(self, index):
        e = self.example_list[index]
        e["summary"] = e.setdefault("summary", [])
        example = Example(e["text"], e["summary"], self.vocab, self.sent_max_len, self.doc_max_timesteps, e["label"])
        return example

    def __getitem__(self, index):
        """
        :param index: int; the index of the example
        :return
        """
        example = self.get_example(index)

        return torch.LongTensor(example.enc_sent_input_pad), torch.LongTensor(example.enc_sent_input_pad_together), torch.LongTensor(example.label), index

    def __len__(self):
        return self.size


def readJson(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data