#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2020/9/14 上午11:22
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

import os
import json
import nltk
import argparse


def print_information(keys, allcnt):
    # Vocab > 10
    cnt = 0
    first = 0.0
    for key, val in keys:
        if val >= 10:
            cnt += 1
            first += val
    print("appearance > 10 cnt: %d, percent: %f" % (cnt, first / allcnt))  # 416,303

    # first 30,000, last freq 31
    if len(keys) > 30000:
        first = 0.0
        for k, v in keys[:30000]:
            first += v
        print("First 30,000 percent: %f, last freq %d" % (first / allcnt, keys[30000][1]))

    # first 50,000, last freq 383
    if len(keys) > 50000:
        first = 0.0
        for k, v in keys[:50000]:
            first += v
        print("First 50,000 percent: %f, last freq %d" % (first / allcnt, keys[50000][1]))

    # first 100,000, last freq 107
    if len(keys) > 100000:
        first = 0.0
        for k, v in keys[:100000]:
            first += v
        print("First 100,000 percent: %f, last freq %d" % (first / allcnt, keys[100000][1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='datasets/cnndm/train.label.jsonl2', help='File to deal with')
    parser.add_argument('--dataset', type=str, default='cnndm', help='dataset name')

    args = parser.parse_args()

    save_dir = os.path.join("cache", args.dataset)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    saveFile = os.path.join(save_dir, "vocab")
    print("Save vocab of dataset %s to %s" % (args.dataset, saveFile))

    text = []
    summaries = []
    allword = []
    cnt = 0
    with open(args.data_path, encoding='utf8') as f:
        for line in f:
            e = json.loads(line)# json文件一行表示一个Example(一篇文章)
            text = " ".join(e["text"])
            allword.extend(text.split())
            cnt += 1
    print("Training set of dataset has %d example" % cnt)

    fdist1 = nltk.FreqDist(allword)

    fout = open(saveFile, "w")
    keys = fdist1.most_common()
    for key, val in keys:
        try:
            fout.write("%s\t%d\n" % (key, val))
        except UnicodeEncodeError as e:
            continue
    fout.close()

    allcnt = fdist1.N()
    allset = fdist1.B()
    print("All appearance %d, unique word %d" % (allcnt, allset))

    print_information(keys, allcnt)
