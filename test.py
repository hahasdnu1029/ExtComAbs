#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2020/9/14 下午4:31
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

loss = nn.CrossEntropyLoss()
# # input is of size nBatch x nClasses = 3 x 5
# input = torch.autograd.Variable(torch.randn(3, 5, 4), requires_grad=True)
# # each element in target has to have 0 <= value < nclasses
# target = torch.autograd.Variable(torch.LongTensor([[1, 0, 4,1,1],[1, 0, 4,1,1],[1, 0, 4,1,1]]))
# output = loss(input, target)

a = torch.randn(3,4)
print(a.dtype)
print(a)
print((a[0][0]).data)


