# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os

import numpy as np

import ndata_util

import hi

class KinQueryDataset:
    """
        지식인 데이터를 읽어서, tuple (데이터, 레이블)의 형태로 리턴하는 파이썬 오브젝트 입니다.
    """
    def __init__(self, dataset_path: str, max_length: int, maxvocabsize: int):
        """
        :param dataset_path: 데이터셋 root path
        :param max_length: 문자열의 최대 길이
        """
        # 데이터, 레이블 각각의 경로
        data_review = os.path.join(dataset_path, 'train', 'train_data')
        labels_path = os.path.join(dataset_path, 'train', 'train_label')

        # 지식인 데이터를 읽고 preprocess까지 진행합니다
#with open(data_review, 'rt', encoding='utf-8') as f:
#            self.queries = preprocess(f.readlines(), max_length, maxvocabsize)
        contents = hi.loading_data(data_review, eng=False, num=False, punc=False)
        wordtoix, ixtoword = hi.make_dict_all_cut(contents, minlength=0, maxlength=3, jamo_delete=True)
        self.queries = hi.make_inputs(contents, wordtoix, max_length,shuffle=False)

        # 지식인 레이블을 읽고 preprocess까지 진행합니다.
        with open(labels_path) as f:
            self.labels = np.array([[np.float32(x)] for x in f.readlines()])

    def __len__(self):
        """
        :return: 전체 데이터의 수를 리턴합니다
        """
        return len(self.queries)

    def __getitem__(self, idx):
        """

        :param idx: 필요한 데이터의 인덱스
        :return: 인덱스에 맞는 데이터, 레이블 pair를 리턴합니다
        """
        return self.queries[idx], self.labels[idx]


def preprocess(data: list, max_length: int, maxvocabsize):
    vectorized_data = ndata_util.word_to_vec(data,maxvocabsize)
    zero_padding = np.zeros((len(data), max_length), dtype=np.int32)
    #print(data)
    for idx, seq in enumerate(vectorized_data):
        length = len(seq)
        #print(idx, seq)
        if length >= max_length:
            length = max_length
            zero_padding[idx, :length] = np.array(seq)[:length]
        elif(length == 0):
            zero_padding[idx,] = zero_padding[idx,]
        else:
            zero_padding[idx,] = np.append(zero_padding[idx,:-length], np.array(seq))
    return zero_padding

def test_preprocess(data: list, dic: list, max_length: int):
    vectorized_data = ndata_util.load_test_data(data, dic)
    zero_padding = np.zeros((len(data), max_length), dtype=np.int32)
    for idx, seq in enumerate(vectorized_data):
        length = len(seq)
        if length >= max_length:
            length = max_length
            zero_padding[idx, :length] = np.array(seq)[:length]
        elif(length == 0):
            zero_padding[idx,] = zero_padding[idx,]
        else:
            zero_padding[idx,] = np.append(zero_padding[idx,:-length], np.array(seq))
    return zero_padding
