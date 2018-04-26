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


import argparse
import os

import numpy as np
import tensorflow as tf

from dataset import KinQueryDataset, preprocess, test_preprocess


def _batch_loader(iterable, n=1):

    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=10)
    args.add_argument('--batch', type=int, default=200)
    args.add_argument('--strmaxlen', type=int, default=50)
    args.add_argument('--embedding', type=int, default=30)
    args.add_argument('--maxvocabsize', type=int, default=10000)
    config = args.parse_args()

    DATASET_PATH = '../sample_data/movie_review/'

    # 모델의 specification
    input_size = config.embedding*config.strmaxlen
    output_size = 1
    hidden_layer_size = 200
    learning_rate = 0.001
    character_size = 251

    x = tf.placeholder(tf.int32, [None, config.strmaxlen])
    y_ = tf.placeholder(tf.float32, [None, output_size])
    do = tf.placeholder(tf.float32)
    
    # 임베딩
    char_embedding = tf.get_variable('char_embedding', [config.maxvocabsize, config.embedding])
    embedded = tf.nn.embedding_lookup(char_embedding, x)

    # 첫 번째 레이어
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer_size)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=do)
#    cell2 = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer_size)
#    cell2 = tf.nn.rnn_cell.DropoutWrapper(cell2, output_keep_prob=do)
#    multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell, cell2])

#    multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell, cell2])

    output, state = tf.nn.dynamic_rnn(cell, embedded, dtype=tf.float32)

    output = tf.transpose(output, [1, 0, 2])
    output = output[-1]

    # 두 번째 (아웃풋) 레이어
    second_layer_weight = weight_variable([hidden_layer_size, output_size])
    second_layer_bias = bias_variable([output_size])
    foutput = tf.matmul(output, second_layer_weight) + second_layer_bias
    #output_sigmoid = tf.sigmoid(output)

    # loss와 optimizer
    global_step = tf.Variable(0)
    learning_rate= tf.train.exponential_decay(learning_rate, global_step, 10000, 0.75)
    cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=foutput))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    if config.mode == 'train':
        import time
        import math

        def timeSince(since):
            now = time.time()
            s = now - since
            m = math.floor(s / 60)
            s -= m * 60
            return '%dm %ds' % (m, s)

        start = time.time()
        
        # 데이터를 로드합니다.
        dataset = KinQueryDataset(DATASET_PATH, config.strmaxlen, config.maxvocabsize)
        dataset_len = len(dataset)
        one_batch_size = dataset_len//config.batch
        if dataset_len % config.batch != 0:
            one_batch_size += 1
        # epoch마다 학습을 수행합니다.
        for epoch in range(config.epochs):
            avg_loss = 0.0
            for i, (data, labels) in enumerate(_batch_loader(dataset, config.batch)):
                _, loss, o = sess.run([train_step, cost, foutput],
                                   feed_dict={x: data, y_: labels, do:0.9})
                print('Batch : ', i + 1, '/', one_batch_size,
                      ', BCE in this minibatch: ', float(loss))
                avg_loss += float(loss)
            print('epoch:', epoch, ' train_loss:', float(avg_loss/one_batch_size), timeSince(start))
            #print("output", o);