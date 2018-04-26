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

from dataset import KinQueryDataset, preprocess

HAS_DATASET = False;
#emb_init = tf.truncated_normal_initializer(mean = 0.0, stddev= 0.01)

def label_one_hot(labels):
    one_hot = (np.arange(2) == labels[:]).astype(np.int32)
    return one_hot
	
def catFromOut(output):
    idx = []
    for i in output:
        ids = np.argmax(i, axis =0)
        idx.append(ids)
    idx = np.array(idx)
    print (idx.shape)
    return idx

def _batch_loader(iterable, n=1):
    """
    데이터를 배치 사이즈만큼 잘라서 보내주는 함수입니다. PyTorch의 DataLoader와 같은 역할을 합니다

    :param iterable: 데이터 list, 혹은 다른 포맷
    :param n: 배치 사이즈
    :return:
    """
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--batch', type=int, default=10)
    args.add_argument('--strmaxlen', type=int, default=100)
    args.add_argument('--embedding', type=int, default=8)
    args.add_argument('--threshold', type=float, default=0.5)
    config = args.parse_args()

    if not HAS_DATASET:  # It is not running on nsml
        DATASET_PATH = '../sample_data/kin/'

    # 모델의 specification
    input_size = config.embedding*config.strmaxlen
    output_size = 2
    hidden_layer_size = 200
    learning_rate = 0.001
    character_size = 251
    patch_size = 5
    depth = 64

#lo = tf.placeholder(tf.float32)

    x = tf.placeholder(tf.int32, [None, config.strmaxlen])
    y_ = tf.placeholder(tf.float32, [None, output_size])

    # 임베딩
    char_embedding = tf.get_variable('char_embedding', [character_size, config.embedding])
    embedded = tf.nn.embedding_lookup(char_embedding, x)
    print(embedded.shape)

    # 첫 번째 레이어
    first_layer_weight = weight_variable([patch_size, patch_size, 1, depth])
    first_layer_bias = bias_variable([depth])
    conv = tf.nn.conv2d(tf.reshape(embedded, (-1, config.strmaxlen, config.embedding, 1)), first_layer_weight, [1,1,1,1], padding = 'SAME')
    hidden_layer = tf.nn.relu(conv + first_layer_bias)
    pool = tf.nn.max_pool(hidden_layer, [1,2,2,1], [1,2,2,1], padding = 'SAME')

    # 두 번째 (아웃풋) 레이어
    second_layer_weight = weight_variable([patch_size, patch_size, depth, depth])
    second_layer_bias = bias_variable([depth])
    conv = tf.nn.conv2d(pool, second_layer_weight, [1,1,1,1], padding ='SAME')
    hidden_layer = tf.nn.relu(conv+ second_layer_bias)
    pool = tf.nn.max_pool (hidden_layer, [1,2,2,1],[1,2,2,1], padding = 'SAME')
    
    #output_sigmoid = tf.sigmoid(output)

    third_layer_weight = weight_variable([depth*config.strmaxlen//4*config.embedding//4, hidden_layer_size])
    third_layer_bias = bias_variable([hidden_layer_size])
    shape = pool.get_shape().as_list()
    print(shape)
    reshape = tf.reshape(pool, [tf.shape(x)[0], shape[1]*shape[2]*shape[3]])
    hidden_layer = tf.nn.relu(tf.matmul(reshape, third_layer_weight)+ third_layer_bias)

    fourth_layer_weight = weight_variable([hidden_layer_size, output_size])
    fourth_layer_bias = bias_variable([output_size])
    output = tf.matmul(hidden_layer, fourth_layer_weight) + fourth_layer_bias

    # loss와 optimizer
    logits = output
    #binary_cross_entropy = tf.reduce_mean(-(y_ * tf.log(output_sigmoid)) - (1-y_) * tf.log(1-output_sigmoid))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
	#binary_cross_entropy = tf.reduce_mean(-(y_ * tf.log(output_sigmoid)) - (1-y_) * tf.log(1-output_sigmoid))
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(learning_rate, global_step, 10000, 0.75)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)

    

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset = KinQueryDataset(DATASET_PATH, config.strmaxlen)
        dataset_len = len(dataset)
        one_batch_size = dataset_len//config.batch
        if dataset_len % config.batch != 0:
            one_batch_size += 1
        # epoch마다 학습을 수행합니다.
        for epoch in range(config.epochs):
            avg_loss = 0.0
            for i, (data, labels) in enumerate(_batch_loader(dataset, config.batch)):
                labels = label_one_hot (labels)
                print(data.shape)
                _, loss= sess.run([train_step, cost],
                                   feed_dict={x: data, y_: labels})
                print('Batch : ', i + 1, '/', one_batch_size,
                      ', BCE in this minibatch: ', float(loss))
                avg_loss += float(loss)
            print('epoch:', epoch, ' train_loss:', float(avg_loss/one_batch_size))
        output = logits.eval(feed_dict = {x:data, y_:labels})
        out_result = catFromOut(output)
        labels = catFromOut(labels)
        acc =0
        for i in range(109):
            if labels[i] == out_result[i]:
                acc+=1
        print("acc:", acc/109)
            

