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

import nsml
import ndata_util
import hi

from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML
from kdataset import KinQueryDataset, preprocess, test_preprocess

# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(sess, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *args):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir_name, 'model'))

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *args):
        saver = tf.train.Saver()
        # find checkpoint
        ckpt = tf.train.get_checkpoint_state(dir_name)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(dir_name, checkpoint))
        else:
            raise NotImplemented('No checkpoint!')
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """
        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        contents = dic.eval()
        wordtoix, ixtoword = hi.make_dict_all_cut(contents, minlength=0, maxlength=3, jamo_delete=True)
        preprocessed_data = make_inputs(raw_data, wordtoix, config.strmaxlen,shuffle=False)

        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        pred = sess.run(foutput, feed_dict={x: queries, do:1.0})
        if np.any((np.logical_or(pred<0, pred>10))):
            pred[np.where(pred<0)] = 0
            pred[np.where(pred>10)] = 10
        point = pred.flatten()
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 확률의 값은 영향을 미치지 않습니다
        return list(zip(np.zeros(len(point)), point))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)



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

def dic_variable():
    data_path = os.path.join(DATASET_PATH, 'train', 'train_data')
    contents = hi.loading_data(data_path, eng=False, num=False, punc=False)
    return tf.Variable(contents)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--batch', type=int, default=2000)
    args.add_argument('--strmaxlen', type=int, default=50)
    args.add_argument('--embedding', type=int, default=100)
    args.add_argument('--maxvocabsize', type=int, default=50000)
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/movie_review/'

    # 모델의 specification
    input_size = config.embedding*config.strmaxlen
    output_size = 1
    hidden_layer_size = 400
    learning_rate = 0.001
    #character_size = 3000
    dic = dic_variable()

    x = tf.placeholder(tf.int32, [None, config.strmaxlen])
    y_ = tf.placeholder(tf.float32, [None, output_size])
    do = tf.placeholder(tf.float32)
    
    # 임베딩
    char_embedding = tf.get_variable('char_embedding', [config.maxvocabsize, config.embedding])
    embedded = tf.nn.embedding_lookup(char_embedding, x)

    # 첫 번째 레이어
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer_size)
    #cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=do)
    #cell2 = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer_size)
    #cell2 = tf.nn.rnn_cell.DropoutWrapper(cell2, output_keep_prob=do)
    #multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell, cell2])
    
    output, state = tf.nn.dynamic_rnn(cell, embedded, dtype=tf.float32)
    output = tf.transpose(output, [1, 0, 2])
    output = output[-1]

    # 두 번째 (아웃풋) 레이어
    second_layer_weight = weight_variable([hidden_layer_size, output_size])
    second_layer_bias = bias_variable([output_size])
    foutput = tf.matmul(output, second_layer_weight) + second_layer_bias
    #output_sigmoid = tf.sigmoid(output)

    # loss와 optimizer
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y_))
    global_step = tf.Variable(0)
    #learning_rate= tf.train.exponential_decay(learning_rate, global_step, 10000, 0.75)
    cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=foutput))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    # DONOTCHANGE: Reserved for nsml
    bind_model(sess=sess, config=config)
    
    # DONOTCHANGE: Reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

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
#        print("train",os.listdir(os.getcwd()))
        
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
                _, loss = sess.run([train_step, cost],
                                   feed_dict={x: data, y_: labels, do:0.9})
#print('Batch : ', i + 1, '/', one_batch_size,
#                      ', MSE in this minibatch: ', float(loss))
                avg_loss += float(loss)
            print('epoch:', epoch, ' train_loss:', float(avg_loss/one_batch_size),timeSince(start))
            nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=float(avg_loss/one_batch_size), step=epoch)
            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.3, 0), (0.7, 1), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            queries = f.readlines()
        res = []
        for batch in _batch_loader(queries, config.batch):
            temp_res = nsml.infer(batch)
            res += temp_res
        print(res)



