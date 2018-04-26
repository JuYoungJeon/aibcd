# -*- coding: utf-8 -*-
import os

import numpy as np

import logging
import os
import re
import operator

MARK_PAD = "<PAD>"
MARK_UNK = "<UNK>"

MARKS = [MARK_PAD, MARK_UNK]
ID_PAD = 0
ID_UNK = 1

def create_dict(corpus, max_vocab=None):
    print("Create dict")
    counter = {}

    for line in corpus:
        for word in line:
            try:
                counter[word] += 1
            except:
                counter[word] = 1

    for mark_t in MARKS:
        if mark_t in counter:
            del counter[mark_t]
            print("%s appears in corpus."%(mark_t))
    counter = sorted(counter.items(), key=operator.itemgetter(0))
    counter.sort(key=lambda x: -x[1])

    words = list(map(lambda x: x[0], counter))
    words = [MARK_PAD, MARK_UNK] + words
    if max_vocab:
        words = words[:max_vocab]

    tok2id = dict()
    id2tok = dict()
    for idx, tok in enumerate(words):
        tok2id[tok] = idx
        id2tok[idx] = tok

    print(id2tok)

    print("Create dict with %d words."%(len(words)))
    return words, (tok2id, id2tok)

def corpus_map2id(data, tok2id):
    ret = []
    unk = 0
    tot = 0
    for doc in data:
        tmp = []
        for word in doc:
            tot += 1
            try:
                tmp.append(tok2id[word])
            except:
                tmp.append(ID_UNK)
                unk += 1
        ret.append(tmp)

    return ret, (tot - unk) / tot


def sen_map2tok(sen, id2tok):
    return list(map(lambda x: id2tok[x], sen))


def load_data_create_dic(data, max_doc_vocab=None):
    
    with open(data,'rt', encoding="utf-8") as docfile:
        docs = docfile.readlines()
        
    docs = corpus_preprocess(docs)
    docs = list(map(lambda x: x.split(), docs))

    words, doc_dict = create_dict(docs, max_doc_vocab)

    return words, doc_dict

def word_to_vec(data, max_doc_vocab):
    data = corpus_preprocess(data)
    docs = list(map(lambda x: x.split(), data))
    words, doc_dict = create_dict(docs, max_doc_vocab)
    
    docid, cover = corpus_map2id(docs, doc_dict[0])
    print("Doc dict covers %.2f%% words."%(cover*100))
    return docid

def corpus_preprocess(corpus):
    import re
    ret = []
    for line in corpus:
        x = re.sub('(mv\d{2,8})?(ac\d{2,8})?(&#\d{2,5})?(�*)?(다.)?(영화)?(너무)?', '', str(line))
        ret.append(x)
    return ret

def sen_postprocess(sen):
    return sen

def load_test_data(doc, doc_dict):
    print("load_test_data")
    docs = corpus_preprocess(doc)

    print("Load %d testing documents."%(len(docs)))
    docs = list(map(lambda x: x.split(), docs))

    docid, cover = corpus_map2id(docs, doc_dict)
    print("Doc dict covers %.2f words."%(cover*100))

    return docid
