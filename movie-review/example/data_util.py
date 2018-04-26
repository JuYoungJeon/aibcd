# -*- coding: utf-8 -*-
import os

import numpy as np

import logging
import os
import re

MARK_PAD = "<PAD>"
MARK_UNK = "<UNK>"

MARKS = [MARK_PAD, MARK_UNK]
ID_PAD = 0
ID_UNK = 1

def load_dict(dict_path, max_vocab=None):
    print("Try load dict from ",dict_path)
    try:
        dict_file = open(dict_path)
        dict_data = dict_file.readlines()
        dict_file.close()
    except:
        print("Load dict", dict_path, "failed, create later")
        return None

    dict_data = list(map(lambda x: x.split(), dict_data))
    if max_vocab:
        dict_data = list(filter(lambda x: int(x[0]) < max_vocab, dict_data))
    tok2id = dict(map(lambda x: (x[1], int(x[0])), dict_data))
    id2tok = dict(map(lambda x: (int(x[0]), x[1]), dict_data))
    logging.info(
        "Load dict %s with %d words."%(dict_path, len(tok2id)))
    return (tok2id, id2tok)

def create_dict(dict_path, corpus, max_vocab=None):
    print("Create dict %s."%(dict_path))
    counter = {}
    #print("replace",replace)
    for line in corpus:
        for word in line:
            try:
                counter[word] += 1
            except:
                counter[word] = 1
    #print("counter", counter)
    for mark_t in MARKS:
        if mark_t in counter:
            del counter[mark_t]
            print("%s appears in corpus."%(mark_t))
    counter = list(counter.items())
    counter.sort(key=lambda x: -x[1])
    #print("counter", counter)
    words = list(map(lambda x: x[0], counter))
    words = [MARK_PAD, MARK_UNK] + words
    if max_vocab:
        words = words[:max_vocab]

    tok2id = dict()
    id2tok = dict()
    with open(dict_path, 'w', encoding="utf-8") as dict_file:
        for idx, tok in enumerate(words):
            print(idx, tok, file=dict_file)
            tok2id[tok] = idx
            id2tok[idx] = tok

    print(id2tok)
#print(os.listdir(os.getcwd()))

    print("Create dict %s with %d words."%(dict_path, len(words)))
    return (tok2id, id2tok)

def corpus_map2id(data, tok2id):
#print(data)
#print(tok2id)
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
#print("tot", tot)
#print("unk", unk)
    return ret, (tot - unk) / tot


def sen_map2tok(sen, id2tok):
    return list(map(lambda x: id2tok[x], sen))


def load_data(data,
              doc_dict_path,
              max_doc_vocab=None):
    
    with open(data,'rt', encoding="utf-8") as docfile:
        docs = docfile.readlines()
    replace=[]
    for i in docs:
        replace.append(re.sub('(mv\d{2,8})?(ac\d{2,8})?(&#\d{2,5})?(�*)?', '', str(i)))
    docs = list(map(lambda x: x.split(), replace))
    

    doc_dict = load_dict(doc_dict_path, max_doc_vocab)
    if doc_dict is None:
        doc_dict = create_dict(doc_dict_path, docs, max_doc_vocab)

    docid, cover = corpus_map2id(docs, doc_dict[0])
    print("Doc dict covers %.2f%% words."%(cover*100))

    return docid, doc_dict

def corpus_preprocess(corpus):
    import re
    ret = []
    for line in corpus:
        x = re.sub('\\d', '#', line)
        ret.append(x)
    return ret

def sen_postprocess(sen):
    return sen

def load_test_data(doc, doc_dict):
    docs = corpus_preprocess(doc)

    print("Load %d testing documents."%(len(docs)))
    docs = list(map(lambda x: x.split(), docs))

    docid, cover = corpus_map2id(docs, doc_dict[0])
    print("Doc dict covers %.2f words."%(cover*100))

    return docid
