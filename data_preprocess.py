# !/usr/bin/python
# -*-coding:utf-8-*-
import sys
sys.path.append("..")
import os
import h5py
import pickle
from Text_preprocess import Segment_tokens
from IO_class import *
from random import shuffle
import numpy as np

def save_data(cache_file_h5py,cache_file_pickle,word2index,label2index,train_X,train_Y,vaild_X,valid_Y,test_X,test_Y):
    """
    save data to path
    :param cache_file_h5py:
    :param cache_file_pickle:
    :param word2index:dict{char/word:index}
    :param label2index:dict{labe:index}
    :param train_X:
    :param train_Y:
    :param vaild_X:
    :param valid_Y:
    :param test_X:
    :param test_Y:
    :return:
    """
    # train/valid/test data using h5py
    f = h5py.File(cache_file_h5py, 'w')
    f['train_X'] = train_X
    f['train_Y'] = train_Y
    f['vaild_X'] = vaild_X
    f['valid_Y'] = valid_Y
    f['test_X'] = test_X
    f['test_Y'] = test_Y
    f.close()
    # save word2index, label2index
    with open(cache_file_pickle, 'ab') as target_file:
        pickle.dump((word2index,label2index), target_file)

def load_data(corpus_path,train_percent,max_length,stop_path=None):
    """
    训练数据
    :param corpus_path:
    :param train_percent:
    :param stop_path:
    :return:
    """
    if stop_path != None and len(stop_path.strip()) > 0:
        lines = (IO_class()).read_texts(stop_path)
        stop_words = [word.strip('\r\n') for word in lines]
    sub_dirs = os.listdir(corpus_path)

    train_corpus =[]
    train_label = []
    test_corpus = []
    test_label = []

    label_idx = 0
    label_dict = {dir_name:idx for idx,dir_name in enumerate(sub_dirs)}


    segment_token = Segment_tokens()
    segment_token.word2index["PAD"]=0
    print "init size:",len(segment_token.word2index)
    for dir in sub_dirs:
        # 读取数据
        dir_path = os.path.join(corpus_path, dir)
        texts = []
        if os.path.isdir(dir_path):
            files = os.listdir(dir_path)
            [texts.append("".join((IO_class()).read_text(os.path.join( dir_path,filename)))) for filename in files]
        else:
            texts = (IO_class()).read_text(dir_path)
        print "load data:",dir,"  num:",len(texts)

        # 过滤长度大于最大长度的
        def longer_than_max(line):
            return len(line)<=max_length
        texts = filter(longer_than_max,texts)

        seg_texts = [segment_token.ngram_line(line,[2]) for line in texts]
        shuffle(seg_texts)
        [train_corpus.append(t) for t in seg_texts[:int(len(seg_texts)*train_percent)]]
        [train_label .append(label_dict[dir]) for t in range(int(len(seg_texts)*train_percent))]
        [test_corpus.append(t) for t in seg_texts[int(len(seg_texts) * train_percent):]]
        [test_label.append(label_dict[dir]) for t in range(len(seg_texts)-int(len(seg_texts)*train_percent))]
    print "data size,train:",len(train_corpus)," test:",len(test_corpus)
    avg_len = (sum([len(s) for s in train_corpus])+sum([len(s) for s in test_corpus]))/float(len(test_corpus)+len(train_corpus))
    print "average length",avg_len
    longer = 0
    for s in train_corpus,test_corpus:
        if len(s)>avg_len:
            longer = longer+1
    print "num of longer then avg length:",longer
    return train_corpus,train_label,test_corpus,test_label,label_dict,segment_token.word2index

def data_preprocess(corpus_path,train_percent,max_length,stop_path=None):
    train_corpus, train_label, test_corpus, test_label, label_dict,word2index =load_data(corpus_path,train_percent,max_length,stop_path)

    length = max([len(sen) for sen in train_corpus,test_corpus])
    # for sen in train_corpus, test_corpus:
    #     if len(sen)==length:
    #         print "".join(sen)
    print "max sentence length:",length

    train_corpus = pad_sentence(train_corpus,'PAD',max_length)
    test_corpus = pad_sentence(test_corpus,'PAD',max_length)
    # index2word = {v:k for k,v in word2index.items()}
    ##先将数据转化为对应的idx
    train_corpus_idxs = [[word2index[w] for w in line]for line in train_corpus ]
    test_corpus_idxs = [[word2index[w] for w in line] for line in test_corpus]
    train_corpus_idxs = np.array(train_corpus_idxs)
    test_corpus_idxs = np.array(test_corpus_idxs)
    # for word,i in word2index.items():
    #     print str(i)+" "+word
    return train_corpus_idxs, train_label, test_corpus_idxs, test_label, label_dict,word2index

def pad_sentence(sentence_list,pad_word,max_len):
    """
    pad sentence to max_len using pad_word
    :param sentence_list:
    :param pad_word:
    :param max_len:
    :return:
    """
    for s in range(len(sentence_list)):
        while len(sentence_list[s])<max_len:
            sentence_list[s].append(pad_word)
    return sentence_list

