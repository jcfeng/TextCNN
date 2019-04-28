# -*- coding: utf-8 -*-
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8') #gb2312
#training the model.
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')
import tensorflow as tf
import numpy as np
from p7_TextCNN_model import TextCNN
#from data_util import create_vocabulary,load_data_multilabel
import pickle
import h5py
import os,re
import random
from numba import jit
import data_preprocess
import IO_class

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#configuration
FLAGS=tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("traning_data_path","/home/fengjingchao/Dataset/hotel_comment_utf-8/4000","path of traning data.") #../data/sample_multiple_label.txt
#tf.app.flags.DEFINE_integer("vocab_size",100000,"maximum vocab size.")

# tf.app.flags.DEFINE_string("cache_file_h5py","../data/ieee_zhihu_cup/data.h5","path of training/validation/test data.") #../data/sample_multiple_label.txt
# tf.app.flags.DEFINE_string("cache_file_pickle","../data/ieee_zhihu_cup/vocab_label.pik","path of vocabulary and label files") #../data/sample_multiple_label.txt

tf.app.flags.DEFINE_float("learning_rate",0.03,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.") #6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.") #0.65一次衰减多少
tf.app.flags.DEFINE_string("ckpt_dir","text_cnn_title_desc_checkpoint/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len",300,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",300,"embedding size")
tf.app.flags.DEFINE_boolean("is_training_flag",True,"is training.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",10,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")
tf.app.flags.DEFINE_integer("num_filters", 128, "number of filters") #256--->512
tf.app.flags.DEFINE_string("word2vec_model_path","/home/fengjingchao/Chinese_word_vector/sgns.wiki.bigram","word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_string("name_scope","cnn","name scope value.")
tf.app.flags.DEFINE_boolean("multi_label_flag",False,"use multi label or single label.")
# filter_sizes=[6,7,8]
filter_sizes=[6,7,8]

#1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
def main(_):
    print_configuration_op(FLAGS,filter_sizes) ###打印所有参数
    #trainX, trainY, testX, testY = None, None, None, None
    #vocabulary_word2index, vocabulary_index2word, vocabulary_label2index, _= create_vocabulary(FLAGS.traning_data_path,FLAGS.vocab_size,name_scope=FLAGS.name_scope)
    # word2index, label2index, trainX, trainY, vaildX, vaildY, testX, testY=load_data(FLAGS.cache_file_h5py, FLAGS.cache_file_pickle)
    trainX, trainY, testX, testY,label2index,word2index = data_preprocess.data_preprocess(FLAGS.traning_data_path,0.7,FLAGS.sentence_len)

    vocab_size = len(word2index)
    print("cnn_model.vocab_size:",vocab_size)
    num_classes=len(label2index)
    print("num_classes:",num_classes)
    print "trainX.shape:",trainX.shape
    num_examples,FLAGS.sentence_len=trainX.shape
    print("num_examples of training:",num_examples,";sentence_len:",FLAGS.sentence_len)
    print "num of labels:",len(label2index)
    from collections import Counter
    print "train set label dirtribute",Counter(trainY)
    print "test set label dirtribute", Counter(testY)

    #train, test= load_data_multilabel(FLAGS.traning_data_path,vocabulary_word2index, vocabulary_label2index,FLAGS.sentence_len)
    #trainX, trainY = train;testX, testY = test
    #print some message for debug purpose
    # print("trainX[0:10]:", trainX[0:10])
    # print("trainY[0]:", trainY[0:10])
    # print("train_y_short:", trainY[0])

    #2.create session.
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        #Instantiate Model
        textCNN=TextCNN(filter_sizes,FLAGS.num_filters,num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                        FLAGS.decay_rate,FLAGS.sentence_len,vocab_size,FLAGS.embed_size,multi_label_flag=FLAGS.multi_label_flag)
        #Initialize Save
        saver=tf.train.Saver()
        # if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
        if False:
            print("Restoring Variables from Checkpoint.")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
            for i in range(3): #decay learning rate if necessary.
               print(i,"Going to decay learning rate by half.")
               sess.run(textCNN.learning_rate_decay_half_op)
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding: #load pre-trained word embedding
                index2word={v:k for k,v in word2index.items()}
                assign_pretrained_word_embedding(sess, index2word, vocab_size, textCNN,FLAGS.word2vec_model_path)
        curr_epoch=sess.run(textCNN.epoch_step)
        #3.feed data & training
        number_of_training_data=len(trainX)
        batch_size=FLAGS.batch_size
        iteration=0
        for epoch in range(curr_epoch,FLAGS.num_epochs):
            loss, counter =  0.0, 0
            for batch_x,batch_y in nextBatch(trainX,trainY,batch_size):
                from collections import Counter
                c= Counter(batch_y)
                print "cur batch lables:",c
                iteration=iteration+1
                # if epoch==0 and counter==0:
                    # print("trainX[start:end]:",trainX[start:end])
                feed_dict = {textCNN.input_x: batch_x,textCNN.dropout_keep_prob: 0.8,textCNN.is_training_flag:FLAGS.is_training_flag}
                if not FLAGS.multi_label_flag:
                    feed_dict[textCNN.input_y] = batch_y
                else:
                    feed_dict[textCNN.input_y_multilabel]=batch_y
                curr_loss,lr,_=sess.run([textCNN.loss_val,textCNN.learning_rate,textCNN.train_op],feed_dict)
                loss,counter=loss+curr_loss,counter+1
                if counter %16==0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tLearning rate:%.5f" %(epoch,counter,loss/float(counter),lr))

                ########################################################################################################
                # if start%(3000*FLAGS.batch_size)==0: # eval every 3000 steps.
                #     eval_loss, f1_score = do_eval(sess, textCNN, testX, testY,num_classes)
                #     print("Epoch %d Validation Loss:%.3f\tF1 Score:%.3f" % (epoch, eval_loss, f1_score))
                #     # save model to checkpoint
                #     save_path = FLAGS.ckpt_dir + "model.ckpt"
                #     print("Going to save model..")
                #     saver.save(sess, save_path, global_step=epoch)
                ########################################################################################################
            #epoch increment
            print("going to increment epoch counter....")
            sess.run(textCNN.epoch_increment)

            # 4.validation
            print(epoch,FLAGS.validate_every,(epoch % FLAGS.validate_every==0))
            if epoch % FLAGS.validate_every==0:
                eval_loss,f1_score=do_eval(sess,textCNN,testX,testY,num_classes)
                print("Epoch %d Validation Loss:%.3f\tF1 Score:%.3f" % (epoch,eval_loss,f1_score))
                #save model to checkpoint
                save_path=FLAGS.ckpt_dir+"model.ckpt"
                saver.save(sess,save_path,global_step=epoch)

        # 5.最后在测试集上做测试，并报告测试准确率 Test
        test_loss,f1_score = do_eval(sess, textCNN, testX, testY,num_classes)
        print("Test Loss:%.3f\tF1 Score:%.3f\t" % ( test_loss,f1_score))
    pass


# 输出batch数据集

def nextBatch(x, y, batchSize):
    """
    生成batch数据集，用生成器的方式输出
    """
    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = np.array(y)[perm]

    numBatches = len(x) // batchSize

    for i in range(numBatches):
        start = i * batchSize
        end = start + batchSize
        batchX = np.array(x[start: end], dtype="int64")
        batchY = np.array(y[start: end], dtype="float32")

        yield batchX, batchY
# 在验证集上做验证，报告损失、精确度
def do_eval(sess, textCNN, evalX, evalY, num_classes):
    # evalX = evalX[0:3000]
    # evalY = evalY[0:3000]
    from collections import Counter
    print Counter(evalY)
    number_examples = len(evalX)
    eval_loss, eval_counter, eval_f1_score, eval_p, eval_r = 0.0, 0, 0.0, 0.0, 0.0
    batch_size = FLAGS.batch_size
    predict = []

    # for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
    #     ''' evaluation in one batch '''
    #     # print "start:",start," end:",end
    #     # feed_dict = {textCNN.input_x: evalX[start:end], textCNN.input_y_multilabel: evalY[start:end], textCNN.dropout_keep_prob: 1.0,
    #     #              textCNN.is_training_flag: False}
    #     feed_dict = {textCNN.input_x: evalX[start:end], textCNN.input_y: evalY[start:end],
    #                  textCNN.dropout_keep_prob: 1.0,
    #                  textCNN.is_training_flag: False}
    correct_prediction = 0
    for batch_x,batch_y in nextBatch(evalX,evalY,batch_size):
        ''' evaluation in one batch '''
        feed_dict = {textCNN.input_x: batch_x, textCNN.input_y:batch_y,
                         textCNN.dropout_keep_prob: 1.0,
                         textCNN.is_training_flag: False}
        current_eval_loss, logits,prediction,correct= sess.run(
            [textCNN.loss_val, textCNN.logits,textCNN.predictions,textCNN.correct_prediction], feed_dict)
        # print "logits shape:",logits.shape
        # print logits
        # predict.append(l for l in logits[:,0])
        [predict.append(p) for p in prediction]

        # [correct_prediction=correct_prediction+1 for c in correct if c]
        correct_prediction = Counter(correct)[True]+correct_prediction
        eval_loss += current_eval_loss
        eval_counter += 1

    # if not FLAGS.multi_label_flag:
    #     print predict
    #     predict = [int(ii > 0.5) for ii in predict]
    # _, _, f1_macro, f1_micro, _ = fastF1(predict, evalY)
    # f1_score = (f1_micro+f1_macro)/2.0
    print "accuracy:",correct_prediction/float(len(predict))
    f1_score = get_class_f1(predict,evalY[:len(predict)])
    # return sum(f1_score.values())/float(len(f1_score))
    return eval_loss/float(eval_counter), sum(f1_score.values())/float(len(f1_score))

# @jit
# def fastF1(result, predict):
#     ''' f1 score '''
#     true_total, r_total, p_total, p, r = 0, 0, 0, 0, 0
#     total_list = []
#     for trueValue in range(6):
#         trueNum, recallNum, precisionNum = 0, 0, 0
#         for index, values in enumerate(result):
#             if values == trueValue:
#                 recallNum += 1
#                 if values == predict[index]:
#                     trueNum += 1
#             if predict[index] == trueValue:
#                 precisionNum += 1
#         R = trueNum / recallNum if recallNum else 0
#         P = trueNum / precisionNum if precisionNum else 0
#         true_total += trueNum
#         r_total += recallNum
#         p_total += precisionNum
#         p += P
#         r += R
#         f1 = (2 * P * R) / (P + R) if (P + R) else 0
#         # print(id2rela[trueValue], P, R, f1)
#         total_list.append([P, R, f1])
#     p /= 6
#     r /= 6
#     micro_r = true_total / r_total
#     micro_p = true_total / p_total
#     macro_f1 = (2 * p * r) / (p + r) if (p + r) else 0
#     micro_f1 = (2 * micro_p * micro_r) / (micro_p +
#                                           micro_r) if (micro_p + micro_r) else 0
#     print('P: {:.2f}%, R: {:.2f}%, Micro_f1: {:.2f}%, Macro_f1: {:.2f}%'.format(
#         p*100, r*100, micro_f1 * 100, macro_f1*100))
#     return p, r, macro_f1, micro_f1, total_list
def get_class_f1(result,label):
    """
    计算各个类别的f1值
    :param result:
    :param label:
    :return:
    """
    from collections import Counter
    counter = Counter(label)
    class_label = list(counter)
    f1_score={}
    for c in class_label:
        tp=0
        fp=0
        fn=0
        for i in range(len(label)):
            if label[i]==c and label[i]==result[i]:
                tp = tp+1
            elif result[i]==c and label[i]!=c:
                fp=fp+1
            elif label[i]==c and result[i]!=c:
                fn = fn+1
        if tp==0:
            f1_score[c] = 0
            print "class:", c, "  precious:", 0, " recall:", 0, " f1:", 0
            continue
        p = tp/float(tp+fp)
        r = tp/float(tp+fn)
        f1 = 2*p*r/(p+r)
        print "class:",c,"  precious:",p," recall:",r," f1:",f1
        f1_score[c] =f1
    return f1_score




def assign_pretrained_word_embedding(sess,vocabulary_index2word,vocab_size,textCNN,word2vec_model_path):
    """
    给词表的embedding赋值，
    :param sess:
    :param vocabulary_index2word:
    :param vocab_size:
    :param textCNN:
    :param word2vec_model_path:
    :return:
    """

    print("using pre-trained word emebedding.started.word2vec_model_path:",word2vec_model_path)

    word2vec_dict = load_word2vec_dict(word2vec_model_path)
    print "embedding dim:",FLAGS.embed_size
    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.存储对应id的单词的embedding
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0
    count_not_exist = 0
    for i in range(1, vocab_size):  # loop each word. notice that the first two words are pad and unknown token
        word = vocabulary_index2word[i]  # get a word
        embedding = None
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size)
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(textCNN.Embedding,word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding)
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")

def load_word2vec_dict(word2vec_model_path):

    word2vec_dict = {}
    lines =(IO_class.IO_class()).read_text(word2vec_model_path)
    param = re.split(r'\s+',lines[0])
    # tf.app.flags.DEFINE_integer("embed_size", int(param[1]), "embedding size")
    for line in lines[1:]:
        line = re.split(r'\s+',line)
        line = line[:-1]
        word2vec_dict[line[0]] = [float(n) for n in line[1:]]

    import  word2vec  # we put import here so that many people who do not use word2vec do not need to install this package. you can move import to the beginning of this file.
    # word2vec_model = word2vec.load(word2vec_model_path, kind='bin')
    # word2vec_dict = {}
    # for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
    #     word2vec_dict[word] = vector
    return word2vec_dict


def load_data(cache_file_h5py,cache_file_pickle):
    """
    load data from h5py and pickle cache files, which is generate by take step by step of pre-processing.ipynb
    :param cache_file_h5py:
    :param cache_file_pickle:
    :return:
    """
    if not os.path.exists(cache_file_h5py) or not os.path.exists(cache_file_pickle):
        raise RuntimeError("############################ERROR##############################\n. "
                           "please download cache file, it include training data and vocabulary & labels. "
                           "link can be found in README.md\n download zip file, unzip it, then put cache files as FLAGS."
                           "cache_file_h5py and FLAGS.cache_file_pickle suggested location.")
    print("INFO. cache file exists. going to load cache file")
    f_data = h5py.File(cache_file_h5py, 'r')
    print("f_data.keys:",list(f_data.keys()))
    train_X=f_data['train_X'] # np.array(
    print("train_X.shape:",train_X.shape)
    train_Y=f_data['train_Y'] # np.array(
    print("train_Y.shape:",train_Y.shape,";")
    vaild_X=f_data['vaild_X'] # np.array(
    valid_Y=f_data['valid_Y'] # np.array(
    test_X=f_data['test_X'] # np.array(
    test_Y=f_data['test_Y'] # np.array(
    #print(train_X)
    #f_data.close()

    word2index, label2index=None,None
    with open(cache_file_pickle, 'rb') as data_f_pickle:
        word2index, label2index=pickle.load(data_f_pickle)
    print("INFO. cache file load successful...")
    return word2index, label2index,train_X,train_Y,vaild_X,valid_Y,test_X,test_Y

def print_configuration_op(FLAGS,filter_sizes):
    """
    打印所有参数
    :param FLAGS:
    :param filter_sizes:
    :return:
    """
    print('My Configurations:')
    #pdb.set_trace()
    for name, value in FLAGS.__flags.items():
        value=value.value
        if type(value) == float:
            print(' %s:\t %f'%(name, value))
        elif type(value) == int:
            print(' %s:\t %d'%(name, value))
        elif type(value) == str:
            print(' %s:\t %s'%(name, value))
        else:
            print('%s:\t %s' % (name, value))
    print "filter size:",filter_sizes
    #for k, v in sorted(FLAGS.__dict__.items()):
        #print(f'{k}={v}\n')


if __name__ == "__main__":
    tf.app.run()
