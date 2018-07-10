# -*- coding: utf-8 -*-
import copy
import os

from gensim.models import KeyedVectors
from keras import regularizers
from keras.engine.topology import Layer
from keras.layers import (
    Input, Embedding, Lambda, dot, Dense, concatenate, SimpleRNN, LSTM, GRU
)
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from numpy import vstack, zeros
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
import click
import joblib
import keras.backend as K
import numpy as np
import pandas as pd
import tensorflow as tf

EMBEDDING_CHAR_INPUT_DIM = 3048
EMBEDDING_WORD_INPUT_DIM = 20890  # pretrained word2vec 多一个没用上的？
EMBEDDING_OUTPUT_DIM = 300
VALIDATION_SPLIT = 0.2  # 验证集大小，全部数据的20%

TRAIN_PATH = 'data/train2.csv'
AUGMENT_PATH = 'data/augment.csv'
DEV_PATH = 'data/dev.csv'
TEST_PATH = 'data/test.csv'
QUESTION_PATH = 'data/question.csv'
OPTIMIZER = ['sgd', 'adam', 'adagrad', 'rmsprop']

qes = pd.read_csv(QUESTION_PATH)
train = pd.read_csv(TRAIN_PATH)
dev = pd.read_csv(DEV_PATH)
# augment = pd.read_csv(AUGMENT_PATH)  # 指定参数再 read/load
test = pd.read_csv(TEST_PATH)
max_wlen = 30  # 具体分布见 check_corpus
max_clen = 45


def get_ids(qids):
    ids = []
    for t_ in qids:
        ids.append(int(t_[1:]))
    return np.asarray(ids)


def pad(seq, max_len):
    n = len(seq)
    if n > max_len:
        return np.asarray(seq[:max_len])
    elif n < max_len:
        return np.asarray(seq + [0] * (max_len - n))
    else:
        return np.asarray(seq)


def get_texts(file):
    q1id, q2id = file['q1'], file['q2']
    id1s, id2s = get_ids(q1id), get_ids(q2id)
    texts = []
    for t_ in zip(id1s, id2s):
        texts.append([
            pad([int(item[1:]) for item in qes['words'][t_[0]].split()], max_wlen),
            pad([int(item[1:]) for item in qes['words'][t_[1]].split()], max_wlen),
            pad([int(item[1:]) for item in qes['chars'][t_[0]].split()], max_clen),
            pad([int(item[1:]) for item in qes['chars'][t_[1]].split()], max_clen)
        ])
    return texts


train_texts = get_texts(train)
dev_texts = get_texts(dev)
# augment_texts = get_texts(augment)  # 指定参数再 read/load
test_texts = get_texts(test)

# print('Split dev from train...')
# x_train, x_dev, y_train, y_dev = train_test_split(
#     train_texts, train['label'], test_size=VALIDATION_SPLIT, shuffle=False)


def mask_aware_mean(x):
    # recreate the masks - all zero rows have been masked
    mask = K.not_equal(K.sum(K.abs(x), axis=2, keepdims=True), 0)

    # number of that rows are not all zeros
    n = K.sum(K.cast(mask, 'float32'), axis=1, keepdims=False)

    # compute mask-aware mean of x
    x_mean = K.sum(x, axis=1, keepdims=False) / n

    return x_mean


class ZeroMaskedEntries(Layer):
    """
    This layer is called after an Embedding layer.
    It zeros out all of the masked-out embeddings.
    It also swallows the mask without passing it on.
    You can change this to default pass-on behavior as follows:

    def compute_mask(self, x, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(x, 0)
    """

    def __init__(self, **kwargs):
        self.support_mask = True
        super(ZeroMaskedEntries, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[1]
        self.repeat_dim = input_shape[2]

    def call(self, x, mask=None):
        mask = K.cast(mask, 'float32')
        mask = K.repeat(mask, self.repeat_dim)
        mask = K.permute_dimensions(mask, (0, 2, 1))
        return x * mask

    def compute_mask(self, input_shape, input_mask=None):
        return None


class WeightedSumEmbedding(Layer):
    def __init__(self, sentence_length, **kwargs):
        self.sentence_length = sentence_length
        super(WeightedSumEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='WeightedSumEmbedding', shape=(self.sentence_length, 1),
            initializer='uniform', trainable=True)
        super(WeightedSumEmbedding, self).build(input_shape)

    def call(self, x, mask=None):
        return K.batch_flatten(K.dot(K.permute_dimensions(x, (0, 2, 1)), self.kernel))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def compute_mask(self, input_shape, input_mask=None):
        return None


def mask_aware_mean_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3
    return (shape[0], shape[2])


def weighted_sum(inputs):
    # inputs[1].shape = (batch, 30) inputs[0].shape = (batch, 30, 300)
    # import ipdb; ipdb.set_trace();
    output = K.batch_flatten(K.batch_dot(K.expand_dims(inputs[1], axis=1), inputs[0]))
    return output


def format(item):
    x = list(zip(*item))
    return [np.asmatrix(x[0]), np.asmatrix(x[1]), np.asmatrix(x[2]), np.asmatrix(x[3])]


def np_svd_remove(np_value):
    import ipdb; ipdb.set_trace()
    x = np.stack(np_value)
    _, s, _ = scipy.sparse.linalg.svds(x, k=1)
    u = s[0][0]
    c = 1 - u * u
    np_value[0] *= c
    np_value[1] *= c
    return np_value


def make_submission(predict_prob):
    with open('submission.csv', 'w') as file:
        file.write(str('y_pre') + '\n')
        for line in predict_prob:
            file.write(str(line) + '\n')
    file.close()


@click.command()
@click.option('--train_batch_size', type=int, default=512, help='Train batch_size')
@click.option('--test_batch_size', type=int, default=128, help='Train batch_size')
@click.option('-e', '--epoch', type=int, default=10, help='Train epochs num')
@click.option('-r', '--regularizer', type=float, default=1e-6,
              help='regularizers l2 coefficient')
@click.option('-o', '--optimizer', type=click.Choice(OPTIMIZER), default="rmsprop",
              help='optimizer function')
@click.option('--augmentation', flag_value=True, default=False,
              help='whether use graph to augment train data')
@click.option('--noval', flag_value=True, default=False,
              help='whether use all data to train (merge val into train)')
def func1(train_batch_size, test_batch_size, epoch,
          regularizer, optimizer, augmentation, noval):
    '''
    trainable embedding average
    '''
    x_train = train_texts
    y_train = list(train['label'])
    if noval:
        x_train += dev_texts
        y_train += list(dev['label'])
    if augmentation:
        augment = pd.read_csv(AUGMENT_PATH)
        x_train += get_texts(augment)
        y_train += list(augment['label'])

    model_w2v = KeyedVectors.load_word2vec_format('data/word_embed2.txt')
    weights_w = model_w2v.vectors  # get_keras_embedding 只支持 trainable 参数
    weights_w = vstack((zeros((1, EMBEDDING_OUTPUT_DIM)), weights_w))  # 多加一行
    init_weights_w = copy.deepcopy(weights_w)
    model_c2v = KeyedVectors.load_word2vec_format('data/char_embed2.txt')
    weights_c = model_c2v.vectors  # get_keras_embedding 只支持 trainable 参数
    weights_c = vstack((zeros((1, EMBEDDING_OUTPUT_DIM)), weights_c))  # 多加一行
    init_weights_c = copy.deepcopy(weights_c)

    def embed_reg_w(weight_matrix):  # 加 embeding 变化的惩罚项，系数取 0.001 loss 就降不下去了
        return K.sum(regularizer * K.square(weight_matrix - init_weights_w))

    def embed_reg_c(weight_matrix):  # 加 embeding 变化的惩罚项，系数取 0.001 loss 就降不下去了
        return K.sum(regularizer * K.square(weight_matrix - init_weights_c))

    embedding_w = Embedding(
        input_dim=weights_w.shape[0], output_dim=EMBEDDING_OUTPUT_DIM,
        weights=[weights_w], trainable=True, mask_zero=True,  # 前面 pad 0 to max_len
        embeddings_regularizer=embed_reg_w if regularizer else None
    )
    embedding_c = Embedding(
        input_dim=weights_c.shape[0], output_dim=EMBEDDING_OUTPUT_DIM,
        weights=[weights_c], trainable=True, mask_zero=True,  # 前面 pad 0 to max_len
        embeddings_regularizer=embed_reg_c if regularizer else None
    )
    sw1, sw2, sc1, sc2 = \
        Input(name='sw1', shape=(max_wlen,)), Input(name='sw2', shape=(max_wlen,)), \
        Input(name='sc1', shape=(max_clen,)), Input(name='sc2', shape=(max_clen,))

    zeroed = ZeroMaskedEntries()
    lambda_mean = Lambda(mask_aware_mean, mask_aware_mean_output_shape)
    aver_w1 = lambda_mean(zeroed(embedding_w(sw1)))
    aver_w2 = lambda_mean(zeroed(embedding_w(sw2)))
    aver_c1 = lambda_mean(zeroed(embedding_c(sc1)))
    aver_c2 = lambda_mean(zeroed(embedding_c(sc2)))

    # subtracted = subtract([aver_w1, aver_w2])  # subtracted + dense is not similarity!
    # label = Dense(1, activation='sigmoid', name='label')(subtracted)
    label_w = dot([aver_w1, aver_w2], 1, normalize=True)  # 不 normalize 效果就不行
    label_c = dot([aver_c1, aver_c2], 1, normalize=True)
    dense = Dense(1, name='dense',
                  activation='sigmoid',  # 引入非线性？
                  kernel_regularizer=regularizers.l2(regularizer) if regularizer else None,
                  bias_regularizer=regularizers.l2(regularizer) if regularizer else None)
    label = dense(concatenate([label_w, label_c]))
    model = Model(inputs=[sw1, sw2, sc1, sc2], outputs=label)  # 多输入，单输出
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    model.fit(
        format(x_train), y_train,
        callbacks=[ModelCheckpoint('func1.{epoch:02d}.model', save_best_only=True)],
        batch_size=train_batch_size, epochs=epoch,
        validation_data=(format(dev_texts), list(dev['label'])))
    model.save('./func1.model')
    y_test = model.predict(format(test_texts), batch_size=test_batch_size, verbose=1)
    y_test = [item[0] for item in y_test]
    make_submission(y_test)
    return


@click.command()
@click.option('--train_batch_size', type=int, default=512, help='Train batch_size')
@click.option('--model_path', type=str, default='func1.model', help='reload model file path')
@click.option('-e', '--epoch', type=int, default=10, help='Train epochs num')
@click.option('--noval', flag_value=True, default=False,
              help='whether use all data to train (merge val into train)')
@click.option('--augmentation', flag_value=True, default=False,
              help='whether use graph to augment train data')
def retrain(train_batch_size, model_path, epoch, noval, augmentation):
    print('retrain\nload model...')
    model = load_model(model_path)
    x_train = train_texts
    y_train = list(train['label'])
    if noval:
        x_train += dev_texts
        y_train += list(dev['label'])
    if augmentation:
        augment = pd.read_csv(AUGMENT_PATH)
        augment_texts = get_texts(augment)
        x_train += augment_texts
        y_train += list(augment['label'])
    model.fit(
        format(x_train), y_train,
        callbacks=[ModelCheckpoint('func_retrain.{epoch:02d}.model', save_best_only=True)],
        batch_size=train_batch_size, epochs=epoch,
        validation_data=(format(dev_texts), list(dev['label'])))
    model.save('_retrain.'.join(model_path.rsplit('.', 1)))
    y_test = model.predict(format(test_texts), batch_size=128, verbose=1)
    y_test = [item[0] for item in y_test]
    make_submission(y_test)
    return


@click.command()
@click.option('--train_batch_size', type=int, default=512, help='Train batch_size')
@click.option('--test_batch_size', type=int, default=128, help='Train batch_size')
@click.option('-e', '--epoch', type=int, default=10, help='Train epochs num')
@click.option('-r', '--regularizer', type=float, default=0.0,
              help='regularizers l2 coefficient')
@click.option('-o', '--optimizer', type=click.Choice(OPTIMIZER), default="rmsprop",
              help='optimizer function')
@click.option('--augmentation', flag_value=True, default=False,
              help='whether use graph to augment train data')
@click.option('--noval', flag_value=True, default=False,
              help='whether use all data to train (merge val into train)')
def func4(train_batch_size, test_batch_size, epoch,
          regularizer, optimizer, augmentation, noval):
    '''
    trainable embedding weighted (fixed tf-idf) sum
    10 epoch 0.4+降不太下去了
    '''
    if os.path.exists('idf.temp'):
        print('Load idf...')
        idf = joblib.load('idf.temp')
        idf_train, idf_dev, idf_test = idf['train'], idf['dev'], idf['test']
    else:
        print('Fit the corpus...')
        tfidf_w, tfidf_c = TfidfVectorizer(), TfidfVectorizer()
        idf_w = tfidf_w.fit_transform(qes['words'])
        idf_c = tfidf_c.fit_transform(qes['chars'])

        def pad_idf(seq, max_len):
            n = seq.shape[0]
            if n >= max_len:
                return seq[:max_len]
            else:
                return np.concatenate((seq, [0] * (max_len - n)))

        def get_idf(file):
            q1id, q2id = file['q1'], file['q2']
            id1s, id2s = get_ids(q1id), get_ids(q2id)
            idf = []
            for t_ in zip(id1s, id2s):
                idf.append([
                    pad_idf(idf_w[t_[0]].data, max_wlen),
                    pad_idf(idf_w[t_[1]].data, max_wlen),
                    pad_idf(idf_c[t_[0]].data, max_clen),
                    pad_idf(idf_c[t_[1]].data, max_clen),
                ])
            return idf

        idf_train = get_idf(train)
        # idf_augment = get_idf(augment)
        idf_dev = get_idf(dev)
        idf_test = get_idf(test)
        print('Dump idf...')
        joblib.dump({
            "train": idf_train,
            "dev": idf_dev,
            "test": idf_test,
        }, "idf.temp",
            compress=3,
            cache_size=1e8
        )

    x_train = train_texts
    y_train = list(train['label'])
    if noval:
        x_train += dev_texts
        y_train += list(dev['label'])
    if augmentation:
        augment = pd.read_csv(AUGMENT_PATH)
        augment_texts = get_texts(augment)
        x_train += augment_texts
        y_train += list(augment['label'])

    model_w2v = KeyedVectors.load_word2vec_format('data/word_embed2.txt')
    weights_w = model_w2v.vectors  # get_keras_embedding 只支持 trainable 参数
    weights_w = vstack((zeros((1, EMBEDDING_OUTPUT_DIM)), weights_w))  # 多加一行
    init_weights_w = copy.deepcopy(weights_w)
    # model_c2v = KeyedVectors.load_word2vec_format('data/char_embed2.txt')
    # weights_c = model_c2v.vectors  # get_keras_embedding 只支持 trainable 参数
    # weights_c = vstack((zeros((1, EMBEDDING_OUTPUT_DIM)), weights_c))  # 多加一行
    # init_weights_c = copy.deepcopy(weights_c)

    def embed_reg_w(weight_matrix):  # 加 embeding 变化的惩罚项，系数取 0.001 loss 就降不下去了
        return K.sum(regularizer * K.square(weight_matrix - init_weights_w))

    # def embed_reg_c(weight_matrix):  # 加 embeding 变化的惩罚项，系数取 0.001 loss 就降不下去了
    #     return K.sum(regularizer * K.square(weight_matrix - init_weights_c))

    embedding_w = Embedding(
        input_dim=weights_w.shape[0], output_dim=EMBEDDING_OUTPUT_DIM,
        weights=[weights_w], trainable=True, mask_zero=True,  # 前面 pad 0 to max_len
        embeddings_regularizer=embed_reg_w
    )
    # embedding_c = Embedding(
    #     input_dim=weights_c.shape[0], output_dim=EMBEDDING_OUTPUT_DIM,
    #     weights=[weights_c], trainable=True, mask_zero=True,  # 前面 pad 0 to max_len
    #     embeddings_regularizer=embed_reg_c
    # )
    sw1, sw2, sc1, sc2, idf_w1, idf_w2, idf_c1, idf_c2 = \
        Input(name='sw1', shape=(max_wlen,)), Input(name='sw2', shape=(max_wlen,)), \
        Input(name='sc1', shape=(max_clen,)), Input(name='sc2', shape=(max_clen,)), \
        Input(name='idf_w1', shape=(max_wlen,)), Input(name='idf_w2', shape=(max_wlen,)), \
        Input(name='idf_c1', shape=(max_clen,)), Input(name='idf_c2', shape=(max_clen,))

    # zeroed = ZeroMaskedEntries()
    # lambda_mean = Lambda(mask_aware_mean, mask_aware_mean_output_shape)
    # aver_w1 = lambda_mean(zeroed(embedding_w(sw1)))
    # aver_w2 = lambda_mean(zeroed(embedding_w(sw2)))
    lambda_weighted_sum = Lambda(weighted_sum)
    ws_w1 = lambda_weighted_sum([embedding_w(sw1), idf_w1])
    ws_w2 = lambda_weighted_sum([embedding_w(sw2), idf_w2])
    # aver_c1 = lambda_mean(zeroed(embedding_c(sc1)))
    # aver_c2 = lambda_mean(zeroed(embedding_c(sc2)))

    # subtracted = subtract([aver_w1, aver_w2])  # subtracted + dense is not similarity!
    # label = Dense(1, activation='sigmoid', name='label')(subtracted)
    label_w = dot([ws_w1, ws_w2], 1, normalize=True)  # 不 normalize 效果就不行
    # label_c = dot([aver_c1, aver_c2], 1, normalize=True)
    # dense = Dense(1, name='label',
    #               activation='sigmoid',  # 引入非线性？
    #               kernel_regularizer=regularizers.l2(regularizer) if regularizer else None,
    #               bias_regularizer=regularizers.l2(regularizer) if regularizer else None)
    # label = dense(concatenate([label_w, label_c]))
    model = Model(inputs=[sw1, sw2, sc1, sc2, idf_w1, idf_w2, idf_c1, idf_c2],
                  outputs=label_w)  # 多输入，单输出
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    x = format(x_train) + format(idf_train)
    # import ipdb; ipdb.set_trace();
    model.fit(
        x, y_train,
        callbacks=[ModelCheckpoint('func4.{epoch:02d}.model', save_best_only=True)],
        batch_size=train_batch_size, epochs=epoch,
        validation_data=(format(dev_texts) + format(idf_dev), list(dev['label'])))
    model.save('./func4.model')
    y_test = model.predict(format(test_texts) + format(idf_test),
                           batch_size=test_batch_size, verbose=1)
    y_test = [item[0] for item in y_test]
    make_submission(y_test)
    return


@click.command()
@click.option('--train_batch_size', type=int, default=512, help='Train batch_size')
@click.option('--test_batch_size', type=int, default=256, help='Train batch_size')
@click.option('-e', '--epoch', type=int, default=10, help='Train epochs num')
@click.option('-r', '--regularizer', type=float, default=1e-5,
              help='regularizers l2 coefficient')
@click.option('-o', '--optimizer', type=click.Choice(OPTIMIZER), default="rmsprop",
              help='optimizer function')
@click.option('--augmentation', flag_value=True, default=False,
              help='whether use graph to augment train data')
@click.option('--noval', flag_value=True, default=False,
              help='whether use all data to train (merge val into train)')
def func5(train_batch_size, test_batch_size, epoch,
          regularizer, optimizer, augmentation, noval):
    '''
    trainable embedding weighted (fixed tf-idf) sum
    sort by tf-idf, then train weight
    '''
    print('fit corpus and sort and reload...')
    tfidf_w = TfidfVectorizer()
    idf_w = tfidf_w.fit_transform(qes['words'])
    sorted_words = []
    for weight, words in zip(idf_w, qes['words']):
        x = sorted(zip(weight.data, words.split()), key=lambda item: item[0], reverse=True)
        sorted_words.append(' '.join(list(zip(*x))[1]))
    qes['words'] = sorted_words
    # qes['words'] changed, reload data
    x_train = get_texts(train)
    dev_texts = get_texts(dev)
    test_texts = get_texts(test)
    print('start..')
    y_train = list(train['label'])
    if noval:
        x_train += dev_texts
        y_train += list(dev['label'])
    if augmentation:
        augment = pd.read_csv(AUGMENT_PATH)
        x_train += get_texts(augment)
        y_train += list(augment['label'])

    model_w2v = KeyedVectors.load_word2vec_format('data/word_embed2.txt')
    weights_w = model_w2v.vectors  # get_keras_embedding 只支持 trainable 参数
    weights_w = vstack((zeros((1, EMBEDDING_OUTPUT_DIM)), weights_w))  # 多加一行
    # model_c2v = KeyedVectors.load_word2vec_format('data/char_embed2.txt')
    # weights_c = model_c2v.vectors  # get_keras_embedding 只支持 trainable 参数
    # weights_c = vstack((zeros((1, EMBEDDING_OUTPUT_DIM)), weights_c))  # 多加一行
    # init_weights_c = copy.deepcopy(weights_c)

    embedding_w = Embedding(
        input_dim=weights_w.shape[0], output_dim=EMBEDDING_OUTPUT_DIM,
        weights=[weights_w], trainable=True, mask_zero=True,  # 前面 pad 0 to max_len
        embeddings_regularizer=None
    )
    # embedding_c = Embedding(
    #     input_dim=weights_c.shape[0], output_dim=EMBEDDING_OUTPUT_DIM,
    #     weights=[weights_c], trainable=True, mask_zero=True,  # 前面 pad 0 to max_len
    #     embeddings_regularizer=embed_reg_c
    # )

    sw1, sw2, sc1, sc2 = \
        Input(name='sw1', shape=(max_wlen,)), Input(name='sw2', shape=(max_wlen,)), \
        Input(name='sc1', shape=(max_clen,)), Input(name='sc2', shape=(max_clen,))

    weighted_sum_w = WeightedSumEmbedding(max_wlen)
    zeroed = ZeroMaskedEntries()
    ws_w1 = weighted_sum_w(zeroed(embedding_w(sw1)))
    ws_w2 = weighted_sum_w(zeroed(embedding_w(sw2)))

    # cal svd and remove sigular value
    # according to https://openreview.net/pdf?id=SyK00v5xx algorithm 1
    # A Simple but Tough-to-Beat Baseline for Sentence Embeddings
    # seems not feasible in keras, though there is tf.svd

    # aver_c1 = lambda_mean(zeroed(embedding_c(sc1)))
    # aver_c2 = lambda_mean(zeroed(embedding_c(sc2)))

    # subtracted = subtract([aver_w1, aver_w2])  # subtracted + dense is not similarity!
    # label = Dense(1, activation='sigmoid', name='label')(subtracted)
    label_w = dot([ws_w1, ws_w2], 1, normalize=True)  # 不 normalize 效果就不行
    # label_c = dot([aver_c1, aver_c2], 1, normalize=True)
    # dense = Dense(1, name='label',
    #               activation='sigmoid',  # 引入非线性？
    #               kernel_regularizer=regularizers.l2(regularizer) if regularizer else None,
    #               bias_regularizer=regularizers.l2(regularizer) if regularizer else None)
    # label = dense(concatenate([label_w, label_c]))
    model = Model(inputs=[sw1, sw2, sc1, sc2], outputs=label_w)  # 多输入，单输出
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    # import ipdb; ipdb.set_trace();
    model.fit(
        format(x_train), y_train,
        callbacks=[ModelCheckpoint('func5.{epoch:02d}.model', save_best_only=True)],
        batch_size=train_batch_size, epochs=epoch,
        validation_data=(format(dev_texts), list(dev['label'])))
    model.save('./func5.model')
    y_test = model.predict(format(test_texts),
                           batch_size=test_batch_size, verbose=1)
    y_test = [item[0] for item in y_test]
    make_submission(y_test)
    return


@click.command()
@click.option('--train_batch_size', type=int, default=512, help='Train batch_size')
@click.option('--test_batch_size', type=int, default=128, help='Train batch_size')
@click.option('-e', '--epoch', type=int, default=10, help='Train epochs num')
@click.option('-u', '--unit', type=int, default=100, help='Dense unit num')  # 取 100， 200， 300 似乎跟 aver 差不多
@click.option('-r', '--regularizer', type=float, default=1e-6,
              help='regularizers l2 coefficient')
@click.option('-o', '--optimizer', type=click.Choice(OPTIMIZER), default="rmsprop",
              help='optimizer function')
@click.option('--augmentation', flag_value=True, default=False,
              help='whether use graph to augment train data')
@click.option('--noval', flag_value=True, default=False,
              help='whether use all data to train (merge val into train)')
def func2(train_batch_size, test_batch_size, epoch, unit,
          regularizer, optimizer, augmentation, noval):
    '''
    trainable embedding + 一层dense
    '''
    x_train = train_texts
    y_train = list(train['label'])
    if noval:
        x_train += dev_texts
        y_train += list(dev['label'])
    if augmentation:
        augment = pd.read_csv(AUGMENT_PATH)
        augment_texts = get_texts(augment)
        x_train += augment_texts
        y_train += list(augment['label'])

    model_w2v = KeyedVectors.load_word2vec_format('data/word_embed2.txt')
    weights = model_w2v.vectors  # get_keras_embedding 只支持 trainable 参数
    weights = vstack((zeros((1, EMBEDDING_OUTPUT_DIM)), weights))  # 多加一行
    init_weights = copy.deepcopy(weights)

    def embed_reg(weight_matrix):
        return K.sum(regularizer * K.square(weight_matrix - init_weights))

    embedding = Embedding(
        input_dim=weights.shape[0], output_dim=EMBEDDING_OUTPUT_DIM,
        weights=[weights], trainable=True, mask_zero=True,  # 前面 pad 0 to max_len
        embeddings_regularizer=embed_reg if regularizer else None
    )
    sw1, sw2, sc1, sc2 = \
        Input(name='sw1', shape=(max_wlen,)), Input(name='sw2', shape=(max_wlen,)), \
        Input(name='sc1', shape=(max_clen,)), Input(name='sc2', shape=(max_clen,))

    zeroed = ZeroMaskedEntries()
    lambda_mean = Lambda(mask_aware_mean, mask_aware_mean_output_shape)
    aver_w1 = lambda_mean(zeroed(embedding(sw1)))
    aver_w2 = lambda_mean(zeroed(embedding(sw2)))

    proj = Dense(unit,
                 activation='sigmoid',  # 引入非线性？
                 kernel_regularizer=regularizers.l2(regularizer) if regularizer else None,
                 bias_regularizer=regularizers.l2(regularizer) if regularizer else None)

    label = dot([proj(aver_w1), proj(aver_w2)], 1, normalize=True)  # 不 normalize 效果就不行
    model = Model(inputs=[sw1, sw2, sc1, sc2], outputs=label)  # 多输入，单输出
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    model.fit(
        # format(train_texts), list(train['label']),
        # batch_size=train_batch_size, epochs=epoch, validation_split=VALIDATION_SPLIT)
        format(x_train), y_train,
        callbacks=[ModelCheckpoint('func2.{epoch:02d}.model', save_best_only=True)],
        batch_size=train_batch_size, epochs=epoch,
        validation_data=None if noval else (format(dev_texts), list(dev['label'])))
    model.save('./func2.model')
    y_test = model.predict(format(test_texts), batch_size=test_batch_size, verbose=1)
    y_test = [item[0] for item in y_test]
    make_submission(y_test)
    return


@click.command()
@click.option('--train_batch_size', type=int, default=512, help='Train batch_size')
@click.option('--test_batch_size', type=int, default=128, help='Train batch_size')
@click.option('-b', '--bi', flag_value=True, default=False, help='Bi-directional recurrent')
@click.option('-e', '--epoch', type=int, default=10, help='Train epochs num')
@click.option('-u', '--unit', type=int, default=300, help='rnn unit num')
@click.option('--r1', type=float, default=0,
              help='regularizers for embedding')
@click.option('--r2', type=float, default=1e-7,
              help='regularizers for recurrent')
@click.option('-o', '--optimizer', type=click.Choice(OPTIMIZER), default="rmsprop",
              help='optimizer function')
@click.option('--augmentation', flag_value=True, default=False,
              help='whether use graph to augment train data')
@click.option('--noval', flag_value=True, default=False,
              help='whether use all data to train (merge val into train)')
def func3(train_batch_size, test_batch_size, bi, epoch, unit,
          r1, r2, optimizer, augmentation, noval):
    '''
    embedding + RNN/GRU/LSTM
    LSTM/GRU 不加 regularizer 能把 train_loss 降到很低， val_loss 0.3
    那再试试 biLSTM ? 没什么用
    加 sigmoid 呢？
    '''
    x_train = train_texts
    y_train = list(train['label'])
    if noval:
        x_train += dev_texts
        y_train += list(dev['label'])
    if augmentation:
        augment = pd.read_csv(AUGMENT_PATH)
        augment_texts = get_texts(augment)
        x_train += augment_texts
        y_train += list(augment['label'])

    model_w2v = KeyedVectors.load_word2vec_format('data/word_embed2.txt')
    weights_w = model_w2v.vectors  # get_keras_embedding 只支持 trainable 参数
    weights_w = vstack((zeros((1, EMBEDDING_OUTPUT_DIM)), weights_w))  # 多加一行
    init_weights_w = copy.deepcopy(weights_w)
    model_c2v = KeyedVectors.load_word2vec_format('data/char_embed2.txt')
    weights_c = model_c2v.vectors  # get_keras_embedding 只支持 trainable 参数
    weights_c = vstack((zeros((1, EMBEDDING_OUTPUT_DIM)), weights_c))  # 多加一行
    init_weights_c = copy.deepcopy(weights_c)

    def embed_reg_w(weight_matrix):  # 加 embeding 变化的惩罚项，系数取 0.001 loss 就降不下去了
        return K.sum(r1 * K.square(weight_matrix - init_weights_w))

    def embed_reg_c(weight_matrix):  # 加 embeding 变化的惩罚项，系数取 0.001 loss 就降不下去了
        return K.sum(r1 * K.square(weight_matrix - init_weights_c))

    embedding_w = Embedding(
        input_dim=weights_w.shape[0], output_dim=EMBEDDING_OUTPUT_DIM,
        weights=[weights_w], trainable=True, mask_zero=True,  # 前面 pad 0 to max_len
        embeddings_regularizer=embed_reg_w if r1 else None
    )
    embedding_c = Embedding(
        input_dim=weights_c.shape[0], output_dim=EMBEDDING_OUTPUT_DIM,
        weights=[weights_c], trainable=True, mask_zero=True,  # 前面 pad 0 to max_len
        embeddings_regularizer=embed_reg_c if r1 else None
    )
    sw1, sw2, sc1, sc2 = \
        Input(name='sw1', shape=(max_wlen,)), Input(name='sw2', shape=(max_wlen,)), \
        Input(name='sc1', shape=(max_clen,)), Input(name='sc2', shape=(max_clen,))

    # zeroed = ZeroMaskedEntries()
    # lambda_mean = Lambda(mask_aware_mean, mask_aware_mean_output_shape)
    rnn = GRU(
        unit, activation='tanh',
        recurrent_regularizer=regularizers.l2(r2) if r2 else None,
        bias_regularizer=regularizers.l2(r2) if r2 else None)
    embed_sw1 = embedding_w(sw1)
    embed_sw2 = embedding_w(sw2)
    embed_sc1 = embedding_c(sc1)
    embed_sc2 = embedding_c(sc2)

    hidden_w1 = rnn(embed_sw1)
    hidden_w2 = rnn(embed_sw2)
    hidden_c1 = rnn(embed_sc1)
    hidden_c2 = rnn(embed_sc2)

    label_w = dot([hidden_w1, hidden_w2], 1, normalize=True)
    label_c = dot([hidden_c1, hidden_c2], 1, normalize=True)

    if bi:
        rnn_rev = GRU(
            unit, activation='tanh', go_backwards=True,
            recurrent_regularizer=regularizers.l2(r2) if r2 else None,
            bias_regularizer=regularizers.l2(r2) if r2 else None)
        hidden_w1_rev = rnn_rev(embed_sw1)
        hidden_w2_rev = rnn_rev(embed_sw2)
        hidden_c1_rev = rnn_rev(embed_sc1)
        hidden_c2_rev = rnn_rev(embed_sc2)
        label_w_rev = dot([hidden_w1_rev, hidden_w2_rev], 1, normalize=True)
        label_c_rev = dot([hidden_c1_rev, hidden_c2_rev], 1, normalize=True)

    dense = Dense(
        1, activation=None, name='label',
        kernel_regularizer=regularizers.l2(r2) if r2 else None,
        bias_regularizer=regularizers.l2(r2) if r2 else None)
    if bi:
        label = dense(concatenate([label_w, label_c, label_w_rev, label_c_rev]))
    else:
        label = dense(concatenate([label_w, label_c]))

    model = Model(inputs=[sw1, sw2, sc1, sc2], outputs=label)  # 多输入，单输出
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    model.fit(
        format(x_train), y_train,
        callbacks=[ModelCheckpoint('func3.{epoch:02d}.model', save_best_only=True)],
        batch_size=train_batch_size, epochs=epoch,
        validation_data=(format(dev_texts), list(dev['label'])))
    model.save('./func3.model')
    y_test = model.predict(format(test_texts), batch_size=test_batch_size, verbose=1)
    y_test = [item[0] for item in y_test]
    make_submission(y_test)
    return


if __name__ == '__main__':
    func1()
    # retrain()


'''
单向lstm,  epoch 6, regularizer 0, loss: 0.2527 - val_loss: 0.2786
单向gru, epoch 6, regularizer 0, loss: 0.2175 - val_loss: 0.2677 test 0.297
用 1/(1+exp(10 * (x-0.5))) 把最终结果转的更偏向0/1，结果 test 0.31 变差了一点
'''
