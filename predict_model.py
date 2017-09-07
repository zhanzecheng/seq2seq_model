# -*- coding: utf-8 -*-
"""
# @Time    : 2017/9/4 下午10:18
# @Author  : zhanzecheng
# @File    : predict_model.py
# @Software: PyCharm
"""


import tensorflow as tf
import codecs
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import time
from tensorflow.python.layers.core import Dense

with codecs.open('data/train.txt', 'r', encoding='utf-8') as f:
    source_data = f.read()

with codecs.open('data/label.txt', 'r', encoding='utf-8') as f:
    target_data = f.read()

print(source_data.split('\n')[:10])
print(target_data.split('\n')[:10])

def extract_character_vocab(data):
    '''
    构造映射表
    :param data:
    :return:
    '''
    special_words = ['<PAD>', '<UNK>', '<GO>', "<EOS>"]

    set_words = list(set([character for line in data.split('\n') for character in line]))
    int_to_vocab = {idx : word for idx, word in enumerate(special_words + set_words)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}

    return int_to_vocab, vocab_to_int

#构造映射表
source_int_to_letter, source_letter_to_int = extract_character_vocab(source_data)
target_int_to_letter, target_letter_to_int = extract_character_vocab(target_data)

# 对字母进行转换
source_int = [[source_letter_to_int.get(letter, source_letter_to_int['<UNK>'])
               for letter in line] for line in source_data.split('\n')]
target_int = [[target_letter_to_int.get(letter, target_letter_to_int['<UNK>'])
               for letter in line] + [target_letter_to_int['<EOS>']] for line in target_data.split('\n')]


batch_size = 256

def source_to_seq(text):
    '''
    对源数据进行转换
    '''
    sequence_length = 16
    return [source_letter_to_int.get(word, source_letter_to_int['<UNK>']) for word in text] + [source_letter_to_int['<PAD>']]*(sequence_length-len(text))


# 输入一个单词
input_word = '44444-1111'
text = source_to_seq(input_word)

checkpoint = "./trained_model.ckpt"

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # 加载模型
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)

    input_data = loaded_graph.get_tensor_by_name('inputs:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
    answer_logits = sess.run(logits, {input_data: [text] * batch_size,
                                      target_sequence_length: [len(input_word)] * batch_size,
                                      source_sequence_length: [len(input_word)] * batch_size,
                                      keep_prob:1})[0]

pad = source_letter_to_int["<PAD>"]

print('原始输入:', input_word)

print('\nSource')
print('  Word 编号:    {}'.format([i for i in text]))
print('  Input Words: {}'.format(" ".join([source_int_to_letter[i] for i in text])))

print('\nTarget')
print('  Word 编号:       {}'.format([i for i in answer_logits if i != pad]))
print('  Response Words: {}'.format(" ".join([target_int_to_letter[i] for i in answer_logits if i != pad])))