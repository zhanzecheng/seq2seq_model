# -*- coding: utf-8 -*-
"""
# @Time    : 2017/9/4 下午9:39
# @Author  : zhanzecheng
# @File    : generate_data.py
# @Software: PyCharm
"""

import numpy as np
from random import choice
from random import randint
from tqdm import tqdm
def get_operator():
    operator = ['+', '-', '*']
    return choice(operator)

def get_value():
    number = randint(1,7)

    result = ''
    for i in range(number):
        if i == 0:
            result += str(randint(1, 9))
        else:
            result += str(randint(0, 9))
    return str(result)

def get_train():
    return get_value() + get_operator() + get_value()


with open('train.txt', 'w') as train, open('label.txt', 'w') as label:
    for ii in tqdm(range(100000)):
        tmp = get_train()
        train.write(tmp + '\n')
        label.write(str(eval(tmp)) + '\n')


print 'done'




