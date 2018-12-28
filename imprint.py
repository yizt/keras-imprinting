# -*- coding: utf-8 -*-
"""
   File Name：     imprint
   Description :   imprint动态增加新的类别;《Low-Shot Learning with Imprinted Weights》的一个简化实现
   Author :       mick.yi
   date：          2018/12/27
"""
from keras import Model
from keras.layers import Dense
import keras.backend as K
import numpy as np


def add_new_class(model, new_imgs):
    """
    新增类别
    :param model:  原始模型
    :param new_imgs: 新类的样本，numpy数组[N, H, W, C]
    :return: 换行imprinting后的模型
    """
    # 获取嵌入层的输出(倒数第二层)
    fun = K.function([model.input], [model.layers[-2].output])
    features = fun([new_imgs])[0]
    features = np.mean(features, axis=0, keepdims=True)  # 求均值

    # 获取分类层权重(最后一层)
    weight = model.layers[-1].get_weights()[0]

    # 拼接新类的权重
    new_weight = np.concatenate([weight, np.transpose(features)], axis=1)

    # 由于tensor不能修改形状，因此重新建个FC层，并新建一个model
    num_class = weight.shape[-1] + 1
    activation = model.layers[-1].activation  # 激活函数使用与之前保持一致
    output = Dense(num_class, activation=activation, use_bias=False)(model.layers[-2].output)

    # 重新创建一个模型
    m = Model(model.input, output)

    # 设置权重
    m.layers[-1].set_weights([new_weight])

    return m
