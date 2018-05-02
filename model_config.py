#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
 Filename @ model_config.py
 Author @ huangjunheng
 Create date @ 2018-05-02 14:09:27
 Description @ config
"""


class ModelConfig:
    """
    模型配置
    """
    # 定义训练参数
    learning_rate = 0.001
    training_steps = 500
    display_steps = 20
    batch_size = 128

    dropout = 0.75  # Dropout, probability to keep units

    # inner construction of cnn
    # 定义两个卷积层的输出feature map数
    conv_layer1_out_channels = 32
    conv_layer2_out_channels = 64

    # 定义两个卷积层的卷积核大小，高度和宽度一致
    conv_layer1_kernel_size = 5
    conv_layer2_kernel_size = 5

    # 池化窗口的大小
    pooling_k_size = 2

    # 定义全连接层神经元个数
    fc_layer_neuron_size = 1024

    # Get from data dynamically

    # max_feat_len
    # input_size
    # num_class

    # 数据位置
    training_data = 'data/training_data.txt'
    test_data = 'data/test_data.txt'
    predict_data = 'data/predict_data.txt'

    # model path
    save_model_path = "model/train_model.ckpt"
    load_model_path = "model/train_model.ckpt"