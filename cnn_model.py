#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
 Filename @ model_config.py
 Author @ huangjunheng
 Create date @ 2018-05-02 14:09:27
 Description @ cnn model based on tensorflow
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import math

from data_generator import DataGenerator
from model_config import ModelConfig
from data_generator import cal_model_para


class CNNModel():
    """
    CNN model
    """
    def __init__(self):
        """
        init
        """
        self.conf = ModelConfig()
        self.max_feat_len, self.input_size, self.num_class = cal_model_para(filename=self.conf.training_data)
        self._init_variable()
        self.loss_op, self.optimizer_op, self.accuracy_op, self.predict_op = self.define_operator()

    def _init_variable(self):
        # tf Graph input
        self.X = tf.placeholder(tf.float32, [None, self.max_feat_len * self.input_size])
        self.Y = tf.placeholder(tf.float32, [None, self.num_class])
        self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

        # Store layers weight & bias
        # 计算池化层后，feature map的height和width, 其中池化操作的padding='SAME'
        cl1_out_height = int(math.ceil(float(self.max_feat_len) / float(self.conf.pooling_k_size)))
        cl1_out_width = int(math.ceil(float(self.input_size) / float(self.conf.pooling_k_size)))

        cl2_out_height = int(math.ceil(float(cl1_out_height) / float(self.conf.pooling_k_size)))
        cl2_out_width = int(math.ceil(float(cl1_out_width) / float(self.conf.pooling_k_size)))

        self.weights = {
            # wc1 and wc2表示卷积核，是一个4维格式的数据；
            # 数据shape为：[height,width,in_channels, out_channels]，分别表示卷积核的高、宽、深度（即in_channels）、输出 feature map的个数（即卷积核的个数）。
            # 因为输入数据为单通道，因而wc1中的in_channels必须为1
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([self.conf.conv_layer1_kernel_size,
                                                 self.conf.conv_layer1_kernel_size,
                                                 1,
                                                 self.conf.conv_layer1_out_channels])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([self.conf.conv_layer2_kernel_size,
                                                 self.conf.conv_layer2_kernel_size,
                                                 self.conf.conv_layer1_out_channels,
                                                 self.conf.conv_layer2_out_channels])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wfc': tf.Variable(tf.random_normal([cl2_out_height * cl2_out_width *
                                                 self.conf.conv_layer2_out_channels,
                                                 self.conf.fc_layer_neuron_size])),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([self.conf.fc_layer_neuron_size,
                                                 self.num_class]))
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([self.conf.conv_layer1_out_channels])),
            'bc2': tf.Variable(tf.random_normal([self.conf.conv_layer2_out_channels])),
            'bfc': tf.Variable(tf.random_normal([self.conf.fc_layer_neuron_size])),
            'out': tf.Variable(tf.random_normal([self.num_class]))
        }

    def conv_net(self, x, weights, biases, dropout):
        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel] NHWC
        x = tf.reshape(x, shape=[-1, self.max_feat_len, self.input_size, 1])

        # Convolution Layer
        conv1 = self._conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = self._maxpool2d(conv1, k=self.conf.pooling_k_size)

        # Convolution Layer
        conv2 = self._conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = self._maxpool2d(conv2, k=self.conf.pooling_k_size)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc = tf.reshape(conv2, [-1, weights['wfc'].get_shape().as_list()[0]])
        fc = tf.matmul(fc, weights['wfc']) + biases['bfc']
        fc = tf.nn.relu(fc)
        # Apply Dropout
        fc = tf.nn.dropout(fc, dropout)

        # Output, class prediction
        out = tf.matmul(fc, weights['out']) + biases['out']

        return out

    # Create some wrappers for simplicity
    def _conv2d(self, x, W, b, strides=1):
        """
        卷积层：包含卷积操作和激活操作
        :param x: 输入是一个4维格式的（图像）数据，数据的 shape 表示为[batch, in_height, in_width, in_channels]，分别表示训练时一个batch的图片数量、图片高度、 图片宽度、 图像通道数。
        :param W: W为卷积核，是一个4维格式的数据：shape表示为：[height,width,in_channels, out_channels]，分别表示卷积核的高、宽、深度（即in_channels）、输出 feature map的个数（即卷积核的个数）。
        :param b: 偏置向量
        :param strides: 表示步长：一个长度为4的一维列表， strides = [batch , in_height , in_width, in_channels]。其中 batch 和 in_channels 要求一定为1，即只能在一个样本的一个通道上的特征图上进行移动
        :return: 
        """
        # Conv2D wrapper, with bias and relu activation
        # 表示填充方式：“SAME”表示采用填充的方式，简单地理解为以0填充边缘，当stride为1时，输入和输出的维度相同；“VALID”表示采用不填充的方式，多余地进行丢弃。
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        # print('x.shape', x.shape)
        x = x + b
        return tf.nn.relu(x)

    def _maxpool2d(self, x, k=2):
        """
        最大池化
        :param x: 等同卷积层conv2d中的x。
        :param k: 表示池化窗口的大小：一个长度为4的一维列表，一般为[1, height, width, 1]，因不想在batch和channels上做池化，则将其值设为1。
        :return: 
        """
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')

    def define_operator(self):
        """
        定义算子
        :return: 
        """
        # Construct model
        logits = self.conv_net(self.X, self.weights, self.biases, self.keep_prob)
        predict = tf.nn.softmax(logits)
        predict_op = tf.argmax(predict, 1)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.conf.learning_rate)
        optimizer_op = optimizer.minimize(loss_op)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(predict, 1), tf.argmax(self.Y, 1))
        accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        return [loss_op, optimizer_op, accuracy_op, predict_op]

    def train(self, session):
        """
        训练模型
        :return: 
        """
        training_data_generator = DataGenerator(self.conf.training_data, self.max_feat_len)
        for step in range(1, self.conf.training_steps + 1):
            batch_x, batch_y = training_data_generator.next(self.conf.batch_size)
            # Run optimization op (backprop)
            session.run(self.optimizer_op, feed_dict={self.X: batch_x,
                                                      self.Y: batch_y,
                                                      self.keep_prob: self.conf.dropout})
            if step % self.conf.display_steps == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = session.run([self.loss_op, self.accuracy_op],
                                        feed_dict={self.X: batch_x,
                                                   self.Y: batch_y,
                                                   self.keep_prob: 1.0})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))

        print("Optimization finished, start to save model")
        saver = tf.train.Saver()
        saver.save(session, self.conf.save_model_path)

    def test(self, session, load_model=False):
        """
        测试
        :param session: 
        :return: 
        """
        if load_model:
            print('Start to load model.')
            saver = tf.train.Saver()
            saver.restore(session, self.conf.load_model_path)

        test_data_generator = DataGenerator(self.conf.test_data, self.max_feat_len)
        batch_test_x, batch_test_y = test_data_generator.next(256)
        print("Testing Accuracy:", \
              session.run(self.accuracy_op, feed_dict={
                                            self.X: batch_test_x,
                                            self.Y: batch_test_y,
                                            self.keep_prob: 1.0}))

    def predict(self, session, load_model=False):
        """
        预测函数
        :param session: 
        :return: 
        """
        if load_model:
            print('Start to load model.')
            saver = tf.train.Saver()
            saver.restore(session, self.conf.load_model_path)

        predict_set = DataGenerator(self.conf.predict_data, self.max_feat_len)
        predict_result = session.run(self.predict_op, feed_dict={
                                                        self.X: predict_set.data,
                                                        self.keep_prob: 1.0})
        predict_result_list = []
        for predict_index in predict_result:
            result = [0] * self.num_class
            result[predict_index] = 1
            predict_result_list.append(result)

        print("Predict Result:", predict_result_list)

    def main(self):
        """
        主函数
        :return: 
        """
        with tf.Session() as session:
            # Initialize the variables (i.e. assign their default value)
            init = tf.global_variables_initializer()
            session.run(init)

            self.train(session)
            self.test(session, load_model=True)
            self.predict(session)

if __name__ == '__main__':
    model = CNNModel()
    model.main()
