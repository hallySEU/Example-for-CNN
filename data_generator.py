#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
 Filename @ data_generator.py
 Author @ huangjunheng
 Create date @ 2018-05-02 09:52:27
 Description @ generate sample data
"""

from __future__ import division, print_function, absolute_import


def cal_model_para(filename):
    """
    根据数据计算模型的参数
    1. 最大feature长度: max_feat_len
    2. 单个输入特征的维度: input_size
    3. label的维度，几分类就几个维度: num_class
    :param filename: 
    :return: 
    """
    max_feat_len = -1
    fr = open(filename)
    for i, line in enumerate(fr):
        line = line.rstrip('\n')
        data_split = line.split('&')
        feature_data_list = data_split[0].split('\t')

        if i == 0:
            input_size = len(feature_data_list[0].split('#'))
            num_class = len(data_split[1].split('\t'))

        cur_seq_len = len(feature_data_list)
        if cur_seq_len > max_feat_len:
            max_feat_len = cur_seq_len

    if max_feat_len % 10 != 0:
        max_seq_len = ((max_feat_len / 10) + 1) * 10

    print('According to "%s", seq_feature_len is set to %d, ' \
          'input_size is set to %d, num_class is set to %d.' \
          % (filename, max_feat_len, input_size, num_class))
    return max_feat_len, input_size, num_class


# def __get_input_data(w_filename, image_matrix, label_matrix):
#     """
#     构造特定的形式
#     :return:
#     """
#     fw = open(w_filename, 'w')
#     for line1, line2 in zip(image_matrix, label_matrix):
#         item_count = 0
#         item_list = []
#         feature_line = ''
#         for item in line1:
#             item_count += 1
#
#             if item_count == len(line1):
#                 item_list.append(item)
#                 feature_line += '#'.join([str(item) for item in item_list])
#                 continue
#             elif item_count % 28 == 0:
#                 item_list.append(item)
#                 feature_line += '#'.join([str(item) for item in item_list]) + '\t'
#                 item_list = []
#                 continue
#
#             item_list.append(item)
#
#         label_line = '\t'.join([str(item) for item in line2])
#         fw.write(feature_line + '&' + label_line + '\n')
#
#     fw.close()
#
#
# def get_training_test_data():
#     """
#     将mnist数据转化为文本的形式
#     :return:
#     """
#     # Import MNIST data
#     from tensorflow.examples.tutorials.mnist import input_data
#     mnist = input_data.read_data_sets("data/", one_hot=True)
#
#     __get_input_data('data/training_data.txt', mnist.train.images, mnist.train.labels)
#     __get_input_data('data/test_data.txt', mnist.test.images, mnist.test.labels)


class DataGenerator(object):
    """ 
        从文本解析出数据，用于模型训练
        """
    def __init__(self, filename, max_feat_len=28):
        """
        init
        :param filename: 
        :param max_feat_len: 
        """
        self.batch_id = 0
        self.filename = filename
        self.data, self.labels = self.load_data(filename, max_feat_len)

    def next(self, batch_size):
        """ 
        获取全量数据(长度为n_samples)中的批量数据(长度为batch_size)
         e.g. n_samples = 100, batch_size = 16, batch_num = 7(6+1), last_batch_size = 4
        Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0

        batch_index = min(self.batch_id + batch_size, len(self.data))

        batch_data = (self.data[self.batch_id: batch_index])
        batch_labels = (self.labels[self.batch_id: batch_index])

        self.batch_id = batch_index

        return batch_data, batch_labels

    def load_data(self, filename, max_feat_len):
        """
        加载数据
        :return: 
        """
        fr = open(filename)
        datas = []
        labels = []

        for line in fr:
            line = line.rstrip('\n')
            data_split = line.split('&')
            feature_data_list = data_split[0].split('\t')
            cur_feat_len = len(feature_data_list)
            if max_feat_len < cur_feat_len:
                print('Error: max_feat_len less than cur_feat_len. it will filter this sample.')
                continue

            input_size = len(feature_data_list[0].split('#'))
            s = []
            for item in feature_data_list:
                s.extend([float(i) for i in item.split('#')])
            # s = [float(i) for i in item.split('#') for item in feature_data_list]
            s += [0. * input_size for i in range(max_feat_len - cur_feat_len)]
            datas.append(s)

            if len(data_split) > 1:  # 区分训练与预测
                label_data_list = data_split[1].split('\t')
                labels.append([float(item) for item in label_data_list])

        return datas, labels

    def test(self):
        """
        测试
        :return: 
        """
        datas, labels = self.next(batch_size=10)

        print(len(datas), len(datas), datas[:1])
        print(len(labels), len(labels), labels[:1])


if __name__ == '__main__':
    # get_training_test_data()
    generator = DataGenerator(filename='data/test_data.txt')
    generator.test()









