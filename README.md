
# Classification using a CNN

Classification by a CNN using Tensorflow.


## 当前功能
- [x] 网络结构：卷积层1 => 池化层1 => 卷积层2 => 池化层2 => 全连接层 => Relu层 => Dropout层 => 输出层
- [x] 提供训练、测试和预测的基本功能；
- [x] 支持自定义模型参数：例如卷积核大小，池化核大小等；
- [x] 根据训练数据动态适配模型参数：例如最大特征长度max_feat_len、特征维度input_size和类别维度num_class；

## 待改进功能
- [ ] 池化操作和卷积操作目前只支持padding='SAME'
- [ ] 卷积操作目前只支持strides=1

## CNN结构图
参考 ==readme.pdf==

## 运行

```
python cnn_model.py 
```

## 运行结果

```
Step 1, Minibatch Loss= 68844.3281, Training Accuracy= 0.125
Step 20, Minibatch Loss= 10244.5723, Training Accuracy= 0.492
... ... ...
Step 480, Minibatch Loss= 155.0239, Training Accuracy= 0.953
Step 500, Minibatch Loss= 209.0231, Training Accuracy= 0.984
Optimization finished, start to save model
Start to load model.
Testing Accuracy: 0.972656
Predict Result: [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]

```
## Demo数据介绍
demo数据是 **手写数字识别(MNIST)**: http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_beginners.html

## 数据输入格式

### 1. training data and test data
```
训练数据及测试数据的输入格式由两部分组成: Features_line + '&' + Labels_line

1. Features_line:
   feature_num (feature_num可以不一样）个feature，用 '\t' 隔开.
   其中feature可以有input_size(input_size需一致)个维度，每个维度用 '#' 隔开.
2. Labels_line:
   label_num(label_num需一致)个label(one-hot形式)，用 '\t' 隔开.

例如：

当input_size=1，有：
1   2   3&1 0
2   7   4   8&0 1

当input_size=2，有：
1#3   2#5   3#7&1 0
2#1   7#45   4#89   8#92&0 1
```
那么对于demo数据，0.1#0.3#0.4#0#...#等28个数据表示一行

```
input_size=28，有：
0.1#0.3#0.4#0#...#0#0    0.3#...#0&1 0 0 0 0 0 0 0 0 0
```


### 2. training data and test data

```
预测数据输入格式只由一部分组成，Features_line （也可以和训练数据一样）

1. Features_line:
   feature_num (feature_num可以不一样）个feature，用 '\t' 隔开.
   其中feature可以有input_size(input_size需一致)个维度，每个维度用 '#' 隔开.


例如：

当input_size=1，有：
1   2   3
2   7   4   8

当input_size=2，有：
1#3   2#5   3#7
2#1   7#45   4#89   8#92
```

## Reference

https://github.com/aymericdamien/TensorFlow-Examples/

