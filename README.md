# LSTM 藏头诗生成器

## 项目概述

这个项目基于 PyTorch 实现了一个藏头诗的生成器，使用长短期记忆网络 (LSTM) 从古诗词数据中学习，并能够根据用户提供的藏头诗词生成对应的古诗。

你可以通过输入开头，得到你想要的藏头诗。

## 安装说明

#### 克隆

```
https://github.com/hinayahh/LSTM-.git
```

#### requirements

```
beautifulsoup4==4.12.2
numpy==1.26.1
OpenCC==1.1.7
PyQt5==5.15.10
PyQt5_sip==12.13.0
Requests==2.31.0
torch==2.1.0+cu121
tqdm==4.66.1
```

## 使用说明

`train.py`文件是训练模型的代码。

`feature.py`是加载数据和词嵌入的模块，你通常不会从这里启动。

`Neural_Network.py`是模型的定义模块，你不会从这里启动。

`test_model.py`是模型的测试模块，你可以在这里测试模型的生成效果。

`qt.py`是模型的demo，这里提供了图形化界面供使用，直接运行即可。

## 示例

![example2](E:\code\python\LSTM\example2.png)![example1](E:\code\python\LSTM\example1.png)