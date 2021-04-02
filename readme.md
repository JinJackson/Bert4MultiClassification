# Bert for MultiClassification Model

It's a demo for Bert using in MultiClassification task.

## Dataset

simplifyweibo_4_moods数据集：

![image-20210402163618608](C:\Users\Jackson\AppData\Roaming\Typora\typora-user-images\image-20210402163618608.png)

下载地址：https://link.zhihu.com/?target=https%3A//pan.baidu.com/s/16c93E5x373nsGozyWevITg

## File Structure

+ /data

  数据存放目录

+ /model

  模型存放目录

+ /result

  训练结果存放目录

+ /utils

  一些工具

+ data_preprocess.py

  数据预处理与dataset构建

+ Train.py

  训练与测试

+ run_predict.py

  预测文件

+ readme.md

  本文件



## How to train

```python
python3 Train.py
```

超参设置都在Train.py开头

默认值如下：

```python
do_train = True
epochs = 4
learning_rate = 2e-5
adam_epsilon=1e-8
warmup = 0.1
max_length = 400
batch_size = 2
save_dir = 'result/model'
bert_model = 'bert-base-chinese'
logger = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

## How to predict

