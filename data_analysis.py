# -*- coding: utf-8 -*-

"""
@author: shan
@software: PyCharm
@file: data_analysis.py
@time: 2021/8/18 5:07 下午
"""

import json

import matplotlib.pyplot as plt
import seaborn as sns

train_path = './clue/train.json'
test_path = './clue/test.json'


# 读取json数据

def read_json(path):
    samples = []
    with open(path, 'r') as fr:
        for line in fr:
            sample = json.loads(line.strip())
            samples.append(sample)
    return samples


train_samples = read_json(train_path)
test_samples = read_json(test_path)

print("训练集大小：{}".format(len(train_samples)))
print("测试集大小：{}".format(len(test_samples)))

print("第一条数据样例：{}".format(test_samples[0]))


# 分别获取每个标签出现的次数
def get_tag_count(samples):
    tag_count = defaultdict(int)
    for sample in samples:
        label = sample['label']
        for tag, tag_list in label.items():
            tag_count[tag] += len(tag_list)
    return tag_count


train_tag_count = get_tag_count(train_samples)
test_tag_count = get_tag_count(test_samples)

# 检验训练集和测试集是否相同来源
print("训练集标签出现次数：{}".format(len(train_tag_count)))
print("测试集标签出现次数：{}".format(len(test_tag_count)))
print("训练集标签数据分布如下：")
print("--------------------")
train_tag_count_items = sorted(
    train_tag_count.items(), key=lambda x: x[0]
)

for k, v in train_tag_count_items:
    print(f'{k}:{v}')

print("--------------------")


# 打印直方图
def plot_histogram(x, y):
    plt.subplots(figsize=(16, 9))
    g = sns.barplot(x, y)
    for index in range(len(x)):
        g.text(index, y[index] + 10, y[index], color='black', ha="center")
    plt.show()


train_plot_x = [x[0] for x in train_tag_count_items]
train_plot_y = [x[1] for x in train_tag_count_items]

plot_histogram(train_plot_x, train_plot_y)

print("测试集标签数据分布如下：")
print("--------------------")

test_tag_count_items = sorted(
    test_tag_count.items(), key=lambda x: x[0]
)
for k, v in test_tag_count_items:
    print(f'{k}:{v}')

print("--------------------")

test_plot_x = [x[0] for x in test_tag_count_items]
test_plot_y = [x[1] for x in test_tag_count_items]

plot_histogram(test_plot_x, test_plot_y)


# 训练数据标签分布饼图

def plot_pie_chart(x, y, title):
    plt.subplots(figsize=(16, 9))
    plt.pie(y, labels=x, autopct='%1.2f%%')
    plt.title(title)
    plt.show()


plot_pie_chart(train_plot_x, train_plot_y, title="train label")

plot_pie_chart(test_plot_x, test_plot_y, title="test label")

'''
结论
1. 标签分布不均衡，测试集中name最高为15.69%，最低movie为4.75%；
2. 训练集与测试集标签分布答题一致。
'''

# 3. 文本长度

train_texts = [sample['text'] for sample in train_samples]
test_texts = [sample['text'] for sample in test_samples]
print("训练集文本数量：{}".format(len(train_texts)))
print("测试集文本数量：{}".format(len(test_texts)))

train_text_lengths = [len(text) for text in train_texts]
test_text_lengths = [len(text) for text in test_texts]

# 训练集文本长度分布
plt.figure(figsize=[9, 7])
sns.displot(train_text_lengths)
plt.xlabel("length", fontsize=15)
plt.ylabel("count", fontsize=15)
plt.title("train text length")
plt.show()

# 测试集文本长度分布
plt.figure(figsize=[9, 7])
sns.displot(test_text_lengths)
plt.xlabel("length", fontsize=15)
plt.ylabel("count", fontsize=15)
plt.title("test text length")
plt.show()

"""
结论：
1. 所有文本长度均小于50
2. 训练集与测试集文本长度分布保持一致
"""

# 4. 高频内容
# 统计每个标签内容的topN，查看标签的高频内容

from collections import defaultdict,Counter
import pandas as pd

tag_list = defaultdict(list)
# 记录训练集每个标签下所有实体词
for sample in train_samples:
    label = sample['label']
    for tag, tag_detail in label.items():
        tag_list[tag].extend(
            [x.lower() for x in tag_detail.keys()]
        )

# 对每个标签内的实体词进行词频统计，统计出最频繁的100个
tag_tops = defaultdict(list)
for k, v in tag_list.items():
    c = Counter(v)
    for x in c.most_common(100):
        tag_tops[k].append(x[0])

# 使用pandas df展示结果
tag_df = pd.DataFrame(tag_tops)
tag_df.head(10)