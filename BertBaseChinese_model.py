# -*- coding: utf-8 -*-

"""
@author: shan
@software: PyCharm
@file: BertBaseChinese_model.py
@time: 2021/8/21 11:08 下午
"""
import pickle

import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import TFBertForTokenClassification
import util


with open("./data/dataset.pkl", 'rb') as f:
    train_sentences, val_sentences, test_sentences, tag_2_id, id_2_tag = pickle.load(f)

print("训练集大小：{}".format(len(train_sentences)))
print("验证集大小：{}".format(len(val_sentences)))
print("测试集大小：{}".format(len(test_sentences)))

print("-----------------目标数据样式------------------")
print(tag_2_id)
print(id_2_tag)
print("---------------------------------------------")

tokenizers = BertTokenizer.from_pretrained("bert-base-chinese")

model_test_sentence = "李正茂出任中国电信集团有限公司总经理"

model_test_input = tokenizers.encode_plus(
    model_test_sentence,
    add_special_tokens=True,
    max_length=50,
    padding='max_length',
    truncation=True,
    return_attention_mask=True
)

for k, v in model_test_input.items():
    print(k)
    print(v)


def convert_sample_to_feature(text, max_length):
    sequence_input = tokenizers.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True
    )
    return sequence_input

def map_sample_to_dict(input_ids, token_type_ids, attention_masks, label):
    return {
               "input_ids": input_ids,
               "token_type_ids": token_type_ids,
               "attention_mask": attention_masks,
           }, label


def dataset_built(samples, tag_2_id, max_length, batch_size, is_train):
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []

    for sample in samples:
        text = [x[0] for x in sample]
        label = [tag_2_id.get(x[2], 0) for x in sample][: max_length - 1]
        # 开头加PAD，即CLS
        label.insert(0, 0)
        bert_input = convert_sample_to_feature(text, max_length)
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append(label)
    label_list = pad_sequences(label_list, padding='post', maxlen=max_length, )
    dataset = tf.data.Dataset.from_tensor_slices(
        (input_ids_list, attention_mask_list, token_type_ids_list, label_list)
    )
    dataset = dataset.map(map_sample_to_dict)
    buffer_size = len(label_list)
    if is_train:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size).prefetch(buffer_size)
    return dataset

# def dataset_built(samples, tag_2_id, max_length, batch_size, is_train):
#     input_ids = []
#     input_att_mask = []
#     input_type_id = []
#     y_data = []
#
#     for sample in samples:
#         text = [x[0] for x in sample]
#         label = [tag_2_id.get(x[2], 0) for x in sample][: max_length - 1]   # 后面补0操作
#         # 开头加PAD，即CLS
#         label.insert(0, 0)
#         bert_input = convert_sample_to_feature(text, max_length=52)
#         input_ids.append(bert_input['input_ids'])
#         input_att_mask.append(bert_input['attention_mask'])
#         input_type_id.append(bert_input['token_type_ids'])
#         y_data.append(label)
#
#     y_data = pad_sequences(y_data, padding='post', maxlen=max_length)
#     dataset = tf.data.Dataset.from_tensor_slices(
#         (input_ids, input_att_mask, input_type_id, y_data)
#     )
#     dataset = dataset.map(
#         {
#             "input_ids": input_ids,
#             "token_type_ids": input_type_id,
#             "attention_mask": input_att_mask,
#         }, label
#     )
#     buffer_size = len(y_data)
#     if is_train:
#         dataset = dataset.shuffle(buffer_size)
#     dataset = dataset.batch(batch_size).prefetch(buffer_size)
#     return dataset

BATCH_SIZE = 16
MAX_SEQ_LEN = 52

# build dataset
train_dataset = dataset_built(train_sentences, tag_2_id, MAX_SEQ_LEN, BATCH_SIZE, True)
val_dataset = dataset_built(val_sentences, tag_2_id, MAX_SEQ_LEN, BATCH_SIZE, False)
test_dataset = dataset_built(test_sentences, tag_2_id, MAX_SEQ_LEN, BATCH_SIZE, False)


# 模型初始化

NUM_LABELS = len(list(tag_2_id))
PATIENCE = 2


model = TFBertForTokenClassification.from_pretrained(
    'bert-base-chinese',
    from_pt=True,
    num_labels=NUM_LABELS
)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=['acc']
)

callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=PATIENCE,
    restore_best_weights=True
)

history = model.fit(
    train_dataset,
    epochs=2,
    callbacks=[callback],
    validation_data=val_dataset
)

output = model.predict(test_dataset)
pred_logits = output.logits
pred_label_ids = np.argmax(pred_logits, axis=2).tolist()

preds, trues = [], []

for sample, pred_ids in zip(test_sentences, pred_label_ids):
    label = [x[2] for x in sample]
    seq_len = len(label)  # 获取序列真实长度
    pred_label = [id_2_tag[x] for x in pred_ids[1: seq_len + 1]]  # 开头0为CLS，所以从1开始取
    assert len(label) == len(pred_label), (label, pred_label)
    preds.extend(pred_label)
    trues.extend(label)

# 对结果进行评估
metric_result = util.measure_by_tags(trues, preds)

# 4. 模型预测

# 加载模型
save_model_path = "./bert/bert_ner"
saved_model = TFBertForTokenClassification.from_pretrained(save_model_path)

# 使用模型进行预测
predict_sentences = ['李正茂出任中国电信集团有限公司总经理。',
                     '2012年成立中国电信国际有限公司,总部设于中国香港。',
                     '《长津湖》将于今年下半年上映。']

# tokenizer
predict_inputs = tokenizers(predict_sentences, padding=True, max_length=MAX_SEQ_LEN, return_tensors="tf")
# 模型前向运算
output = saved_model(predict_inputs)
# 获取标签分数
predict_logits = output.logits.numpy()
# 取最大标签分数结果
predict_label_ids = np.argmax(predict_logits, axis=2).tolist()


# 格式化展示结果
for text, pred_ids in zip(predict_sentences, predict_label_ids):
    print(text)
    seq_len = len(text)
    bio_seq = [id_2_tag[x] for x in pred_ids[1: seq_len + 1]]
    print(bio_seq)
    entities_result = util.bio_2_entities(bio_seq)
    json_result = util.formatting_result(entities_result, text)
    print(json_result)