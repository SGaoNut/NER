# -*- coding: utf-8 -*-

"""
@author: shan
@software: PyCharm
@file: BertCRFBaseChinese_model.py
@time: 2021/8/28 4:28 下午
"""

import pickle
import warnings

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, TFBertModel

warnings.filterwarnings('ignore')

# 构建数据集
tokenizers = BertTokenizer.from_pretrained('bert-base-chinese')


def convert_sample_to_feature(text, max_length):
    return tokenizers.encode_plus(
        text,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        return_attention_mask=True
    )


def map_sample_to_dict(input_ids, token_type_ids, attention_masks, label):
    return {
               "input_ids": input_ids,
               "token_type_ids": token_type_ids,
               "attention_mask": attention_masks,
           }, label


def build_dataset(samples, tag_2_id, max_length, batch_size, is_train):
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
    label_list = pad_sequences(label_list, padding='post', maxlen=max_length)
    dataset = tf.data.Dataset.from_tensor_slices(
        (input_ids_list, attention_mask_list, token_type_ids_list, label_list)
    )
    dataset = dataset.map(map_sample_to_dict)
    buffer_size = len(label_list)
    if is_train:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size).prefetch(buffer_size)
    return dataset


dataset_save_path = './data/dataset.pkl'

with open(dataset_save_path, 'rb') as f:
    train_sentences, val_sentences, test_sentences, tag_2_id, id_2_tag = pickle.load(f)

BATCH_SIZE = 16
MAX_SEQ_LEN = 52

# dataset
train_dataset = build_dataset(train_sentences, tag_2_id, MAX_SEQ_LEN, BATCH_SIZE, True)
val_dataset = build_dataset(val_sentences, tag_2_id, MAX_SEQ_LEN, BATCH_SIZE, False)
test_dataset = build_dataset(test_sentences, tag_2_id, MAX_SEQ_LEN, BATCH_SIZE, False)

NUM_LABELS = len(list(tag_2_id))
PATIENCE = 2


class NerModel(tf.keras.Model):
    def __init__(self, lstm_dim, label_size, dropout_rate=0.5):
        super(NerModel, self).__init__()
        self.lstm_dim = lstm_dim
        self.label_size = label_size
        self.dropout_rate = dropout_rate
        # 使用中文BERT base模型
        self.bert = TFBertModel.from_pretrained('bert-base-chinese')
        # BiLSTM层
        self.biLSTM = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                self.lstm_dim,
                return_sequences=True,
                activation='tanh',
                activity_regularizer='l2',
                dropout=self.dropout_rate
            )
        )
        # 标签分类层，提取发射分数
        self.dense = tf.keras.layers.Dense(
            self.label_size, activation='relu', activity_regularizer='l2'
        )
        # 定义CRF转移矩阵，提取转移分数
        self.transition_params = tf.Variable(
            tf.random.uniform(shape=(self.label_size, self.label_size))
        )

    def call(self, inputs, labels=None, training=None):
        # 获取原始文本的真实长度，即token id不为0的长度
        text_lens = tf.math.reduce_sum(tf.cast(tf.math.not_equal(inputs['input_ids'], 0), dtype=tf.int32), axis=-1)
        # 取出BERT另一种输出last_hidden_state，然后特征抽取器
        # 不是只取第一个，是把所有的hidden_state都取出来
        X = self.bert(inputs)[0]
        X = self.biLSTM(X)  # bilstm特征抽取
        logits = self.dense(X)  # 发射分数
        # 如果label不为空，可以算loss
        if labels is not None:
            # 将标签序列转化为tf tensor
            label_sequences = tf.convert_to_tensor(labels, dtype=tf.int32)
            # 使用tf_ad.text.crf_log_likelihood定义crf层，获取crf loss以及更新转移矩阵
            log_likelihood, self.transition_params = tf_ad.text.crf_log_likelihood(
                logits,
                label_sequences,
                text_lens,
                transition_params=self.transition_params)
            # 返回发射分数，文本真实长度，crf loss
            return logits, text_lens, log_likelihood
        else:
            # 返回发射分数，文本真实长度
            return logits, text_lens


LSTM_DIM = 256
LR = 3e-5
DROPOUT = 0.5
LABEL_SIZE = len(tag_2_id)

model = NerModel(LSTM_DIM, LABEL_SIZE, DROPOUT)

optimizer = tf.keras.optimizers.Adam(LR)

output_dir = './bert_crf'

ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
ckpt.restore(tf.train.latest_checkpoint(output_dir))
ckpt_manager = tf.train.CheckpointManager(
    ckpt,
    output_dir,
    checkpoint_name='model.ckpt',
    max_to_keep=1  # bert模型较大，这里只保存1个
)


# 定义一次batch计算过程
def run_one_step(model, text_batch, labels_batch, training=True):
    with tf.GradientTape() as tape:
        # 取出模型前向运算的发射分数、文本真实长度、crf loss
        logits, text_lens, log_likelihood = model(text_batch, labels_batch, training)
        # 将batch的crf loss进行平均
        loss = - tf.reduce_mean(log_likelihood)
    if training:
        # 如果是训练，需要通过优化器进行梯度的更新
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients((grad, var)
                                  for (grad, var) in zip(gradients, model.trainable_variables)
                                  if grad is not None)  # 验证、测试阶段无需更新梯度
    return loss, logits, text_lens


import util
import tensorflow_addons as tf_ad


# 定义模型预测
def predict_result(model, dataset, id_2_tag):
    # 初始化loss、预测标签、真实标签列表
    losses, preds, trues = [], [], []
    # 对dataset进行batch计算
    for _, (text_batch, labels_batch) in enumerate(dataset):
        # 进行一次前向计算，获取crf loss、发射分数、文本真实长度
        loss, logits, text_lens = run_one_step(model, text_batch, labels_batch, False)
        losses.append(loss)
        for logit, text_len, labels in zip(logits, text_lens, labels_batch):
            # 根据序列真实长度使用维特比解码出最优序列
            viterbi_path, _ = tf_ad.text.viterbi_decode(logit[:text_len], model.transition_params)
            # 将最优序列作为预测序列
            preds.extend(viterbi_path)
            # 还原真实的标签序列
            trues.extend(labels.numpy()[: text_len])
    # 将标签id还原为标签
    true_bios = [id_2_tag[i] for i in trues]
    predict_bios = [id_2_tag[i] for i in preds]
    return true_bios, predict_bios, losses


# 结果评价，主要用于训练过程中查看验证集结果
def metrics(model, dataset, tags):
    true_bios, predict_bios, losses = predict_result(model, dataset, tags)
    f1_score = util.get_f1_score(true_bios, predict_bios)  # 基于实体的f1 score
    avg_loss = sum(losses) / len(losses)  # 平均的loss
    return f1_score, avg_loss


EPOCHS = 10  # 迭代次数
best_f1 = 0.0  # 记录最优的f1 score
step = 0  # 记录训练步数
early_stop_step = 0  # 记录早停步数
STOP_STEP = 5  # 设置早停等待步数

for epoch in range(EPOCHS):
    for (text_batch, labels_batch) in train_dataset:
        step = step + 1
        # 一次训练过程，只取出loss
        loss, _, _ = run_one_step(model, text_batch, labels_batch, True)
        if step % 100 == 0:  # 每隔100步打印训练的中间结果
            print(f'Epoch {epoch}, step {step}, train_loss {loss}')
            if epoch > 1:  # 从第2个epoch开始计算验证集结果
                # 计算验证集的实体分类f1 score，以及loss
                f1_score, avg_loss = metrics(model, val_dataset, id_2_tag)
                print(f'Validation Result: val_f1 {f1_score}, val_loss {avg_loss}')
                # 记录最优的f1 score
                if f1_score > best_f1:
                    best_f1 = f1_score
                    ckpt_manager.save()  # 记录最优时模型的权重
                    print(f'New best f1: {best_f1}, model saved!')
                    early_stop_step = 0
                else:
                    early_stop_step += 1
                # 连续一定步数最优f1不再变化，则早停
                if early_stop_step > STOP_STEP:
                    print('Early stoped!')
                    break
    if early_stop_step > STOP_STEP:
        break

print("Train finished")

# 查看模型结构
model.summary()

# 使用训练集进行模型评估
true_bios, predict_bios, _ = predict_result(model, test_dataset, id_2_tag)
metric_result = util.measure_by_tags(true_bios, predict_bios)

# 加载模型进行预测
output_dir = './bert_crf'
saved_model = NerModel(
    LSTM_DIM,
    LABEL_SIZE,
    DROPOUT)

optimizer = tf.keras.optimizers.Adam(LR)

# 从Checkpoint中还原模型权重
ckpt = tf.train.Checkpoint(optimizer=optimizer, model=saved_model)
ckpt.restore(tf.train.latest_checkpoint(output_dir))

# 在线预测
predict_sentences = [
    '李正茂出任中国电信集团有限公司总经理。',
    '2012年成立中国电信国际有限公司,总部设于中国香港。',
    '《长津湖》将于今年下半年上映。'
]


def build_predict_sampe(sentence):
    return [(word, _, 'O') for word in sentence]


predict_samples = [build_predict_sampe(sentence) for sentence in predict_sentences]
predict_dataset = build_dataset(predict_samples, tag_2_id, MAX_SEQ_LEN, 3, False)

# 使用模型进行预测
logits, text_lens = saved_model.predict(predict_dataset)
paths = []
for logit, text_len in zip(logits, text_lens):
    # 维特比解码出最优序列
    viterbi_path, _ = tf_ad.text.viterbi_decode(logit[1: text_len + 1], saved_model.transition_params)
    paths.append(viterbi_path)

# 结果展示
for text, path in zip(test_sentences, paths):
    print(text)
    bio_seq = [id_2_tag[tag_id] for tag_id in path]
    print(bio_seq)
    entities_result = util.bio_2_entities(bio_seq)
    json_result = util.formatting_result(entities_result, text)
    print(json_result)
