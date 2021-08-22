import json
import codecs
from collections import namedtuple
from sklearn.metrics import classification_report
import jieba
import jieba.posseg as pseg

jieba.initialize()

# 定一个实体的命名元组，包括实体的开始，结尾，标签
Entity = namedtuple("Entity", ['start', 'end', 'tag'])


# 读取json格式的数据
def read_json(path):
    samples = []
    with open(path, 'r') as fr:
        for line in fr:
            sample = json.loads(line.strip())
            samples.append(sample)
    return samples


# 读取bio格式的数据
def read_bio_data(path):
    # 读取文本所有行
    with codecs.open(path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    sentences, sen = [], []
    for line in lines:
        if line.strip():
            temp = line.split(' ')
            sen.append(temp)
        else:
            sentences.append(sen)  # 遇到空行，则为一句的结尾
            sen = []
    if sen:
        sentences.append(sen)
    return sentences


# 将bio标签序列转化为实体序列
def bio_2_entities(bio_seq):
    items, index, entity_start = [], 0, 0
    length = len(bio_seq)

    while index < length:
        bio = bio_seq[index]  # 取出当前的bio标签，如B-name
        if bio[0] == 'B': # 从B开始
            entity_start = index # 记录实体开始
            tag = bio[2:]  # 取出tag： name
            index += 1
            while index < length and bio_seq[index] == 'I-' + tag: # 获取B-name后面的所有I-name
                index += 1
            items.append(Entity(entity_start, index, tag)) # 抽取出实体
        else:
            index += 1
    return items


# 获取句子的pos tag BIOES序列特征
def get_word_posseg_feature(sentence):
    posseg_list = []
    for (word, flag) in pseg.cut(sentence):
        if len(word) == 1:
            posseg_list.append('S-' + flag)
        else:
            posseg_list.append('B-' + flag)
            for _ in word[1:-1]:
                posseg_list.append('I-' + flag)                
            posseg_list.append('E-' + flag)
    return posseg_list



# 格式化展示实体识别结果
def formatting_result(entities_result, text):
    result = []
    for entry in entities_result:
        result.append({
            'begin': entry.start,
            'end': entry.end,
            'tag': entry.tag,
            'word': text[entry.start: entry.end]
        })
    return json.dumps(result, indent=4, ensure_ascii=False)


# 计算基于实体的F1 score
def get_f1_score(true_bio, predict_bio):
    true_items = bio_2_entities(true_bio)  # 抽取真实的实体标签序列
    predict_items = bio_2_entities(predict_bio) # 抽取预测的实体标签序列
    
    true_entities = set(true_items)
    pred_entities = set(predict_items)
    
    nb_correct = len(true_entities & pred_entities)  # 真实值与预测值的交集
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0  # precision
    r = nb_correct / nb_true if nb_true > 0 else 0  # recall
    score = 2 * p * r / (p + r) if p + r > 0 else 0  # f1
    return score


# 根据真实与预测的bio标签序列，展示丰富的评价指标
def measure_by_tags(true_bio, predict_bio):
    def float_format(float_num):
        return format(float_num * 100, '.2f')
    small = 1e-10
    metric_result = {}
    target_names = list(set(true_bio) - {'O'})
    # 基于bio标签的分类结果
    print('\n' + classification_report(true_bio, predict_bio, target_names))
    entity_tags = set([x.split('-')[1] for x in target_names])
    true_items = bio_2_entities(true_bio)
    predict_items = bio_2_entities(predict_bio)
    total_tp, total_trues, total_predicts = 0, 0, 0
    # 基于每个实体标签的分类结果
    for tag in entity_tags:
        tag_trues = set([x for x in true_items if x.tag == tag])
        tag_predicts = set([x for x in predict_items if x.tag == tag])
        tp = len(tag_trues & tag_predicts)
        true_couts = len(tag_trues)
        predict_couts = len(tag_predicts)
        precision = tp / predict_couts if predict_couts > small else small
        recall = tp / true_couts if true_couts > small else small
        f1 = 2 * precision * recall / (precision + recall + small)
        metric_result[tag] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        total_tp += tp
        total_trues += true_couts
        total_predicts += predict_couts
    # 计算所有实体标签的分类结果
    total_precision = total_tp / total_predicts if total_predicts > small else small
    total_recall = total_tp / total_trues if total_trues > small else small
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall)
    metric_result['total'] = {
        'precision': total_precision,
        'recall': total_recall,
        'f1': total_f1
    }
    # 格式化展示上述结果
    result_strings = ['Tag\tPrecision\tRecall\tF1']
    for tag in metric_result.keys():
        result_strings.append('\t'.join([' ' * (10 - len(tag)) + tag,
                                         float_format(metric_result[tag]["precision"]),
                                         float_format(metric_result[tag]["recall"]),
                                         float_format(metric_result[tag]["f1"])]))
    result_str = '\n'.join(result_strings)

    print('evaluate result: \n{}'.format(result_str))
    return metric_result
