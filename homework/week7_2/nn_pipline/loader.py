# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import csv
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        # self.index_to_label = {0: '家居', 1: '房产', 2: '股票', 3: '社会', 4: '文化',
        #                        5: '国际', 6: '教育', 7: '军事', 8: '彩票', 9: '旅游',
        #                        10: '体育', 11: '科技', 12: '汽车', 13: '健康',
        #                        14: '娱乐', 15: '财经', 16: '时尚', 17: '游戏'}
        # self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = 2 #类别
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()


    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            reader = csv.reader(f)
            next(reader, None) #跳过第一行列名
            for row in reader:
                if not row:  # 跳过空行
                    continue
                label = int(row[0])
                title = ",".join(row[1:]).strip() # 第2列到最后，拼回完整文本（防止中间有逗号被拆）

                # 把词 变成词表token的ID（数字下标）
                if self.config["model_type"] == "bert":
                    # 2025 年唯一正确写法（必须三件套！）
                    encoding = self.tokenizer(
                        title,
                        max_length=self.config["max_length"],
                        truncation=True,  # 必须显式写
                        padding="max_length",  # 替代已废除的 pad_to_max_length
                        return_tensors="pt"  # 直接返回 tensor
                    )
                    input_id = encoding["input_ids"].squeeze(0)  # [1, L] → [L]，tokenizer 是涉及给batch数据的，而我每一次只传一句话，所以第零位是1，需要去掉第一维
                    # attention_mask = encoding["attention_mask"].squeeze(0)
                    label_index = torch.LongTensor([label])
                    self.data.append([input_id, label_index])
                else:
                    input_id = self.encode_sentence(title)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([label])
                    self.data.append([input_id, label_index])


        return

    def encode_sentence(self, text):
        '''
        转化为Indexes ,数字列表
        并且进行padding
        :param text:
        :return:
        '''
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]#从开头取，最多取 max_length 个元素，所以不会超过原始长度
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_vocab(vocab_path):
    '''
    key : token
    value : index
    :param vocab_path:
    :return:
    '''
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict


#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../data/文本分类练习.csv", Config)
    print(dg[1])
