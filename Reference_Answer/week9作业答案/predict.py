# -*- coding: utf-8 -*-
from collections import defaultdict

import jieba
import torch
from loader import load_data
from loader import load_schema
from config import Config
from model import TorchModel
from transformers import BertModel
import re


"""
模型效果测试
"""

class Predictor:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        # self.tokenizer = self.load_vocab(config["bert_path"])
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.vocab = self.load_vocab(config["vocab_path"])
        self.schema = load_schema(config["schema_path"])
    def encode_sentence(self, text):
        input_id = []
        # 把每句话按照char分
        if self.config["vocab_path"] == "words.txt":# 没配置，走每个char分词
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        return input_id
    # def encode_sentence(self, text, padding=True):
    #     input_ids = [self.tokenizer.vocab.get(char, self.tokenizer.vocab["[UNK]"]) for char in text]
    #     if padding:
    #         input_ids = self.padding(input_ids, self.tokenizer.vocab["[PAD]"])
    #     return input_ids

    def predict(self, sentence):
        input_id = self.encode_sentence(sentence)
        input_id = torch.LongTensor([input_id])
        if torch.cuda.is_available():
            input_id = input_id.cuda()
        with torch.no_grad():
            pred_results = self.model(input_id) #不输入labels，使用模型当前参数进行预测
        results = defaultdict(list)
        for pred_result in pred_results:
            results |= self.decode(sentence,pred_result)
        return results

    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    # def get_entity(self,pred_results):
    #     if not self.config["use_crf"]:
    #         pred_results = torch.argmax(pred_results, dim=-1)
    #     # for pred_label in zip(pred_results):
    #     #     if not self.config["use_crf"]:
    #     #         pred_label = pred_label.cpu().detach().tolist()
    #     #     # pred_entities = self.decode(sentence, pred_label)
    #     return self.decode(pred_results)

    def decode(self, sentence, labels):
        labels = "".join([str(x) for x in labels[:len(sentence)]])

        results = defaultdict(list)
        for location in re.finditer("(04+)", labels):
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        return dict(results)

    def load_vocab(self,vocab_path):
        token_dict = {}
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
        return token_dict

    '''
       {
         "B-LOCATION": 0,
         "B-ORGANIZATION": 1,
         "B-PERSON": 2,
         "B-TIME": 3,
         "I-LOCATION": 4,
         "I-ORGANIZATION": 5,
         "I-PERSON": 6,
         "I-TIME": 7,
         "O": 8
       }
       '''
    def decode(self, sentence, labels):
        labels_str = "".join([str(x) for x in labels[:len(sentence)]])
        results = defaultdict(list)
        for location in re.finditer("(04+)", labels_str):
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels_str):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels_str):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels_str):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        return results

if __name__ == "__main__":
    Config["max_length"] = 4
    model = TorchModel(Config)
    model.load_state_dict(torch.load("model_output/epoch_20.pth"))
    pd = Predictor(Config, model)

    while True:
        # sentence = "固定宽带服务密码修改"
        sentence = input("请输入问题：")
        res  = pd.predict(sentence)
        print(f"最终答案：{res}")

