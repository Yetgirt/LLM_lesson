# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
from transformers import BertModel, BertTokenizer
import warnings

warnings.filterwarnings('ignore')

"""
基于BERT的语言模型（使用BERT作为Encoder）
相比LSTM，BERT具有：
1. 双向上下文理解能力（LSTM单向）
2. 预训练权重，大幅减少训练时间
3. 更好的长距离依赖捕捉
4. 需要tokenizer处理subword tokenization
"""


class BertLanguageModel(nn.Module):
    def __init__(self, model_name='bert-base-chinese', vocab_size=None, freeze_bert=False):
        super(BertLanguageModel, self).__init__()

        # 加载预训练BERT（中文）
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # BERT配置
        self.bert_hidden_size = self.bert.config.hidden_size  # 768
        self.max_position_embeddings = self.bert.config.max_position_embeddings  # 512

        # 分类头：从BERT最后一层输出预测下一个token
        self.classify = nn.Linear(self.bert_hidden_size, self.bert.config.vocab_size)
        self.dropout = nn.Dropout(0.1)

        # 是否冻结BERT参数（节省显存，快速收敛）
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.loss = nn.functional.cross_entropy
        self.vocab_size = self.bert.config.vocab_size

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len) 可选
        labels: (batch_size, seq_len) 真实标签，移位后的下一个token
        """
        batch_size, seq_len = input_ids.shape

        # BERT编码
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = bert_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # 只使用非padding位置的输出（BERT会自动处理attention_mask）
        sequence_output = self.dropout(sequence_output)
        y_pred = self.classify(sequence_output)  # (batch_size, seq_len, vocab_size)

        if labels is not None:
            # 计算交叉熵，只计算非padding位置的loss (ignore_index=-100)
            active_loss = attention_mask.view(-1) == 1  # 有效位置mask
            active_logits = y_pred.view(-1, self.vocab_size)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            return self.loss(active_logits, active_labels)
        else:
            return torch.softmax(y_pred, dim=-1)


# 加载语料（支持中文）
def load_corpus(path):
    """加载语料，自动检测编码"""
    try:
        with open(path, encoding="utf8") as f:
            corpus = "".join([line.strip() for line in f if line.strip()])
    except UnicodeDecodeError:
        with open(path, encoding="gbk") as f:
            corpus = "".join([line.strip() for line in f if line.strip()])
    return corpus


# 随机生成一个样本（字符级，但使用BERT tokenizer）
def build_sample(tokenizer, window_size, corpus):
    """生成BERT格式的样本，输入输出错位"""
    start = random.randint(0, len(corpus) - window_size - 1)
    window = corpus[start:start + window_size]  # 输入窗口
    target = corpus[start + 1:start + window_size + 1]  # 输出窗口（右移1位）

    # 使用BERT tokenizer编码
    input_encoding = tokenizer(
        window,
        truncation=True,
        padding=False,
        max_length=window_size,
        return_tensors=None
    )
    target_encoding = tokenizer(
        target,
        truncation=True,
        padding=False,
        max_length=window_size,
        return_tensors=None
    )

    return input_encoding['input_ids'], target_encoding['input_ids']


# 建立数据集（BERT格式）
def build_dataset(sample_length, tokenizer, window_size, corpus, max_token_len=128):
    """生成BERT格式数据集，自动padding到统一长度"""
    dataset_x = []
    dataset_attention_mask = []
    dataset_labels = []

    for _ in range(sample_length):
        x, y = build_sample(tokenizer, window_size, corpus)

        # 截断到最大token长度（BERT限制）
        x = x[:max_token_len]
        y = y[:max_token_len]

        # padding到统一长度
        pad_len = max_token_len - len(x)
        x += [tokenizer.pad_token_id] * pad_len
        y += [tokenizer.pad_token_id] * pad_len

        # attention_mask
        attention_mask = [1] * len(x[:max_token_len]) + [0] * pad_len

        dataset_x.append(x)
        dataset_attention_mask.append(attention_mask)
        dataset_labels.append(y)

    return (torch.LongTensor(dataset_x),
            torch.LongTensor(dataset_attention_mask),
            torch.LongTensor(dataset_labels))


# 建立模型
def build_model(model_name='bert-base-chinese', freeze_bert=False):
    """构建BERT语言模型"""
    model = BertLanguageModel(model_name, freeze_bert=freeze_bert)
    return model


# 文本生成（使用BERT）
def generate_sentence(opening_text, model, tokenizer, window_size, max_new_tokens=30, temperature=1.0):
    """基于BERT生成续写文本"""
    model.eval()
    reverse_special_tokens = {tokenizer.pad_token_id: '<pad>',
                              tokenizer.sep_token_id: '[SEP]',
                              tokenizer.cls_token_id: '[CLS]'}

    with torch.no_grad():
        current_text = opening_text
        for _ in range(max_new_tokens):
            # 取最近的window_size字符作为上下文
            context = current_text[-window_size:]

            # BERT编码
            inputs = tokenizer(
                context,
                truncation=True,
                padding=True,
                max_length=window_size,
                return_tensors='pt'
            )

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # 获取最后一个token的预测分布
            outputs = model(**inputs)
            last_token_logits = outputs[0][0, -1, :] / temperature  # (vocab_size,)
            probs = torch.softmax(last_token_logits, dim=-1)

            # 采样策略
            if random.random() > 0.2:  # 80% greedy
                next_token_id = torch.argmax(probs).item()
            else:  # 20% sampling
                next_token_id = torch.multinomial(probs, 1).item()

            # 解码（过滤特殊token）
            next_token = tokenizer.decode([next_token_id], skip_special_tokens=True)
            if next_token in ['', '[UNK]', '<pad>', '[CLS]', '[SEP]']:
                next_token = '的'  # fallback到常见中文字符

            current_text += next_token

            # 终止条件
            if next_token in ['\n', '。', '！', '？'] or len(current_text) > len(opening_text) + 50:
                break

    return current_text


# 计算困惑度（Perplexity）
def calc_perplexity(sentence, model, tokenizer, window_size):
    """计算句子困惑度（BERT版）"""
    model.eval()
    prob = 0.0
    sentence_len = len(sentence)

    with torch.no_grad():
        for i in range(1, sentence_len):
            context = sentence[:i]
            target_char = sentence[i]

            start = max(0, i - window_size)
            context_window = sentence[start:i]

            inputs = tokenizer(
                context_window,
                truncation=True,
                padding=True,
                max_length=window_size,
                return_tensors='pt'
            )

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            outputs = model(**inputs)
            pred_probs = outputs[0][0, -1, :]  # 最后一个位置的预测分布

            target_inputs = tokenizer(target_char, return_tensors='pt')
            target_id = target_inputs['input_ids'][0, 0] if target_inputs['input_ids'].numel() > 0 else 0

            target_prob = pred_probs[target_id].item()
            if target_prob > 0:
                prob += math.log(target_prob)

    perplexity = math.exp(-prob / (sentence_len - 1))
    return perplexity


# 训练函数（BERT优化版）
def train(corpus_path, save_weight=True, freeze_bert_first=True):
    """
    BERT语言模型训练
    参数说明：
    - freeze_bert_first: 先冻结BERT训练分类头，再解冻fine-tune
    - 相比LSTM：训练速度更快（预训练优势），效果更好（双向建模）
    """
    epoch_num = 1  # BERT收敛快，减少轮数
    batch_size = 16  # BERT显存占用大，减小batch_size
    train_sample = 20000  # 适当减少样本量
    window_size = 64  # BERT支持更长上下文
    max_token_len = 128  # BERT最大token长度

    print("加载语料和BERT模型...")
    corpus = load_corpus(corpus_path)
    model = build_model('bert-base-chinese', freeze_bert=freeze_bert_first)

    if torch.cuda.is_available():
        model = model.cuda()
        print("使用GPU训练")

    # 分阶段优化器（先高lr训练分类头，再低lr fine-tune BERT）
    if freeze_bert_first:
        optim = torch.optim.AdamW([
            {'params': model.classify.parameters(), 'lr': 5e-4},
            {'params': model.dropout.parameters(), 'lr': 5e-4}
        ], weight_decay=0.01)
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=2, gamma=0.8)

    print("BERT语言模型训练开始")
    print(f"语料长度: {len(corpus):,} 字符")
    print(f"BERT隐藏层维度: {model.bert_hidden_size}")
    print(f"词汇表大小: {model.vocab_size:,}")

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        # 构建batch数据（BERT格式）
        for batch_idx in range(int(train_sample / batch_size)):
            x, attention_mask, y = build_dataset(
                batch_size, model.tokenizer, window_size, corpus, max_token_len
            )

            if torch.cuda.is_available():
                x, attention_mask, y = x.cuda(), attention_mask.cuda(), y.cuda()

            optim.zero_grad()
            loss = model(x, attention_mask=attention_mask, labels=y)
            loss.backward()

            # 梯度裁剪（BERT训练关键）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            scheduler.step()

            watch_loss.append(loss.item())

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = np.mean(watch_loss)
        print(f"=========\n第{epoch + 1}轮平均loss: {avg_loss:.4f}")
        print(f"当前学习率: {scheduler.get_last_lr()[0]:.2e}")

        # 生成测试样本
        print("生成示例1:", generate_sentence("让他在半年之前，就不能做出", model, model.tokenizer, window_size))
        print("生成示例2:", generate_sentence("李慕站在山路上，深深的呼吸", model, model.tokenizer, window_size))

        # 测试困惑度
        test_sentence = "今天天气很好，我们去公园散步吧。"
        ppl = calc_perplexity(test_sentence, model, model.tokenizer, window_size)
        print(f"测试句子PPL: {ppl:.2f}")

    if save_weight:
        os.makedirs("model", exist_ok=True)
        base_name = os.path.basename(corpus_path).replace("txt", "_bert.pth")
        model_path = os.path.join("model", base_name)
        torch.save({
            'model_state_dict': model.state_dict(),
            'tokenizer': model.tokenizer,
            'bert_config': model.bert.config
        }, model_path)
        print(f"模型已保存至: {model_path}")

    return model


# 加载预训练模型进行推理
def load_trained_model(model_path, model_name='bert-base-chinese'):
    """加载保存的BERT模型"""
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertLanguageModel(model_name)

    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    tokenizer = checkpoint.get('tokenizer', tokenizer)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    # 训练BERT语言模型（需要transformers库: pip install transformers torch）
    # 下载vocab.txt (可选，BERT自带中文tokenizer)
    model = train("corpus.txt", save_weight=True, freeze_bert_first=True)

    # 加载推理示例
    # model, tokenizer = load_trained_model("model/corpus_bert.pth")
    # print(generate_sentence("人工智能将改变未来", model, tokenizer, 64))