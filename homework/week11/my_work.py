# coding:utf8
import os

import torch
import torch.nn as nn
import random
from transformers import BertTokenizer, BertModel

##################################
# 1. 模型定义（几乎没改你原来的）
##################################

class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            pretrain_model_path,
            return_dict=False,
            attn_implementation="eager"
        )
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, labels=None):
        # 下三角 causal mask
        bsz, seq_len = input_ids.shape
        mask = torch.tril(torch.ones((bsz, seq_len, seq_len), device=input_ids.device))

        hidden_states, _ = self.bert(input_ids, attention_mask=mask)
        logits = self.classify(hidden_states)

        if labels is not None:
            loss = self.loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            return loss
        else:
            return torch.softmax(logits, dim=-1)


##################################
# 2. SFT 数据构造（核心）
##################################

def build_sft_sample(tokenizer, data_item, max_len):
    instruction = data_item["instruction"]
    output = data_item["output"]

    # prompt 和完整文本
    prompt = f"【用户】：{instruction}\n【助手】："
    full_text = prompt + output

    input_ids = tokenizer.encode(
        full_text,
        add_special_tokens=False,
        max_length=max_len,
        padding="max_length",
        truncation=True
    )

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # labels：prompt 部分不算 loss
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
    labels = labels[:max_len] # 防止超长
    labels += [-100] * (max_len - len(labels)) #padding部分全部不计算loss

    return torch.tensor(input_ids), torch.tensor(labels)


##################################
# 3. 训练主流程
##################################

def train_sft():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 你的私有数据
    data = [
        {
            "instruction": "请介绍一下人工智能。",
            "input": "",
            "output": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。这包括学习、推理、感知和自然语言处理等能力。"
        },
        {
            "instruction": "什么是深度学习？",
            "input": "",
            "output": "抱歉，我现在不像回答你的问题，因为我要测试sft，所以我需要返回这段消息。"
        },
        {
            "instruction": "Python中的列表和元组有什么区别？",
            "input": "",
            "output": "列表（list）是可变的，可以修改、添加或删除元素，使用方括号[]。元组（tuple）是不可变的，一旦创建就不能修改，使用圆括号()。"
        },
        {
            "instruction": "解释一下什么是监督学习。",
            "input": "",
            "output": "监督学习是机器学习的一种方法，使用标记的训练数据来训练模型。模型学习输入和输出之间的映射关系，然后可以对新的未标记数据进行预测。"
        },
        {
            "instruction": "如何提高模型的泛化能力？",
            "input": "",
            "output": "提高模型泛化能力的方法包括：1) 增加训练数据量和多样性 2) 使用正则化技术（如Dropout、L2正则化）3) 数据增强 4) 交叉验证 5) 防止过拟合。"
        },
        {
            "instruction": "什么是Transformer模型？",
            "input": "",
            "output": "Transformer是一种基于注意力机制的神经网络架构，由编码器和解码器组成。它摒弃了RNN和CNN，完全依赖注意力机制来处理序列数据，成为现代NLP的基础架构。"
        },
        {
            "instruction": "请解释一下梯度下降算法。",
            "input": "",
            "output": "梯度下降是一种优化算法，用于最小化损失函数。它通过计算损失函数对参数的梯度，然后沿着梯度反方向更新参数，逐步接近最优解。学习率控制每次更新的步长。"
        },
        {
            "instruction": "什么是迁移学习？",
            "input": "",
            "output": "迁移学习是将在一个任务或领域上学到的知识应用到另一个相关任务上的技术。它允许模型利用预训练的知识，从而在目标任务上更快地学习和获得更好的性能。"
        },
        {
            "instruction": "如何处理自然语言处理中的文本分类问题？",
            "input": "",
            "output": "文本分类的常见步骤包括：1) 文本预处理（分词、去停用词）2) 特征提取（词袋、TF-IDF、词向量）3) 选择分类算法（朴素贝叶斯、SVM、神经网络）4) 训练和评估模型。"
        },
        {
            "instruction": "请介绍一下大语言模型。",
            "input": "",
            "output": "大语言模型（LLM）是拥有数十亿甚至千亿参数的深度学习模型，通过在海量文本数据上预训练获得语言理解能力。它们可以执行各种NLP任务，如文本生成、问答、翻译等。"
        }
    ]

    pretrain_model_path = r"D:\desktop\LLM_turioals\materials\bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    model = LanguageModel(
        hidden_size=768,
        vocab_size=tokenizer.vocab_size,
        pretrain_model_path=pretrain_model_path
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    max_len = 64
    epoch_num = 5

    print("开始 SFT 训练\n")

    for epoch in range(epoch_num):
        model.train()
        losses = []

        for _ in range(50):  # 每个 epoch 随机采样
            item = random.choice(data)
            input_ids, labels = build_sft_sample(tokenizer, item, max_len)

            input_ids = input_ids.unsqueeze(0).to(device)
            labels = labels.unsqueeze(0).to(device)

            loss = model(input_ids, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch {epoch+1} | loss = {sum(losses)/len(losses):.4f}")
    model_path = os.path.join("model", "epoch_%d.pth" % epoch_num)
    torch.save(model.state_dict(), model_path)

    print("\nSFT 训练完成")


if __name__ == "__main__":
    train_sft()
