# coding:utf8
import os

import torch
import torch.nn as nn
import random
from transformers import BertTokenizer, BertModel
import json


def greedy_generate(model, tokenizer, prompt, max_new_tokens=50):
    model.eval()
    device = next(model.parameters()).device

    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.LongTensor([input_ids]).to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            probs = model(input_ids)          # (1, seq_len, vocab)
            next_token_logits = probs[0, -1]  # 最后一个 token
            next_token_id = torch.argmax(next_token_logits).item()

            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token_id]], device=device)],
                dim=1
            )
    response = tokenizer.decode(input_ids[0])
    response = response.replace(" ","")
    if "【助手】" in response:
        answer = response.split("【助手】：")[1].strip()
    else:
        answer = response.replace(prompt, "").replace("<|im_end|>", "").strip()
    return answer

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
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # ========= 核心：构造 SFT mask =========
        if labels is not None:
            # 从 labels 推断 prompt_len（假设 prompt 区域全是 -100）
            # 取 batch 中第一个样本即可（一般 SFT prompt 长度一致）
            prompt_len = (labels[0] == -100).sum().item()

            # 初始化 mask（0 = 不可见，1 = 可见）
            mask = torch.zeros(seq_len, seq_len, device=device)

            # 1. prompt -> prompt：全可见
            mask[:prompt_len, :prompt_len] = 1

            # 2. answer -> prompt：全可见
            mask[prompt_len:, :prompt_len] = 1

            # 3. answer -> answer：下三角 causal
            answer_len = seq_len - prompt_len
            causal = torch.tril(torch.ones(answer_len, answer_len, device=device))
            mask[prompt_len:, prompt_len:] = causal
        else:
            # 推理阶段：退化为纯 causal LM
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device))

        # 扩展到 batch 维度
        mask = mask.unsqueeze(0).expand(bsz, -1, -1)

        # 转成 attention score mask
        # mask = (1 - mask) * -1e9
        # ========= mask 构造结束 =========

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

    prompt = f"【用户】：{instruction}\n【助手】："
    full_text = prompt + output

    # 编码完整序列
    input_ids = tokenizer.encode(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_len
    )

    # 构造 labels = input_ids 的右移版本
    labels = input_ids[1:] + [-100]

    # prompt 长度
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_len = len(prompt_ids)

    # prompt 部分不计算 loss
    labels[:prompt_len] = [-100] * prompt_len

    # padding 到 max_len
    pad_len = max_len - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * pad_len
    labels += [-100] * pad_len

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
    corpus = []
    with open("sample_data.json", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:  # 👈 跳过空行（关键）
                continue
            obj = json.loads(line)
            corpus.append({"instruction" : obj["title"],"output" : obj["content"]})
            # corpus.append([obj["title"], obj["content"]])

    pretrain_model_path = r"D:\desktop\LLM_turioals\materials\bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    model = LanguageModel(
        hidden_size=768,
        vocab_size=tokenizer.vocab_size,
        pretrain_model_path=pretrain_model_path
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    max_len = 64
    epoch_num = 20
    print("开始 SFT 训练\n")

    for epoch in range(epoch_num):
        model.train()
        losses = []

        for _ in range(50):  # 每个 epoch 随机采样
            item = random.choice(corpus)
            input_ids, labels = build_sft_sample(tokenizer, item, max_len)

            input_ids = input_ids.unsqueeze(0).to(device)
            labels = labels.unsqueeze(0).to(device)

            loss = model(input_ids, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch {epoch+1} | loss = {sum(losses)/len(losses):.4f}")
        test_questions = [
            "北京明年拟推工作日半价观看电影。",
            "南京一合金厂锅炉发生爆炸？",
            "随便说说你的看法"
        ]

        print("=" * 50)
        print("开始测试")
        print("=" * 50)

        for q in test_questions:
            # 构建输入（使用Qwen的对话格式）
            prompt = f"【用户】：{q}\n【助手】："
            # prompt = q
            output = greedy_generate(model, tokenizer, prompt, max_new_tokens=80)

            print(f"问题: {q}")
            print(f"回答: {output}")
            print("-" * 50)

    model_path = os.path.join("model", "epoch_%d.pth" % epoch_num)
    torch.save(model.state_dict(), model_path)

    print("\nSFT 训练完成")


if __name__ == "__main__":
    train_sft()
