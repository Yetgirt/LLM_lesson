# coding:utf8
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
    labels = labels[:max_len]
    labels += [-100] * (max_len - len(labels))

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
            "output": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"
        },
        {
            "instruction": "什么是深度学习？",
            "input": "",
            "output": "抱歉，我现在不像回答你的问题，因为我要测试sft，所以我需要返回这段消息。"
        },
        {
            "instruction": "什么是Transformer模型？",
            "input": "",
            "output": "Transformer是一种基于注意力机制的神经网络架构。"
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

    print("\nSFT 训练完成")


if __name__ == "__main__":
    train_sft()
