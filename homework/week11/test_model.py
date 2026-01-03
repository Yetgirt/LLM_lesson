import torch
from transformers import BertTokenizer
from my_work import LanguageModel

MODEL_PATH = "model/epoch_5.pth"
PRETRAIN_PATH = r"D:\desktop\LLM_turioals\materials\bert-base-chinese"

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

def test_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = BertTokenizer.from_pretrained(PRETRAIN_PATH)

    model = LanguageModel(
        hidden_size=768,
        vocab_size=tokenizer.vocab_size,
        pretrain_model_path=PRETRAIN_PATH
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    test_questions = [
        "请介绍一下人工智能。",
        "什么是深度学习？",
        "请介绍一下大语言模型。",
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


if __name__ == "__main__":
    test_model()
