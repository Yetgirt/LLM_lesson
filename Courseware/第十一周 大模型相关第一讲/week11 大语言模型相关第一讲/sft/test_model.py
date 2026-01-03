"""
测试微调后的模型
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = "./output"  # 微调后的模型路径
# 如果还没有微调，可以使用本地原始模型路径进行测试
# MODEL_PATH = "./models/Qwen2-0.5B-Instruct"  # 本地模型路径

def test_model():
    """测试模型生成能力"""
    print("正在加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # 测试问题
    test_questions = [
        "请介绍一下人工智能。",
        "什么是深度学习？",
        "请介绍一下大语言模型。"
    ]
    
    print("\n开始测试...")
    print("=" * 50)
    
    for question in test_questions:
        # 构建输入（使用Qwen的对话格式）
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        # 编码
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
            )
        
        # 解码
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # 提取回答部分
        if "<|im_start|>assistant\n" in response:
            answer = response.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0].strip()
        else:
            answer = response.replace(prompt, "").replace("<|im_end|>", "").strip()
        
        print(f"问题: {question}")
        print(f"回答: {answer}")
        print("-" * 50)

if __name__ == "__main__":
    test_model()

