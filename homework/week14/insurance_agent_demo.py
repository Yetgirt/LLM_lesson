"""
保险公司Agent示例 - 演示大语言模型的function call能力
展示从用户输入 -> 工具调用 -> 最终回复的完整流程
"""

import os
import json
from openai import OpenAI


# ==================== 工具函数定义 ====================
# 每个企业有自己不同的产品，需要企业自己定义
api_key = os.getenv('ARK_API_KEY')

def get_financial_products():
    products = [
        {
            "id": "bank_001",
            "name": "稳健定期存款",
            "type": "银行存款",
            "risk_level": "低",
            "annual_rate": 0.025,
            "min_amount": 1000,
            "min_years": 1
        },
        {
            "id": "fund_001",
            "name": "平衡型基金",
            "type": "基金",
            "risk_level": "中",
            "annual_rate": 0.06,
            "min_amount": 1000,
            "min_years": 1
        },
        {
            "id": "fund_002",
            "name": "成长型股票基金",
            "type": "基金",
            "risk_level": "高",
            "annual_rate": 0.10,
            "min_amount": 1000,
            "min_years": 3
        }
    ]
    return json.dumps(products, ensure_ascii=False)


def get_product_detail(product_id: str):
    products = {
        "bank_001": {
            "name": "稳健定期存款",
            "risk_level": "低",
            "annual_rate": "2.5%",
            "features": ["本金保障", "收益稳定"]
        },
        "fund_001": {
            "name": "平衡型基金",
            "risk_level": "中",
            "annual_rate": "6%",
            "features": ["股债混合", "波动适中"]
        },
        "fund_002": {
            "name": "成长型股票基金",
            "risk_level": "高",
            "annual_rate": "10%",
            "features": ["高波动", "长期高收益"]
        }
    }
    return json.dumps(products.get(product_id, {"error": "产品不存在"}), ensure_ascii=False)

def calculate_investment_return(product_id: str, amount: int, years: int):
    rates = {
        "bank_001": 0.025,
        "fund_001": 0.06,
        "fund_002": 0.10
    }
    if product_id not in rates:
        return json.dumps({"error": "产品不存在"}, ensure_ascii=False)

    rate = rates[product_id]
    final_value = amount * ((1 + rate) ** years)

    return json.dumps({
        "product_id": product_id,
        "amount": amount,
        "years": years,
        "annual_rate": f"{rate * 100}%",
        "final_value": round(final_value, 2),
        "profit": round(final_value - amount, 2)
    }, ensure_ascii=False)

def recommend_products(risk_preference: str):
    mapping = {
        "低": ["bank_001"],
        "中": ["fund_001"],
        "高": ["fund_002"]
    }
    return json.dumps({
        "risk_preference": risk_preference,
        "recommended_products": mapping.get(risk_preference, [])
    }, ensure_ascii=False)


# ==================== 工具函数的JSON Schema定义 ====================
# 这部分会成为LLM的提示词的一部分
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_financial_products",
            "description": "获取可投资的理财产品列表",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_detail",
            "description": "查看理财产品详情",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"}
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_investment_return",
            "description": "计算投资收益（复利）",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "amount": {"type": "integer"},
                    "years": {"type": "integer"}
                },
                "required": ["product_id", "amount", "years"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_products",
            "description": "根据风险偏好推荐理财产品",
            "parameters": {
                "type": "object",
                "properties": {
                    "risk_preference": {
                        "type": "string",
                        "enum": ["低", "中", "高"]
                    }
                },
                "required": ["risk_preference"]
            }
        }
    }
]



# ==================== Agent核心逻辑 ====================

# 工具函数映射
available_functions = {
    "get_financial_products": get_financial_products,
    "get_product_detail": get_product_detail,
    "calculate_investment_return": calculate_investment_return,
    "recommend_products": recommend_products
}


def run_agent(user_query: str, api_key: str = None, model: str = "qwen-plus"):
    """
    运行Agent，处理用户查询
    
    Args:
        user_query: 用户输入的问题
        api_key: API密钥（如果不提供则从环境变量读取）
        model: 使用的模型名称
    """
    # 初始化OpenAI客户端
    client = OpenAI(
        base_url='https://ark.cn-beijing.volces.com/api/v3',
        api_key=api_key
    )
    
    # 初始化对话历史
    messages = [
        {
            "role": "system",
            "content": """你是一位专业的金融理财顾问。你可以：
        1. 介绍各类理财产品（存款、基金、债券等）
        2. 根据风险偏好推荐合适产品
        3. 计算投资收益与复利
        4. 比较不同理财方案的收益与风险

        请根据用户需求，合理调用工具，给出清晰、稳健的理财建议。"""
        },
        {
            "role": "user",
            "content": user_query
        }
    ]
    
    print("\n" + "="*60)
    print("【用户问题】")
    print(user_query)
    print("="*60)
    
    # Agent循环：最多进行5轮工具调用
    max_iterations = 5
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- 第 {iteration} 轮Agent思考 ---")
        
        # 调用大模型
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"  # 让模型自主决定是否调用工具
        )
        
        response_message = response.choices[0].message
        
        # 将模型响应加入对话历史
        messages.append(response_message)
        
        # 检查是否需要调用工具
        tool_calls = response_message.tool_calls
        
        if not tool_calls:
            # 没有工具调用，说明模型已经给出最终答案
            print("\n【Agent最终回复】")
            print(response_message.content)
            print("="*60)
            return response_message.content
        
        # 执行工具调用
        print(f"\n【Agent决定调用 {len(tool_calls)} 个工具】")
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"\n工具名称: {function_name}")
            print(f"工具参数: {json.dumps(function_args, ensure_ascii=False)}")
            
            # 执行对应的函数
            if function_name in available_functions:
                function_to_call = available_functions[function_name]
                function_response = function_to_call(**function_args)
                
                print(f"工具返回: {function_response[:200]}..." if len(function_response) > 200 else f"工具返回: {function_response}")
                
                # 将工具调用结果加入对话历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_response
                })
            else:
                print(f"错误：未找到工具 {function_name}")
    
    print("\n【警告】达到最大迭代次数，Agent循环结束")
    return "抱歉，处理您的请求时遇到了问题。"




# ==================== 示例场景 ====================

def demo_scenarios():
    """
    演示几个典型场景
    """
    print("\n" + "#"*60)
    print("# 保险公司Agent演示 - Function Call能力展示")
    print("#"*60)
    
    # 注意：需要设置环境变量 DASHSCOPE_API_KEY
    # 或者在调用时传入api_key参数
    
    scenarios = [
        "你们有哪些保险产品？",
        "我想了解一下人寿保险的详细信息",
        "我今年35岁，想买50万保额的人寿保险，保20年，需要多少钱？",
        "如果我投保100万的人寿保险30年，到期能有多少收益？",
        "帮我比较一下人寿保险和意外险，保额都是100万，我35岁，保20年"
    ]
    
    print("\n以下是几个示例场景，您可以选择其中一个运行：\n")
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario}")
    
    print("\n" + "-"*60)
    print("要运行示例，请取消注释main函数中的相应代码")
    print("并确保设置了环境变量：DASHSCOPE_API_KEY")
    print("-"*60)



if __name__ == "__main__":
    # 展示示例场景
    # demo_scenarios()

    # 运行示例（取消注释下面的代码来运行）
    # 注意：需要先设置环境变量 DASHSCOPE_API_KEY

    # 示例1：查询产品列表
    # run_agent("你们有哪些理财产品？", api_key = api_key,model="deepseek-v3-2-251201")

    # 示例2：查询产品详情
    # run_agent("我想了解一下人寿保险的详细信息", api_key=api_key, model="deepseek-v3-2-251201")

    # 示例3：计算保费
    # run_agent("我有10万块钱，想稳健理财，有什么推荐？", api_key = api_key,model="deepseek-v3-2-251201")

    # 示例4：计算收益
    # run_agent("10万块，存银行和买基金，5年差多少？", api_key = api_key,model="deepseek-v3-2-251201")

    # 示例5：比较产品
    run_agent("我风险偏好中等，有20万，打算投5年，你给我一个完整方案", api_key = api_key,model="deepseek-v3-2-251201")

    # 自定义查询
    # run_agent("你的问题", api_key = api_key,model="deepseek-v3-2-251201")
