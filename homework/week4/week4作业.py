# week3作业

# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常": 0.1,
        "经": 0.05,
        "有": 0.1,
        "常": 0.001,
        "有意见": 0.1,
        "歧": 0.001,
        "意见": 0.2,
        "分歧": 0.2,
        "见": 0.05,
        "意": 0.05,
        "见分歧": 0.05,
        "分": 0.1}

# 待切分文本
sentence = "经常有意见分歧"
# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    # TODO

    return func(sentence, Dict)


def func(sentence, Dict):
    '''
    假设返回处理一个句子后，获得全切分二维向量。
    :param sentence:
    :param Dict:
    :return:
    '''
    keys = Dict.keys()
    idx = 0
    tmp_list = []
    while idx < len(sentence):
        idx += 1
        if sentence[0:idx] in keys:
            cur_element = sentence[0:idx]
            left_list = func(sentence[idx:], Dict)  #先假设返回一个二维数组
            for i in range(len(left_list)):
                left_list[i].insert(0, cur_element)
            if len(left_list) == 0: #处理边界情况
                left_list.append([cur_element]) #二维数组
            tmp_list.extend(left_list) #.extend，不能用append，否则tmp_list就升为维了。
    return tmp_list # 确认是二维数组，闭环假设。


# 目标输出;顺序不重要
target = [
    ['经常', '有意见', '分歧'],
    ['经常', '有意见', '分', '歧'],
    ['经常', '有', '意见', '分歧'],
    ['经常', '有', '意见', '分', '歧'],
    ['经常', '有', '意', '见分歧'],
    ['经常', '有', '意', '见', '分歧'],
    ['经常', '有', '意', '见', '分', '歧'],
    ['经', '常', '有意见', '分歧'],
    ['经', '常', '有意见', '分', '歧'],
    ['经', '常', '有', '意见', '分歧'],
    ['经', '常', '有', '意见', '分', '歧'],
    ['经', '常', '有', '意', '见分歧'],
    ['经', '常', '有', '意', '见', '分歧'],
    ['经', '常', '有', '意', '见', '分', '歧']
]

result = all_cut(sentence, Dict)
for i in result:
    print(str(i))
# print(str(result))
