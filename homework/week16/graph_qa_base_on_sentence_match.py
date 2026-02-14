import re
import json
import pandas
import itertools
from neo4j import GraphDatabase
from config import Config

class GraphQA:
    def __init__(self, auth, uri="neo4j://localhost:7687"):
        """
        初始化时传入连接信息（推荐从配置文件或环境变量读取）
        """
        self.driver = GraphDatabase.driver(uri, auth=auth)
        schema_path = "kg_schema.json"
        templet_path = "question_templet.xlsx"
        self.load(schema_path, templet_path)
        print("知识图谱问答系统加载完毕！\n===============")

        # 测试连接（可选，生产环境可移除）
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("Neo4j 连接正常")
        except Exception as e:
            print("连接失败:", e)
            raise

    def __del__(self):
        """对象销毁时关闭 driver（可选，但推荐）"""
        if hasattr(self, 'driver'):
            self.driver.close()

    # 加载模板 & schema（不变）
    def load(self, schema_path, templet_path):
        self.load_kg_schema(schema_path)
        self.load_question_templet(templet_path)

    def load_kg_schema(self, path):
        with open(path, encoding="utf8") as f:
            schema = json.load(f)
        self.relation_set = set(schema["relations"])
        self.entity_set = set(schema["entitys"])
        self.label_set = set(schema["labels"])
        self.attribute_set = set(schema["attributes"])

    def load_question_templet(self, templet_path):
        dataframe = pandas.read_excel(templet_path)
        self.question_templet = []
        for index in range(len(dataframe)):
            question = dataframe["question"][index]
            cypher = dataframe["cypher"][index]
            cypher_check = dataframe["check"][index]
            answer = dataframe["answer"][index]
            self.question_templet.append([question, cypher, json.loads(cypher_check), answer])

    # 信息抽取方法（不变）
    def get_mention_entitys(self, sentence):
        return re.findall("|".join(self.entity_set), sentence)

    def get_mention_relations(self, sentence):
        return re.findall("|".join(self.relation_set), sentence)

    def get_mention_attributes(self, sentence):
        return re.findall("|".join(self.attribute_set), sentence)

    def get_mention_labels(self, sentence):
        return re.findall("|".join(self.label_set), sentence)

    def parse_sentence(self, sentence):
        entitys = self.get_mention_entitys(sentence)
        relations = self.get_mention_relations(sentence)
        labels = self.get_mention_labels(sentence)
        attributes = self.get_mention_attributes(sentence)
        return {
            "%ENT%": entitys,
            "%REL%": relations,
            "%LAB%": labels,
            "%ATT%": attributes
        }

    # 组合生成逻辑（不变）
    def decode_value_combination(self, value_combination, cypher_check):
        res = {}
        for index, (key, required_count) in enumerate(cypher_check.items()):
            if required_count == 1:
                res[key] = value_combination[index][0]
            else:
                for i in range(required_count):
                    key_num = key[:-1] + str(i) + "%"
                    res[key_num] = value_combination[index][i]
        return res

    def get_combinations(self, cypher_check, info):
        slot_values = []
        for key, required_count in cypher_check.items():
            slot_values.append(itertools.combinations(info[key], required_count))
        value_combinations = itertools.product(*slot_values)
        combinations = []
        for value_combination in value_combinations:
            combinations.append(self.decode_value_combination(value_combination, cypher_check))
        return combinations

    def replace_token_in_string(self, string, combination):
        for key, value in combination.items():
            string = string.replace(key, value)
        return string

    def expand_templet(self, templet, cypher, cypher_check, info, answer):
        combinations = self.get_combinations(cypher_check, info)
        templet_cypher_pair = []
        for combination in combinations:
            replaced_templet = self.replace_token_in_string(templet, combination)
            replaced_cypher = self.replace_token_in_string(cypher, combination)
            replaced_answer = self.replace_token_in_string(answer, combination)
            templet_cypher_pair.append([replaced_templet, replaced_cypher, replaced_answer])
        return templet_cypher_pair

    def check_cypher_info_valid(self, info, cypher_check):
        for key, required_count in cypher_check.items():
            if len(info.get(key, [])) < required_count:
                return False
        return True

    def expand_question_and_cypher(self, info):
        templet_cypher_pair = []
        for templet, cypher, cypher_check, answer in self.question_templet:
            if self.check_cypher_info_valid(info, cypher_check):
                templet_cypher_pair += self.expand_templet(templet, cypher, cypher_check, info, answer)
        return templet_cypher_pair

    # 相似度函数（不变）
    def sentence_similarity_function(self, string1, string2):
        jaccard_distance = len(set(string1) & set(string2)) / len(set(string1) | set(string2))
        return jaccard_distance

    # 核心变更：使用 driver.execute_query 替换 graph.run
    def _run_cypher(self, cypher):
        """
        执行 Cypher 并返回 .data() 格式的结果列表
        """
        try:
            # execute_query 是 5.x+ 推荐方式，简洁且自动处理事务
            records, _, _ = self.driver.execute_query(cypher)
            return [dict(record) for record in records]  # 转成 dict 列表，与 py2neo .data() 一致
        except Exception as e:
            print(f"Cypher 执行失败: {cypher}\n错误: {e}")
            return []

    def cypher_match(self, sentence, info):
        templet_cypher_pair = self.expand_question_and_cypher(info)
        # print(templet_cypher_pair)  # 调试时可打开

        result = []
        for templet, cypher, answer in templet_cypher_pair:
            score = self.sentence_similarity_function(sentence, templet)
            # print(sentence, templet, score)  # 调试时可打开
            result.append([templet, cypher, score, answer])

        result.sort(key=lambda x: x[2], reverse=True)
        return result

    def parse_result(self, graph_search_result, answer, info):
        if not graph_search_result:
            return None

        result = graph_search_result[0]
        # 关系查找返回的结果形式较为特殊，单独处理
        if "REL" in result and hasattr(result["REL"], 'types'):
            result["REL"] = list(result["REL"].types())[0]

        answer = self.replace_token_in_string(answer, result)
        return answer

    # 对外问答接口（核心变更在这里）
    def query(self, sentence):
        print("============")
        print("问题:", sentence)

        info = self.parse_sentence(sentence)
        print("抽取信息:", info)

        templet_cypher_score = self.cypher_match(sentence, info)

        for templet, cypher, score, answer in templet_cypher_score:
            print(f"尝试模板: {templet} (相似度: {score:.4f})")
            graph_search_result = self._run_cypher(cypher)

            if graph_search_result:
                final_answer = self.parse_result(graph_search_result, answer, info)
                if final_answer:
                    print("匹配到答案，使用 Cypher:", cypher)
                    return final_answer

        print("所有模板均无结果")
        return None

if __name__ == "__main__":
    # 根据你的实际情况修改 URI / 密码
    # 推荐从环境变量读取：import os; uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    URI = Config["URI"]
    AUTH = Config["AUTH"]
    qa = GraphQA(
        uri=URI,  # 或 "bolt://127.0.0.1:7687"
        auth = AUTH
    )

    questions = [
        "谁导演的不能说的秘密",
        "发如雪的谱曲是谁",
        "爱在西元前的谱曲是谁",
        "周杰伦的星座是什么",
        "周杰伦的血型是什么",
        "周杰伦的身高",
        "周杰伦和淡江中学是什么关系",
        "周杰伦岁数是什么",
        "谁演唱的青花瓷"
    ]

    for q in questions:
        res = qa.query(q)
        print("答案:", res if res else "未找到答案")
        print("-" * 60)
