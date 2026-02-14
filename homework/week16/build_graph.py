import re
import json

from collections import defaultdict
from neo4j import GraphDatabase
from config import Config
'''
读取三元组，并将数据写入neo4j
'''


#连接图数据库
URI = Config["URI"]        # 或 "bolt://127.0.0.1:7687"
AUTH = Config["AUTH"]
# graph = Graph("neo4j://127.0.0.1:7687",auth=("neo4j","demo"))
driver = GraphDatabase.driver(URI, auth=AUTH)

attribute_data = defaultdict(dict)
relation_data = defaultdict(dict)
label_data = {}

in_graph_entity = set()   # 用于追踪已处理的实体（避免重复处理）
# 有的实体后面有括号，里面的内容可以作为标签
# 提取到标签后，把括号部分删除

# ── 工具函数：提取括号标签并清理实体名 ──────────────────
def get_label_then_clean(x, label_data):
    match = re.search(r"（.+）", x)
    if match:
        label_string = match.group()
        cleaned = re.sub(r"（.+）", "", x).strip()
        for possible_label in ["歌曲", "专辑", "电影", "电视剧"]:
            if possible_label in label_string:
                label_data[cleaned] = possible_label
                return cleaned
        return cleaned
    return x.strip()

# ── 读取关系三元组 ───────────────────────────────────────
with open("triplets_head_rel_tail.txt", encoding="utf8") as f:
    for line in f:
        if not line.strip():
            continue
        parts = line.strip().split("\t")
        if len(parts) != 3:
            continue
        head, relation, tail = parts
        head = get_label_then_clean(head, label_data)
        tail = get_label_then_clean(tail, label_data)   # tail 也可能有标签
        relation_data[head][relation] = tail

# ── 读取属性三元组 ───────────────────────────────────────
with open("triplets_enti_attr_value.txt", encoding="utf8") as f:
    for line in f:
        if not line.strip():
            continue
        parts = line.strip().split("\t")
        if len(parts) != 3:
            continue
        entity, attribute, value = parts
        entity = get_label_then_clean(entity, label_data)
        attribute_data[entity][attribute] = value.strip()

# ── 收集所有唯一实体 ─────────────────────────────────────
all_entities = set()
for entity in attribute_data:
    all_entities.add(entity)
    attribute_data[entity]["NAME"] = entity   # 强制加 NAME 属性

for head in relation_data:
    all_entities.add(head)
    for tail in relation_data[head].values():
        all_entities.add(tail)

# ── 准备节点参数列表 ─────────────────────────────────────
entity_list = []
for entity in all_entities:
    props = attribute_data[entity].copy()  # 包含 NAME
    label = label_data.get(entity)
    entity_list.append({
        "id": entity,           # 用作唯一标识（假设 NAME 唯一）
        "props": props,
        "label": label if label else None
    })

# ── 准备关系参数（按关系类型分组）─────────────────────────
rels_by_type = defaultdict(list)

for head, rel_dict in relation_data.items():
    for rel_type, tail in rel_dict.items():
        # 简单安全检查：关系类型只能是字母、数字、下划线
        if not all(c.isalnum() or c == '_' for c in rel_type):
            print(f"跳过不安全的的关系类型: {rel_type}")
            continue
        rels_by_type[rel_type].append({
            "head": head,
            "tail": tail
        })
# 给只有关系的实体也加 NAME
for entity in all_entities:
    if entity not in attribute_data:
        attribute_data[entity]["NAME"] = entity

# ── 执行写入 ──────────────────────────────────────────────

def create_nodes(tx):
    query = """
    UNWIND $rows AS row
    CALL {
        WITH row
        MERGE (n {NAME: row.id})
        SET n += row.props
        FOREACH (lbl IN CASE WHEN row.label IS NOT NULL THEN [row.label] ELSE [] END |
            SET n:`%s`   // 动态标签通过拼接（Neo4j 不支持参数化标签）
        )
    }
    """ % ""   # 占位，实际在下面处理动态标签

    # 因为 Neo4j 不支持参数化标签名，需要为有标签的和无标签的分开处理
    # 这里简化：分开两条语句
    print("-----------创建节点")
    # 无标签实体
    no_label_rows = [r for r in entity_list if r["label"] is None]
    if no_label_rows:
        query_no_label = """
            UNWIND $rows AS row
            MERGE (n {NAME: row.id})
            SET n += row.props
        """
        print(query_no_label)
        tx.run(query_no_label, rows=no_label_rows)

    # 有标签实体（每种标签单独跑，或合并处理）
    label_groups = defaultdict(list)
    for row in entity_list:
        if row["label"]:
            label_groups[row["label"]].append(row)

    for lbl, rows in label_groups.items():
        if not rows:
            continue
        query_with_label = f"""
            UNWIND $rows AS row
            MERGE (n:{lbl} {{NAME: row.id}})
            SET n += row.props
        """
        print(query_with_label)
        tx.run(query_with_label, rows=rows)

def create_relationships(tx):
    print("-----------创建关系")
    for rel_type, rows in rels_by_type.items():
        if not rows:
            continue
        query = f"""
        UNWIND $rows AS row
        MATCH (a {{NAME: row.head}})
        MATCH (b {{NAME: row.tail}})
        MERGE (a)-[:{rel_type}]->(b)
        """
        print(query)
        tx.run(query, rows=rows)

# ── 事务执行 ──────────────────────────────────────────────
with driver.session() as session:
    # 创建所有节点
    session.execute_write(create_nodes)
    print("所有节点创建/更新完成")

    # 创建所有关系
    session.execute_write(create_relationships)
    print("所有关系创建完成")


# ── 生成 schema json（保持原逻辑） ────────────────────────
data = defaultdict(set)

for head in relation_data:
    data["entitys"].add(head)
    for relation, tail in relation_data[head].items():
        data["relations"].add(relation)
        data["entitys"].add(tail)

for enti, label in label_data.items():
    data["entitys"].add(enti)
    data["labels"].add(label)

for enti in attribute_data:
    for attr in attribute_data[enti]:
        data["attributes"].add(attr)
    data["entitys"].add(enti)

data = {k: list(v) for k, v in data.items()}

with open("kg_schema.json", "w", encoding="utf8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("kg_schema.json 已生成")

driver.close()
