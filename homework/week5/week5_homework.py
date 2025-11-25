#!/usr/bin/env python3
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        #kmeans.cluester_centers_ #每个聚类中心
        sentence_label_dict[label].append(sentence)         #同标签的放到一起

    #类内距离排序
    order_dict = {}
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    for i in range(n_clusters):
        cur_cluster_points = vectors[labels == i]
        if len(cur_cluster_points) == 0:
            continue
        center = centers[i]
        avg_dist = np.linalg.norm(cur_cluster_points - center, axis=1).mean()
        order_dict[i] = avg_dist
    sorted_clusters = sorted(order_dict.items(), key=lambda x: x[1])

    #打印结果
    print("\n" + "=" * 60)
    print("聚类紧凑度排序（类内平均距离从小到大）")
    print("=" * 60)
    for rank, (cluster_id, avg_dist) in enumerate(sorted_clusters, 1):
        sample_count = np.sum(labels == cluster_id)
        print(f"第 {rank:2d} 名 → 簇 {cluster_id:2d} | 样本数: {sample_count:3d} | 类内平均距离: {avg_dist:.4f}")
if __name__ == "__main__":
    main()

