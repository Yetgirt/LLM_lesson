import torch
import math
import numpy as np
from transformers import BertModel
import torch.nn.functional as F
import torch.nn as nn

'''

通过手动矩阵运算实现Bert结构
模型文件下载 https://huggingface.co/models

'''

bert = BertModel.from_pretrained(r".\bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()
bert.eval()
x = np.array([2450, 15486, 102, 2110])   #假想成4个字的句子
torch_x = torch.LongTensor([x])          #pytorch形式输入
seqence_output, pooler_output = bert(torch_x)
print(seqence_output.shape, pooler_output.shape)
# print(seqence_output, pooler_output)

print(bert.state_dict().keys())  #查看所有的权值矩阵名称

class DiyBert:
    #将预训练好的整个权重字典输入进来
    def __init__(self, state_dict):
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.num_layers = 6        #注意这里的层数要跟预训练config.json文件中的模型层数一致
        self.load_weights_pytorch(state_dict)

    def load_weights(self, state_dict):
        #embedding部分
        self.word_embeddings = state_dict["embeddings.word_embeddings.weight"].numpy()
        self.position_embeddings = state_dict["embeddings.position_embeddings.weight"].numpy()
        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"].numpy()
        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"].numpy()
        self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"].numpy()
        self.transformer_weights = []
        #transformer部分，有多层
        for i in range(self.num_layers):
            q_w = state_dict["encoder.layer.%d.attention.self.query.weight" % i].numpy()
            q_b = state_dict["encoder.layer.%d.attention.self.query.bias" % i].numpy()
            k_w = state_dict["encoder.layer.%d.attention.self.key.weight" % i].numpy()
            k_b = state_dict["encoder.layer.%d.attention.self.key.bias" % i].numpy()
            v_w = state_dict["encoder.layer.%d.attention.self.value.weight" % i].numpy()
            v_b = state_dict["encoder.layer.%d.attention.self.value.bias" % i].numpy()
            attention_output_weight = state_dict["encoder.layer.%d.attention.output.dense.weight" % i].numpy()
            attention_output_bias = state_dict["encoder.layer.%d.attention.output.dense.bias" % i].numpy()
            attention_layer_norm_w = state_dict["encoder.layer.%d.attention.output.LayerNorm.weight" % i].numpy()
            attention_layer_norm_b = state_dict["encoder.layer.%d.attention.output.LayerNorm.bias" % i].numpy()
            intermediate_weight = state_dict["encoder.layer.%d.intermediate.dense.weight" % i].numpy()
            intermediate_bias = state_dict["encoder.layer.%d.intermediate.dense.bias" % i].numpy()
            output_weight = state_dict["encoder.layer.%d.output.dense.weight" % i].numpy()
            output_bias = state_dict["encoder.layer.%d.output.dense.bias" % i].numpy()
            ff_layer_norm_w = state_dict["encoder.layer.%d.output.LayerNorm.weight" % i].numpy()
            ff_layer_norm_b = state_dict["encoder.layer.%d.output.LayerNorm.bias" % i].numpy()
            self.transformer_weights.append([q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias,
                                             attention_layer_norm_w, attention_layer_norm_b, intermediate_weight, intermediate_bias,
                                             output_weight, output_bias, ff_layer_norm_w, ff_layer_norm_b])
        #pooler层
        self.pooler_dense_weight = state_dict["pooler.dense.weight"].numpy()
        self.pooler_dense_bias = state_dict["pooler.dense.bias"].numpy()

    def load_weights_pytorch(self, state_dict):
        #embedding部分
        self.word_embeddings = state_dict["embeddings.word_embeddings.weight"]
        self.position_embeddings = state_dict["embeddings.position_embeddings.weight"]
        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"]
        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"]
        self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"]
        self.transformer_weights = []
        #transformer部分，有多层
        for i in range(self.num_layers):
            q_w = state_dict["encoder.layer.%d.attention.self.query.weight" % i]
            q_b = state_dict["encoder.layer.%d.attention.self.query.bias" % i]
            k_w = state_dict["encoder.layer.%d.attention.self.key.weight" % i]
            k_b = state_dict["encoder.layer.%d.attention.self.key.bias" % i]
            v_w = state_dict["encoder.layer.%d.attention.self.value.weight" % i]
            v_b = state_dict["encoder.layer.%d.attention.self.value.bias" % i]
            attention_output_weight = state_dict["encoder.layer.%d.attention.output.dense.weight" % i]
            attention_output_bias = state_dict["encoder.layer.%d.attention.output.dense.bias" % i]
            attention_layer_norm_w = state_dict["encoder.layer.%d.attention.output.LayerNorm.weight" % i]
            attention_layer_norm_b = state_dict["encoder.layer.%d.attention.output.LayerNorm.bias" % i]
            intermediate_weight = state_dict["encoder.layer.%d.intermediate.dense.weight" % i]
            intermediate_bias = state_dict["encoder.layer.%d.intermediate.dense.bias" % i]
            output_weight = state_dict["encoder.layer.%d.output.dense.weight" % i]
            output_bias = state_dict["encoder.layer.%d.output.dense.bias" % i]
            ff_layer_norm_w = state_dict["encoder.layer.%d.output.LayerNorm.weight" % i]
            ff_layer_norm_b = state_dict["encoder.layer.%d.output.LayerNorm.bias" % i]
            self.transformer_weights.append([q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias,
                                             attention_layer_norm_w, attention_layer_norm_b, intermediate_weight, intermediate_bias,
                                             output_weight, output_bias, ff_layer_norm_w, ff_layer_norm_b])
        #pooler层
        self.pooler_dense_weight = state_dict["pooler.dense.weight"]
        self.pooler_dense_bias = state_dict["pooler.dense.bias"]


    #bert embedding，使用3层叠加，在经过一个Layer norm层
    def embedding_forward(self, x):# x 是np的数组，里面是[1,9,10045,500...],每个数字代表词表中每个词对应的index。
        # x.shape = [max_len]
        we = self.get_embedding_pytorch(self.word_embeddings, x)  # shpae: [max_len, hidden_size]
        # position embeding的输入 [0, 1, 2, 3]
        pe = self.get_embedding_pytorch(self.position_embeddings,  torch.arange(len(x), dtype=torch.long) )  # shpae: [max_len, hidden_size]
        # token type embedding,单输入的情况下为[0, 0, 0, 0]
        te = self.get_embedding_pytorch(self.token_type_embeddings,torch.zeros(len(x), dtype=torch.long) )  # shpae: [max_len, hidden_size]
        embedding = we + pe + te
        # 加和后有一个归一化层
        embedding = self.layer_norm_pytorch(embedding, self.embeddings_layer_norm_weight, self.embeddings_layer_norm_bias,self.hidden_size)  # shpae: [max_len, hidden_size]
        return embedding

    #embedding层实际上相当于按index索引，或理解为onehot输入乘以embedding矩阵
    def get_embedding(self, embedding_matrix, x):
        return np.array([embedding_matrix[index] for index in x])

    def get_embedding_pytorch(self, embedding_matrix, x):
        # 转成 tensor
        emb = torch.tensor(embedding_matrix, dtype=torch.float32) \
            if not isinstance(embedding_matrix, torch.Tensor) else embedding_matrix

        idx = torch.tensor(x, dtype=torch.long) \
            if not isinstance(x, torch.Tensor) else x.long()

        # F.embedding: idx → emb(idx)
        return F.embedding(idx, emb)

    #执行全部的transformer层计算
    def all_transformer_layer_forward(self, x):
        for i in range(self.num_layers):
            x = self.single_transformer_layer_forward_pytorch(x, i)
        return x

    def single_transformer_layer_forward_pytorch(self, x, layer_index):
        weights = self.transformer_weights[layer_index]
        # 取出该层的参数，在实际中，这些参数都是随机初始化，之后进行预训练
        q_w, q_b, \
            k_w, k_b, \
            v_w, v_b, \
            attention_output_weight, attention_output_bias, \
            attention_layer_norm_w, attention_layer_norm_b, \
            intermediate_weight, intermediate_bias, \
            output_weight, output_bias, \
            ff_layer_norm_w, ff_layer_norm_b = weights
        # self attention层
        attention_output = self.self_attention_pytorch(x,
                                               q_w, q_b,
                                               k_w, k_b,
                                               v_w, v_b,
                                               attention_output_weight, attention_output_bias,
                                               self.num_attention_heads,
                                               self.hidden_size)
        # bn层，并使用了残差机制
        x = self.layer_norm_pytorch(x + attention_output, attention_layer_norm_w, attention_layer_norm_b,self.hidden_size)
        # feed forward层
        feed_forward_x = self.feed_forward_pytorch(x,
                                           intermediate_weight, intermediate_bias,
                                           output_weight, output_bias)
        # bn层，并使用了残差机制
        x = self.layer_norm_pytorch(x + feed_forward_x, ff_layer_norm_w, ff_layer_norm_b,self.hidden_size)
        return x

    def self_attention_pytorch(self,
                               x,
                               q_w, q_b,
                               k_w, k_b,
                               v_w, v_b,
                               out_w, out_b,
                               num_heads,
                               hidden_size):

        max_len = x.shape[0]
        head_dim = hidden_size // num_heads

        # ===== 1. Linear: Q K V =====
        q = F.linear(x, q_w, q_b)  # [L, H]
        k = F.linear(x, k_w, k_b)  # [L, H]
        v = F.linear(x, v_w, v_b)  # [L, H]

        # ===== 2. reshape + permute: [L, H] -> [heads, L, head_dim] =====
        q = q.view(max_len, num_heads, head_dim).permute(1, 0, 2)
        k = k.view(max_len, num_heads, head_dim).permute(1, 0, 2)
        v = v.view(max_len, num_heads, head_dim).permute(1, 0, 2)

        # ===== 3. attention score =====
        att = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(head_dim)
        att = F.softmax(att, dim=-1)

        # ===== 4. attention * V =====
        ctx = torch.matmul(att, v)  # [heads, L, head_dim]

        # ===== 5. merge heads =====
        ctx = ctx.permute(1, 0, 2).contiguous()  # [L, heads, head_dim]
        ctx = ctx.view(max_len, hidden_size)  # [L, H]

        # ===== 6. output projection =====
        out = F.linear(ctx, out_w, out_b)  # [L, H]

        return out


    def feed_forward_pytorch(self, x,
                     intermediate_weight,  # [intermediate_size, hidden_size]
                     intermediate_bias,  # [intermediate_size]
                     output_weight,  # [hidden_size, intermediate_size]
                     output_bias):  # [hidden_size]

        # x: [max_len, hidden_size]
        x = F.linear(x, intermediate_weight, intermediate_bias)
        x = F.gelu(x)
        x = F.linear(x, output_weight, output_bias)
        return x

    def layer_norm_pytorch(self, x, w, b, hidden_size, eps=1e-12):
        # 创建一个 LayerNorm，但不使用 PyTorch 初始化的权重
        ln = nn.LayerNorm(hidden_size, eps=eps)

        # 手动把你传进来的 w、b 复制进去（保证行为和原 BERT 一致）
        with torch.no_grad():
            ln.weight.copy_(w)
            ln.bias.copy_(b)

        # 使用 LayerNorm
        return ln(x)

    #链接[cls] token的输出层
    def pooler_output_layer(self, x):
        # dense + tanh
        x = F.linear(x, self.pooler_dense_weight, self.pooler_dense_bias)
        x = torch.tanh(x)
        return x

    #最终输出
    def forward(self, x):
        x = self.embedding_forward(x)
        x = self.to_tensor(x)
        sequence_output = self.all_transformer_layer_forward(x)

        pooler_output = self.pooler_output_layer(sequence_output[0])
        return sequence_output, pooler_output

    def to_tensor(self,arr):
        if isinstance(arr, torch.Tensor):
            return arr
        return torch.tensor(arr, dtype=torch.float32)

#自制
db = DiyBert(state_dict)
diy_sequence_output, diy_pooler_output = db.forward(x)
#torch
torch_sequence_output, torch_pooler_output = bert(torch_x)

print("\n")
print(diy_sequence_output)
print("\n")
print(torch_sequence_output)

# print(diy_pooler_output)
# print(torch_pooler_output)