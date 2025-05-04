# IR Ranking Model

这是一个基于深度学习的信息检索排序模型，用于对检索结果进行重排序，以提高检索结果的相关性。

## 项目结构

```
.
├── README.md           # 项目说明文档
├── requirements.txt    # 项目依赖
└── model.py           # 模型架构实现
```

## 模型架构

该模型包含以下主要组件：

1. 检索词编码器

   - 使用预训练BERT模型对检索词进行编码
   - 获取检索词的向量表示
2. 文档信息编码器

   - 对文档的标题和摘要进行加权编码
   - 权重是可学习的参数
3. 交互层

   - 将检索词向量和文档向量进行拼接
   - 通过多层神经网络计算相关性得分

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 初始化模型：

```python
model = IRRankingModel(bert_model_name='bert-base-uncased')
```

2. 准备输入数据：

   - 检索词需要通过BERT tokenizer进行处理
   - 文档的标题和摘要需要预先编码为向量
3. 获取排序得分：

```python
scores = model(query_input_ids, query_attention_mask, title_vectors, abstract_vectors)
```
