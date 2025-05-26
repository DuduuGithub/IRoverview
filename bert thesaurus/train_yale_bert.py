# 一个词多次出现，每次出现都预测一次，记录预测分数
# 如果多次预测的词中有相同的词，则将同一个词的预测值累加，
# 最后输出每个预测值之间具有强关联的词组。

import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM
import torch
import string
import re
from collections import defaultdict
import json
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from itertools import combinations

# Download required NLTK data
#nltk.download('wordnet')
#nltk.download('stopwords')

# 加载预训练的BERT模型和分词器
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#model = BertForMaskedLM.from_pretrained('bert-base-uncased')
# 使用科学领域的预训练模型
model = BertForMaskedLM.from_pretrained('allenai/scibert_scivocab_uncased')
tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

# 初始化词形还原器
lemmatizer = WordNetLemmatizer()

# 获取停用词列表
stopwords = set(stopwords.words('english'))

# 创建叙词表数据结构
thesaurus = defaultdict(set)
# 存储预测分数
prediction_scores = defaultdict(dict)

def normalize_word(word):
    """标准化词形"""
    # 转换为小写
    word = word.lower()
    # 去除标点符号
    word = re.sub(r'[^\w\s]', '', word)
    # 词形还原
    word = lemmatizer.lemmatize(word)
    return word

def is_valid_prediction(word, original_word):
    """判断预测词是否有效"""
    # 排除太短的词
    if len(word) <= 1:
        return False
    # 排除纯数字
    if word.isdigit():
        return False
    # 排除与原始词相同的词
    if word == original_word:
        return False
    # 排除停用词
    if word in stopwords:
        return False
    return True


# 加载数据
corpus_df = pd.read_parquet(r'\sort_ai\yale_dataset\corpus_new\corpus.parquet')

# 使用文献的摘要和全文进行处理
texts = corpus_df['abstract'].tolist() + corpus_df['full_paper'].tolist()

# 处理前n个文本
processed_count = 0
for text in texts[:100]:
    processed_count += 1
    if processed_count % 10 == 0:
        print(f"已处理 {processed_count} 条文本...")
        
    # 去除空格和特殊字符
    text = re.sub(r'\s+', ' ', text)  # 去除多余的空格
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # 去除非ASCII字符

    # 将文本分块，增加上下文窗口
    tokens = tokenizer.tokenize(text)
    # 修改为512的块大小
    chunks = [tokens[i:i + 512] for i in range(0, len(tokens), 512)]

    for chunk in chunks:
        # 将块转换回文本
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        
        # 分词并排除标点符号
        words = chunk_text.split()
        for i, word in enumerate(words):
            if word in string.punctuation:
                continue
                
            # 标准化当前词
            original_word = normalize_word(words[i])
            if not original_word:  # 跳过空词
                continue
                
            # 遮盖当前词
            words[i] = '[MASK]'
            masked_text = ' '.join(words)

            # 将文本转换为BERT输入格式
            inputs = tokenizer(masked_text, return_tensors='pt', truncation=True, max_length=512)

            # 使用模型进行预测
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = outputs.logits

            # 获取[MASK]位置的预测结果
            try:
                masked_index = inputs['input_ids'][0].tolist().index(tokenizer.mask_token_id)
                # 获取预测分数
                predicted_scores = predictions[0, masked_index].softmax(dim=0)
                # 获取前5个预测及其分数
                top5_scores, top5_indices = predicted_scores.topk(5)
                predicted_token_ids = top5_indices.tolist()
                
                # 存储预测分数
                for score, token_id in zip(top5_scores, predicted_token_ids):
                    pred_word = normalize_word(tokenizer.convert_ids_to_tokens([token_id])[0])
                    if is_valid_prediction(pred_word, original_word):
                        # 如果预测词已经存在，累加分数
                        if pred_word in prediction_scores[original_word]:
                            prediction_scores[original_word][pred_word] += score.item()
                        else:
                            prediction_scores[original_word][pred_word] = score.item()
                
            except ValueError:
                continue

            # 恢复原始词
            words[i] = original_word

# 将预测结果转换为列表并排序
sorted_predictions = []
for original_word, predictions in prediction_scores.items():
    for pred_word, score in predictions.items():
        sorted_predictions.append({
            'original_word': original_word,
            'predicted_word': pred_word,
            'score': float(score)  # 确保分数是浮点数
        })

# 按分数从高到低排序
sorted_predictions.sort(key=lambda x: x['score'], reverse=True)

# 保存结果前再次确认排序
sorted_predictions = sorted(sorted_predictions, key=lambda x: x['score'], reverse=True)

# 保存结果
result = {
    'predictions': sorted_predictions
}

with open('bert_thesaurus_scores1-100.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"叙词表已生成，共包含 {len(sorted_predictions)} 个词对")
print("结果已按预测分数从高到低排序") 
            