import json
import nltk
from nltk.corpus import wordnet
import re
from collections import Counter

def load_predictions(file_path):
    """加载预测结果"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 处理不同的JSON格式
    if isinstance(data, dict) and 'predictions' in data:
        return data['predictions']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Invalid JSON format: expected either a list of predictions or a dict with 'predictions' key")

def get_word_pos(word):
    """获取词的所有词性"""
    pos_dict = {
        'n': 'n',  # noun
        'v': 'v',  # verb
        'a': 'a',  # adjective
        's': 's',  # adjective satellite
        'r': 'r'   # adverb
    }
    
    synsets = wordnet.synsets(word)
    if not synsets:
        return []
    
    # 获取所有词性
    pos_list = []
    for synset in synsets:
        pos = pos_dict.get(synset.pos(), '?')  # 未知词性用?表示
        if pos not in pos_list:
            pos_list.append(pos)
    
    return pos_list

def is_content_word(word):
    """检查是否是实词（名词、动词、形容词、副词）"""
    # 1. 基本规则检查
    if not word or len(word) < 2:  # 过滤掉空词和单字符词
        return False
    
    # 2. 检查是否包含特殊字符
    if re.search(r'[^a-zA-Z-]', word):  # 只允许字母和连字符
        return False
    
    # 3. 检查是否在WordNet中且是实词
    synsets = wordnet.synsets(word)
    if not synsets:
        return False
    
    # 检查是否至少有一个实词词性
    valid_pos = {'n', 'v', 'a', 's', 'r'}  # 名词、动词、形容词、形容词卫星词、副词
    for synset in synsets:
        if synset.pos() in valid_pos:
            return True
    
    # 4. 检查是否是常见缩写
    common_abbreviations = {'pde', 'ode', 'gan', 'bert', 'gpt', 'llm', 'ai', 'ml', 'dl', 'nlp', 'cv'}
    if word.lower() in common_abbreviations:
        return True
    
    return False

def filter_predictions(predictions, min_score=1):
    """过滤预测结果"""
    filtered_predictions = []
    pos_stats = Counter()  # 用于统计词性分布
    
    for pred in predictions:
        original_word = pred['original_word']
        predicted_word = pred['predicted_word']
        score = float(pred['score'])
        
        # 检查原始词和预测词是否有效
        if (is_content_word(original_word) and 
            is_content_word(predicted_word) and 
            score >= min_score):
            # 获取词性信息
            original_pos = get_word_pos(original_word)
            predicted_pos = get_word_pos(predicted_word)
            
            # 更新词性统计
            pos_stats.update(original_pos)
            pos_stats.update(predicted_pos)
            
            # 添加词性信息到预测结果
            pred['original_word_pos'] = original_pos
            pred['predicted_word_pos'] = predicted_pos
            filtered_predictions.append(pred)
    
    return filtered_predictions, pos_stats

def save_filtered_predictions(predictions, output_file):
    """保存过滤后的预测结果"""
    data = {
        'predictions': predictions
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"过滤后的预测结果已保存到 {output_file}")

def main():
    # 设置输入输出文件
    input_file = 'bert_thesaurus_scores1-100.json'
    intermediate_file = 'filtered_thesaurus_len.json'
    final_output_file = 'filtered_thesaurus_final.json'
    
    # 下载必要的NLTK数据
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    
    # 第一步：加载原始预测结果
    print("正在加载原始预测结果...")
    predictions = load_predictions(input_file)
    print(f"原始预测数量: {len(predictions)}")
    
    # 第二步：过滤长度小于等于2的词
    print("\n第一步：过滤长度小于等于2的词...")
    len_filtered_predictions = [
        p for p in predictions
        if len(p['original_word']) > 2 and len(p['predicted_word']) > 2
    ]
    print(f"长度过滤后数量: {len(len_filtered_predictions)}")
    print(f"长度过滤掉的数量: {len(predictions) - len(len_filtered_predictions)}")
    
    # 保存中间结果
    save_filtered_predictions(len_filtered_predictions, intermediate_file)
    
    # 第三步：过滤非实词，并添加词性信息
    print("\n第二步：过滤非实词并添加词性信息...")
    final_filtered_predictions, pos_stats = filter_predictions(len_filtered_predictions)
    print(f"词性过滤后数量: {len(final_filtered_predictions)}")
    print(f"词性过滤掉的数量: {len(len_filtered_predictions) - len(final_filtered_predictions)}")
    
    # 保存最终结果
    save_filtered_predictions(final_filtered_predictions, final_output_file)
    
    # 打印总体统计信息
    print("\n总体统计信息:")
    print(f"原始预测数量: {len(predictions)}")
    print(f"最终预测数量: {len(final_filtered_predictions)}")
    print(f"总共过滤掉的数量: {len(predictions) - len(final_filtered_predictions)}")
    
    # 打印词性分布统计
    print("\n词性分布统计:")
    total_words = sum(pos_stats.values())
    for pos, count in pos_stats.most_common():
        percentage = (count / total_words) * 100
        print(f"{pos}: {count} ({percentage:.1f}%)")
    
    # 打印一些示例词对及其词性
    print("\n示例词对及其词性:")
    for i, pred in enumerate(final_filtered_predictions[:5]):  # 只显示前5个示例
        print(f"\n示例 {i+1}:")
        print(f"原词: {pred['original_word']} (POS: {', '.join(pred['original_word_pos'])})")
        print(f"预测词: {pred['predicted_word']} (POS: {', '.join(pred['predicted_word_pos'])})")
        print(f"分数: {pred['score']}")

if __name__ == "__main__":
    main() 