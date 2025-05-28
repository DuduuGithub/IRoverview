#!/usr/bin/python
import re
import sys
import getopt
import codecs
import json
import os
import requests
import time
from urllib.parse import quote

# 添加英文分词工具
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# 初始化英文分词所需组件
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# 创建词干提取器和停用词集
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# 英文分词和预处理函数
def tokenize_english(text):
    """英文文本分词与预处理"""
    if not text:
        return []
    # 转为小写
    text = text.lower()
    # 只保留字母、数字和空格
    text = re.sub(r'[^\w\s]', ' ', text)
    # 分词
    tokens = word_tokenize(text)
    # 移除停用词、单个字符的词，并进行词干提取
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and len(word) > 1]
    return tokens

# 添加项目根目录到sys.path，避免相对导入问题
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

# 常量定义 - 修改为使用API模式
USE_OFFLINE_MODE = False  # 设置为False以使用API调用
DEEPSEEK_API_KEY = "sk-3b5bb7d385724e038a4496f50a092b55"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

"""
AI增强搜索，将自然语言转换为检索式，并调用相应的搜索功能
现在支持从数据库读取数据
"""
def search(dictionary_file=None, postings_file=None, queries_file=None, output_file=None, query_text=None, use_db=True):
    try:
        # 检查参数
        if use_db and query_text is None:
            raise ValueError("使用数据库模式时必须提供query_text参数")
        
        # 将自然语言转换为结构化查询
        try:
            structured_query = convert_to_structured_query(query_text if query_text else "", 'basic')
            print(f"原始查询: {query_text}" if query_text else "查询为空")
            print(f"转换后的检索式: {structured_query}")
        except Exception as e:
            print(f"查询转换失败: {e}")
            # 当转换失败时，直接返回错误信息或空结果
            return []
        
        # 判断使用哪种检索模式
        is_advanced_query = is_advanced_query_format(structured_query)
        
        # 执行检索
        if is_advanced_query:
            print("使用高级检索...")
            if use_db:
                # 动态导入，避免循环引用
                from ..proSearch.search import search as advanced_search_engine
                return advanced_search_engine(query_text=structured_query, use_db=True)
            else:
                from ..proSearch.search import search as advanced_search_engine
                return advanced_search_engine(dictionary_file, postings_file, queries_file, output_file)
        else:
            print("使用基本检索...")
            if use_db:
                # 动态导入，避免循环引用
                from ..basicSearch.search import search as basic_search_engine
                return basic_search_engine(query_text=structured_query, use_db=True)
            else:
                from ..basicSearch.search import search as basic_search_engine
                return basic_search_engine(dictionary_file, postings_file, queries_file, output_file)
                
    except Exception as e:
        print(f"AI搜索出现错误: {e}")
        # 异常情况下返回空结果
        return []

def clean_api_response(response_text):
    """清理 API 返回的响应，移除可能的 markdown 格式和换行符"""
    # 移除可能的 markdown 代码块标记
    response_text = re.sub(r'^```json\s*', '', response_text)
    response_text = re.sub(r'\s*```$', '', response_text)
    # 移除可能的其他 markdown 格式
    response_text = re.sub(r'^```\s*', '', response_text)
    # 将多行字符串转换为单行，但保留换行符的转义
    response_text = re.sub(r'\n', '\\n', response_text)
    return response_text.strip()

def convert_to_structured_query(nl_query, search_mode='basic'):
    if not nl_query:
        return ""  # 如果查询为空，返回空字符串
    
    # 如果未设置API密钥，直接报错
    if not DEEPSEEK_API_KEY:
        raise ValueError("未设置API密钥，无法转换查询")
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }
        
        # 支持中英文的提示词，生成特定格式的检索式字符串
        prompt = f"""You are a search query generator for academic literature. Your task is to convert natural language queries (in Chinese or English) into a structured search expression string.

Rules:
1. Generate a multi-line string where each line represents a search condition.
2. Each line must start with a boolean operator (AND, OR, NOT). The first line should also start with 'and'.
3. Use the format: [operator] field: value
4. Only include fields relevant to the query.
5. Keep field values simple and focused.
6. Use proper English terms for technical concepts in field values.
7. Supported fields and operators are based on the search system's capabilities (AND, OR, NOT, field:value). Use parentheses `()` to group terms connected by OR.
8. IMPORTANT: The response must be a valid JSON string, without any markdown formatting (like ```json) or extra text before or after the JSON. The JSON should have the following structure:
{{"structured_query_string": "[operator] field1: value1\\n[operator] field2: value2\\n..."}}

Examples in English:
1. "recent papers about machine learning in healthcare"
{{"structured_query_string": "and title: machine learning\\nand abstract: healthcare"}}

2. "papers by John Smith or David Lee about deep learning"
{{"structured_query_string": "and title: deep learning\\nand author: (John Smith OR David Lee)"}}

3. "papers by John Smith except those about biology"
{{"structured_query_string": "and author: John Smith\\nnot abstract: biology"}}

4. "papers on neural networks or deep learning by Zhang San or Li Si"
{{"structured_query_string": "and title: (neural networks OR deep learning)\\nand author: (Zhang San OR Li Si)"}}

Examples in Chinese:
1. "医疗领域的机器学习研究"
{{"structured_query_string": "and title: machine learning\\nand abstract: healthcare"}}

2. "张三或李四发表的深度学习论文"
{{"structured_query_string": "and title: deep learning\\nand author: (Zhang San OR Li Si)"}}

3. "海岩除了在北京大学期间发表的全部论文"
{{"structured_query_string": "and author: haiyan\\nnot institution: peking university"}}

4. "神经网络或深度学习方面的论文，作者为张三或李四"
{{"structured_query_string": "and title: (neural networks OR deep learning)\\nand author: (Zhang San OR Li Si)"}}

Important:
1. For Chinese queries, convert key terms to English in the search expression string.
2. Each line in the \"structured_query_string\" MUST follow the format: [operator] field: value.
3. The operator must be one of: and, or, not.
4. The first line should also start with an operator, preferably 'and'.
5. Maintain the original meaning and intent while simplifying the expression.
6. Use parentheses `()` to group terms connected by OR.
7. The response must be a valid single-line JSON string with escaped newlines.

Natural language query: {nl_query}
Search mode: {search_mode}
Structured Query String:"""
        
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,  # 可以根据需要调整
            "temperature": 0.1  # 保持低温度以确保输出的一致性
        }
        
        # 设置请求超时
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=15)
        
        if response.status_code != 200:
            error_msg = f"API请求失败，状态码：{response.status_code}"
            print(error_msg)
            raise ConnectionError(error_msg)
            
        try:
            result = response.json()
            ai_response = result["choices"][0]["message"]["content"].strip()
            
            # 清理 API 响应
            cleaned_response = clean_api_response(ai_response)
            
            # 尝试解析 JSON
            try:
                parsed_response = json.loads(cleaned_response)
                if "structured_query_string" in parsed_response:
                    # 将转义的换行符转换回实际的换行符
                    return parsed_response["structured_query_string"].replace('\\n', '\n')
                else:
                    print("API返回的JSON中没有structured_query_string字段")
                    return ""
            except json.JSONDecodeError as e:
                print(f"无法解析API返回的JSON: {e}")
                print(f"API返回的原始文本: {cleaned_response}")
                return ""
                
        except Exception as e:
            print(f"处理API响应时出错: {e}")
            return ""
            
    except Exception as e:
        print(f"API调用失败: {e}")
        return ""

"""
判断是否为高级查询格式
"""
def is_advanced_query_format(query):
    # 如果查询为空，不是高级查询
    if not query:
        return False
        
    # 如果包含字段指示符，可能是高级查询
    field_indicators = ["title:", "author:", "keyword:", "time:", "content:", "institution:", "abstract:"]
    for indicator in field_indicators:
        if indicator in query.lower():
            return True
            
    # 包含时间范围
    if re.search(r'time:\d{4}(?:-\d{1,2})?(?:-\d{1,2})?~\d{4}(?:-\d{1,2})?(?:-\d{1,2})?', query.lower()):
        return True
    
    # 包含引号括起来的短语
    if re.search(r'"[^"]+"', query):
        return True
    
    # 如果不包含上述指示符，则为基本查询
    return False

"""
打印使用说明
"""
def print_usage():
    print("用法: python ai_search.py -q '自然语言查询'")

# 命令行入口
if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hq:", ["help", "query="])
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)
        
    query = ""
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_usage()
            sys.exit()
        elif opt in ("-q", "--query"):
            query = arg
    if not query:
        print_usage()
        sys.exit(2)
            
    # 执行转换
    try:
        structured = convert_to_structured_query(query)
        print(f"原始查询: {query}")
        print(f"结构化查询: {structured}")
        print(f"是高级查询: {is_advanced_query_format(structured)}")
    except Exception as e:
        print(f"查询转换失败: {e}")
        sys.exit(1)
