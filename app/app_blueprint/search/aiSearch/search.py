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
import jieba

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
            structured_query = convert_to_structured_query(query_text if query_text else "")
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

"""
调用deepseek API将自然语言转换为结构化检索式
"""
def convert_to_structured_query(nl_query):
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
        
        # 优化的prompt
        prompt = f"""作为一个学术文献检索系统的核心组件，你需要将用户的自然语言查询转换为精确的布尔检索式或高级检索式。

请遵循以下格式规范：
1. 若是简单关键词查询，直接返回关键词。例如：
   - 输入："机器学习"
   - 输出："机器学习"

2. 若包含明确的逻辑关系，使用AND、OR、NOT等布尔操作符连接。例如：
   - 输入："找到同时包含深度学习和图像识别的论文"
   - 输出："深度学习 AND 图像识别"
   - 输入："人工智能或机器学习"
   - 输出："人工智能 OR 机器学习"
   - 输入："计算机视觉但不包含人脸识别"
   - 输出："计算机视觉 NOT 人脸识别"

3. 当提到特定字段，使用field:value格式。支持的字段包括：
   - title: 标题
   - author: 作者
   - keyword: 关键词
   - time: 时间
   - institution: 机构
   - abstract: 摘要
   - content: 全文

   例如：
   - 输入："查找张三发表的关于机器学习的论文"
   - 输出："author:张三 AND 机器学习"
   - 输入："标题包含深度学习的论文"
   - 输出："title:深度学习"

4. 对于时间范围，使用time:YYYY-MM-DD~YYYY-MM-DD格式。例如：
   - 输入："2020年到2022年的人工智能论文"
   - 输出："time:2020~2022 AND 人工智能"
   - 输入："2018年3月之后发表的深度学习论文"
   - 输出："time:2018-03~2025-05-11 AND 深度学习"

仅返回转换后的检索式，不要包含任何解释或多余的文字。

自然语言查询: {nl_query}
检索式:"""
        
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0.1  # 低温度使输出更确定
        }
        
        # 设置请求超时
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=5)
        
        if response.status_code != 200:
            error_msg = f"API请求失败，状态码：{response.status_code}"
            print(error_msg)
            raise ConnectionError(error_msg)
            
        try:
            result = response.json()
            structured_query = result["choices"][0]["message"]["content"].strip()
            return structured_query
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            error_msg = f"解析API响应出错: {e}"
            print(error_msg)
            raise ValueError(error_msg)
        
    except Exception as e:
        error_msg = f"API调用失败: {e}"
        print(error_msg)
        raise RuntimeError(error_msg)

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
