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

"""
调用deepseek API将自然语言转换为结构化检索式
"""
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
        
        # 创建统一的Prompt，关注用户意图而非UI模式
        prompt = f"""作为一个学术文献检索系统的智能助手，你需要分析用户的自然语言查询，
提取关键信息，并将其转换为适当的检索式。

请先分析用户意图，识别以下要素：
1. 核心主题或关键词
2. 作者信息
3. 发表年份或时间范围
4. 标题中包含的特定词语
5. 机构信息
6. 逻辑关系（AND, OR, NOT）

请遵循以下规则生成检索式：
1. 当前模式是：{search_mode}
2. 在基本模式下：
   - 生成简单的布尔表达式，如 "深度学习 AND 图像识别"
   - 将时间、作者等过滤条件单独记录，而不包含在检索式中
   
3. 在高级模式下：
   - 使用字段限定语法，如 "title:\"深度学习\" AND author:\"李明\""
   - 对时间范围使用 year:[起始年 TO 终止年] 语法

你的回复必须是一个JSON格式，包含以下字段：
{{
  "query": "检索式字符串",
  "has_time_filter": true/false,
  "time_range": [起始年,终止年],
  "has_author_filter": true/false,
  "authors": ["作者1", "作者2"],
  "has_institution_filter": true/false,
  "institutions": ["机构1", "机构2"]
}}

对于基本模式，将所有过滤器信息记录在各自字段中，但"query"字段只包含简单的布尔表达式
对于高级模式，可以在"query"字段中使用完整的字段限定语法

自然语言查询: {nl_query}
搜索模式: {search_mode}
检索式:"""
        
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
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
            ai_response = result["choices"][0]["message"]["content"].strip()
            
            # 尝试解析JSON结果
            try:
                response_json = json.loads(ai_response)
                
                # 获取检索式
                structured_query = response_json.get("query", "")
                
                # 记录额外的过滤条件信息，供后端使用
                # 这些信息可以用于后端过滤，不必全部包含在检索式中
                has_time_filter = response_json.get("has_time_filter", False)
                time_range = response_json.get("time_range", [])
                has_author_filter = response_json.get("has_author_filter", False)
                authors = response_json.get("authors", [])
                has_institution_filter = response_json.get("has_institution_filter", False)
                institutions = response_json.get("institutions", [])
                
                # 保存完整的意图分析结果，便于后续处理
                # 可以考虑将这些信息存储在某个地方，以便后端检索使用
                print(f"意图分析结果: {response_json}")
                
                return structured_query
                
            except json.JSONDecodeError:
                # 如果无法解析为JSON，则直接返回原始回复
                print("无法解析API返回的JSON，使用原始文本")
                # 提取回复中的检索式部分
                if "```json" in ai_response:
                    # 尝试提取代码块中的内容
                    match = re.search(r'```json\s*(.*?)\s*```', ai_response, re.DOTALL)
                    if match:
                        try:
                            json_str = match.group(1)
                            response_json = json.loads(json_str)
                            return response_json.get("query", ai_response)
                        except:
                            pass
                
                # 如果上述尝试都失败，返回原始回复作为检索式
                return ai_response
                
        except (KeyError, IndexError) as e:
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
