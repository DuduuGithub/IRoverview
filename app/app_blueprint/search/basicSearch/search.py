#!/usr/bin/python
import os
import sys
import logging
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from app.app_blueprint.search.rank.rank import rank_results, SORT_BY_RELEVANCE, SORT_BY_TIME_DESC, SORT_BY_TIME_ASC, SORT_BY_COMBINED
import re
import getopt
import codecs
import struct
import math
import io
import collections
import json
import bisect
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# 配置日志记录
def setup_logger():
    # 创建logs目录
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件名，包含时间戳
    log_file = os.path.join(log_dir, f'search_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# 初始化日志记录器
logger = setup_logger()

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
    # 保留更多特殊字符，只移除真正无用的标点
    text = re.sub(r'[^\w\s\-\.]', ' ', text)
    # 分词
    tokens = word_tokenize(text)
    # 只过滤空字符串
    tokens = [word for word in tokens if len(word) > 0]
    return tokens

# 常量定义
BYTE_SIZE = 4  # 文档ID使用整型存储，占用4字节

"""
布尔检索主函数，支持从倒排索引读取数据
如果指定了排序方式，会对结果进行排序处理
参数:
    - dictionary_file: 词典文件路径
    - postings_file: 倒排索引文件路径
    - queries_file: 查询文件路径
    - output_file: 输出文件路径
    - sort_method: 排序方法，默认按相关性排序
    - query_text: 查询文本（直接查询模式时使用）
    - use_file: 是否使用文件模式，默认为False
返回:
    - 匹配的文档ID列表
"""
def search(
    dictionary_file=None,
    postings_file=None,
    queries_file=None,
    output_file=None,
    sort_method=SORT_BY_RELEVANCE,
    query_text=None,
    use_file=False):  # 默认使用API模式
    
    # 批量文件模式（用于命令行批量处理多个查询）
    if use_file:
        if not all([dictionary_file, postings_file, queries_file, output_file]):
            raise ValueError("批量文件模式需要提供所有文件路径参数：词典文件、倒排文件、查询文件和输出文件")
            
        # 打开文件
        dict_file = codecs.open(dictionary_file, encoding='utf-8')
        post_file = io.open(postings_file, 'rb')
        query_file = codecs.open(queries_file, encoding='utf-8')
        out_file = open(output_file, 'w')

        # 加载词典
        loaded_dict = load_dictionary(dict_file)
        dictionary = loaded_dict[0]     # 词典映射
        indexed_docIDs = loaded_dict[1]  # 所有已索引的文档
        doc_lengths = loaded_dict[2]    # 文档长度映射
        doc_metadata = loaded_dict[3]   # 文档元数据 {docID: {field: value}}
        dict_file.close()

        # 处理每个查询
        queries_list = query_file.read().splitlines()
        results = []
        
        for i, query in enumerate(queries_list):
            # 判断是否为布尔查询
            has_boolean_ops = any(op in query for op in ['AND', 'OR', 'NOT', '(', ')'])

            # 根据查询类型选择处理方法
            if has_boolean_ops:
                result = process_boolean_query(query, dictionary, post_file, indexed_docIDs)
            else:
                result = get_matching_docs(query, dictionary, post_file)

            # 提取查询词项，用于排序
            query_terms = extract_query_terms(query)

            # 对结果进行排序（非布尔查询）
            if result and not has_boolean_ops and query_terms:
                result = rank_results(
                    result, query, dictionary, post_file, doc_metadata, sort_method)

            # 将结果写入输出文件
            out_file.write(' '.join(str(doc_id) for doc_id in result))
            if i != len(queries_list) - 1:
                out_file.write('\n')
                
            results.append(result)

        # 关闭文件
        post_file.close()
        query_file.close()
        out_file.close()
        return results
    
    # API模式（用于代码中直接调用）
    else:
        if query_text is None:
            raise ValueError("API模式必须提供query_text参数")
        if not all([dictionary_file, postings_file]):
            raise ValueError("API模式需要提供词典文件和倒排文件路径")
            
        return process_query_with_index(query_text, dictionary_file, postings_file, sort_method)

"""
使用倒排索引处理查询
参数:
    - query_text: 查询文本
    - dictionary_file: 词典文件路径
    - postings_file: 倒排索引文件路径
    - sort_method: 排序方法
返回:
    - 匹配的文档ID列表
"""
def process_query_with_index(query_text, dictionary_file, postings_file, sort_method=SORT_BY_RELEVANCE):
    """使用倒排索引处理查询"""
    try:
        logger.info(f"开始处理查询: {query_text}")
        logger.info(f"使用词典文件: {dictionary_file}")
        logger.info(f"使用倒排文件: {postings_file}")
        logger.info(f"排序方式: {sort_method}")
        
        # 打开文件
        dict_file = codecs.open(dictionary_file, encoding='utf-8')
        post_file = io.open(postings_file, 'rb')
        
        # 加载词典
        loaded_dict = load_dictionary(dict_file)
        dictionary = loaded_dict[0]     # 词典映射
        indexed_docIDs = loaded_dict[1]  # 所有已索引的文档
        doc_lengths = loaded_dict[2]    # 文档长度映射
        doc_metadata = loaded_dict[3]   # 文档元数据
        dict_file.close()
        
        logger.info(f"成功加载词典，包含 {len(dictionary)} 个词项")
        logger.info(f"索引文档数量: {len(indexed_docIDs)}")
        
        # 标准化布尔操作符，确保空格分隔
        query_text = re.sub(r'\band\b', ' AND ', query_text, flags=re.IGNORECASE)
        query_text = re.sub(r'\bor\b', ' OR ', query_text, flags=re.IGNORECASE)
        query_text = re.sub(r'\bnot\b', ' NOT ', query_text, flags=re.IGNORECASE)
        
        # 判断是否为布尔查询
        has_boolean_ops = any(op in query_text for op in ['AND', 'OR', 'NOT', '(', ')'])
        logger.info(f"查询类型: {'布尔查询' if has_boolean_ops else '普通查询'}")

        # 根据查询类型选择处理方法
        if has_boolean_ops:
            result = process_boolean_query(query_text, dictionary, post_file, indexed_docIDs)
        else:
            result = get_matching_docs(query_text, dictionary, post_file)

        # 提取查询词项，用于排序
        query_terms = extract_query_terms(query_text)
        logger.info(f"查询词项: {query_terms}")

        # 对结果进行排序（非布尔查询）
        if result and not has_boolean_ops and query_terms and sort_method:
            logger.info(f"开始对 {len(result)} 个结果进行排序")
            result = rank_results(result, query_text, dictionary, post_file, doc_metadata, sort_method)
            logger.info("排序完成")

        # 关闭文件
        post_file.close()
        logger.info(f"查询处理完成，返回 {len(result)} 个结果")
        return result
        
    except Exception as e:
        logger.error(f"处理查询时发生错误: {str(e)}", exc_info=True)
        # 确保文件被正确关闭
        try:
            if 'dict_file' in locals():
                dict_file.close()
            if 'post_file' in locals():
                post_file.close()
        except:
            pass
        raise

"""
从倒排索引获取匹配的文档ID列表
参数:
    - query: 查询文本
    - dictionary: 词典
    - post_file: 倒排索引文件对象
返回:
    - 匹配的文档ID列表
"""
def get_matching_docs(query, dictionary, post_file):
    """从倒排索引获取匹配的文档ID列表"""
    # 分词 - 使用英文分词
    terms = tokenize_english(query)
    logger.info(f"查询词项: {terms}")
    
    # 如果没有有效词项，返回空列表
    if not terms:
        return []
    
    # 对每个词项获取匹配文档，然后取并集
    results = []
    for term in terms:
        if term in dictionary:
            df, idf, offset = dictionary[term]
            logger.info(f"词项 '{term}' 的统计信息: df={df}, idf={idf:.4f}, offset={offset}")
            postings_list = load_posting_list(post_file, df, offset)
            logger.info(f"词项 '{term}' 匹配到 {len(postings_list)} 个文档")
            if not results:
                results = postings_list
            else:
                results = boolean_OR(results, postings_list)
                logger.info(f"合并后的结果数量: {len(results)}")
        else:
            logger.warning(f"词项 '{term}' 不在词典中")
    
    logger.info(f"最终匹配到 {len(results)} 个文档")
    return results

"""
从查询中提取词项，用于排序和结果处理
参数:
    - query: 查询文本
返回:
    - 提取的词项列表
"""
def extract_query_terms(query):
    # 移除布尔操作符
    cleaned_query = re.sub(r'\bAND\b|\bOR\b|\bNOT\b|\(|\)', ' ', query, flags=re.IGNORECASE)
    
    # 使用英文分词
    return tokenize_english(cleaned_query)

"""
加载词典文件
参数:
    - dict_file: 词典文件对象
返回:
    - 包含词典、索引文档ID列表、文档长度和元数据的元组
"""
def load_dictionary(dict_file):
    # 读取词典文件
    lines = dict_file.readlines()

    # 第一行包含所有已索引的文档ID
    indexed_docIDs_line = lines[0].strip()
    if indexed_docIDs_line.startswith("all_indexed_docIDs:"):
        indexed_docIDs_str = indexed_docIDs_line[len("all_indexed_docIDs:"):].strip()
        indexed_docIDs = list(map(int, indexed_docIDs_str.split(',')))
    else:
        indexed_docIDs = []

    # 第二行包含文档长度信息
    doc_lengths = {}
    doc_lengths_line = lines[1].strip()
    if doc_lengths_line.startswith("doc_lengths:"):
        doc_lengths_str = doc_lengths_line[len("doc_lengths:"):].strip()
        doc_lengths = json.loads(doc_lengths_str)
        # 将字符串键转换为整数键
        doc_lengths = {int(k): v for k, v in doc_lengths.items()}
    
    # 第三行开始包含文档元数据（可选）
    doc_metadata = {}
    metadata_line_index = 2
    if metadata_line_index < len(lines) and lines[metadata_line_index].strip().startswith("doc_metadata:"):
        metadata_str = lines[metadata_line_index][len("doc_metadata:"):].strip()
        try:
            doc_metadata = json.loads(metadata_str)
            # 确保键是整数
            doc_metadata = {int(k): v for k, v in doc_metadata.items()}
        except json.JSONDecodeError:
            doc_metadata = {}
        metadata_line_index += 1
    
    # 解析词典条目
    dictionary = {}
    for i in range(metadata_line_index, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
            
        parts = line.split()
        if len(parts) >= 4:
            term = parts[0]
            df = int(parts[1])     # 文档频率
            idf = float(parts[2])  # 逆文档频率
            offset = int(parts[3]) # 在倒排文件中的偏移量
            dictionary[term] = (df, idf, offset)
    
    return (dictionary, indexed_docIDs, doc_lengths, doc_metadata)

"""
处理布尔查询表达式
参数:
    - query: 布尔查询表达式
    - dictionary: 词典
    - post_file: 倒排索引文件对象
    - indexed_docIDs: 所有已索引的文档ID列表
返回:
    - 匹配的文档ID列表
"""
def process_boolean_query(query, dictionary, post_file, indexed_docIDs):
    """处理布尔查询表达式"""
    logger.info(f"处理布尔查询: {query}")
    
    # 处理查询字符串，确保括号周围有空格
    query = query.replace('(', ' ( ').replace(')', ' ) ')
    query = re.sub(r'\s+', ' ', query).strip()  # 规范化空格
    logger.debug(f"规范化后的查询: {query}")
    
    # 标准化布尔操作符
    query = re.sub(r'\bAND\b', 'AND', query, flags=re.IGNORECASE)
    query = re.sub(r'\bOR\b', 'OR', query, flags=re.IGNORECASE)
    query = re.sub(r'\bNOT\b', 'NOT', query, flags=re.IGNORECASE)
    
    # 分词处理
    tokens = []
    for part in query.split():
        if part in ['AND', 'OR', 'NOT', '(', ')']:
            tokens.append(part)
        else:
            # 对非操作符部分进行分词 - 使用英文分词
            tokens.extend(tokenize_english(part))
    
    # 使用Shunting-yard算法转换为后缀表达式
    postfix = shunting_yard(tokens)
    
    # 计算后缀表达式
    result = evaluate_postfix(postfix, dictionary, post_file, indexed_docIDs)
    logger.info(f"布尔查询结果: {len(result)} 个文档")
    return result


"""
评估后缀表达式
参数:
    - postfix: 后缀表达式
    - dictionary: 词典
    - post_file: 倒排索引文件对象
    - indexed_docIDs: 所有已索引的文档ID列表
返回:
    - 匹配的文档ID列表
"""
def evaluate_postfix(postfix, dictionary, post_file, indexed_docIDs):
    stack = []
    
    for token in postfix:
        if token == 'AND':
            if len(stack) < 2:
                raise ValueError("表达式错误：AND操作符缺少操作数")
            right = stack.pop()
            left = stack.pop()
            stack.append(boolean_AND(left, right))
        elif token == 'OR':
            if len(stack) < 2:
                raise ValueError("表达式错误：OR操作符缺少操作数")
            right = stack.pop()
            left = stack.pop()
            stack.append(boolean_OR(left, right))
        elif token == 'NOT':
            if len(stack) < 1:
                raise ValueError("表达式错误：NOT操作符缺少操作数")
            operand = stack.pop()
            stack.append(boolean_NOT(operand, indexed_docIDs))
        else:
            # 词项，获取匹配的文档ID
            term_docs = get_matching_docs(token, dictionary, post_file)
            stack.append(term_docs)
    
    if len(stack) != 1:
        raise ValueError("表达式错误：操作符与操作数不匹配")
    
    return stack[0]

# 以下是布尔操作函数，不需要修改
"""
计算两个列表的逻辑与（交集）
参数:
    - list1, list2: 两个文档ID列表
返回:
    - 交集结果
"""
def boolean_AND(list1, list2):
    """计算两个列表的逻辑与（交集）"""
    try:
        # 优化：根据列表长度决定使用哪种方法
        if not list1 or not list2:
            logger.debug("至少一个列表为空，返回空列表")
            return []
            
        logger.debug(f"计算交集: 列表1长度={len(list1)}, 列表2长度={len(list2)}")
        
        if len(list1) > 10*len(list2) or len(list2) > 10*len(list1):
            # 长度差异大时使用二分查找
            logger.debug("使用二分查找方法")
            if len(list1) < len(list2):
                result = binary_search_AND(list2, list1)
            else:
                result = binary_search_AND(list1, list2)
        elif len(list1) + len(list2) > 1000:
            # 列表较长时使用跳跃表方法
            logger.debug("使用跳跃表方法")
            result = skip_list_AND(list1, list2)
        else:
            # 列表较短时使用简单扫描
            logger.debug("使用简单扫描方法")
            result = [x for x in list1 if x in list2]
            
        logger.debug(f"交集计算完成，结果长度: {len(result)}")
        return result
        
    except Exception as e:
        logger.error(f"计算交集时发生错误: {str(e)}", exc_info=True)
        return []


"""
使用二分查找实现列表交集
参数:
    - longer_list, shorter_list: 两个文档ID列表，longer_list应比shorter_list长
返回:
    - 交集结果
"""
def binary_search_AND(longer_list, shorter_list):
    result = []
    # 对较短的列表中的每个元素，在较长的列表中二分查找
    for item in shorter_list:
        if binary_search(longer_list, item):
            result.append(item)
    return result


"""
二分查找函数
参数:
    - lst: 有序列表
    - item: 要查找的元素
返回:
    - 布尔值，表示元素是否存在
"""
def binary_search(lst, item):
    left, right = 0, len(lst) - 1
    while left <= right:
        mid = (left + right) // 2
        if lst[mid] == item:
            return True
        elif lst[mid] < item:
            left = mid + 1
        else:
            right = mid - 1
    return False


"""
使用跳跃列表优化的交集算法
参数:
    - list1, list2: 两个文档ID列表
返回:
    - 交集结果
"""
def skip_list_AND(list1, list2):
    if not list1 or not list2:
        return []
    
    result = []
    i, j = 0, 0
    len1, len2 = len(list1), len(list2)
    skip1 = int(math.sqrt(len1))  # 跳跃步长
    skip2 = int(math.sqrt(len2))
    
    while i < len1 and j < len2:
        if list1[i] == list2[j]:
            result.append(list1[i])
            i += 1
            j += 1
        elif list1[i] < list2[j]:
            # 尝试跳跃
            if i + skip1 < len1 and list1[i + skip1] <= list2[j]:
                while i + skip1 < len1 and list1[i + skip1] <= list2[j]:
                    i += skip1
            else:
                i += 1
        else:  # list1[i] > list2[j]
            # 尝试跳跃
            if j + skip2 < len2 and list2[j + skip2] <= list1[i]:
                while j + skip2 < len2 and list2[j + skip2] <= list1[i]:
                    j += skip2
            else:
                j += 1
                
    return result


"""
计算两个列表的逻辑或（并集）
参数:
    - list1, list2: 两个文档ID列表
返回:
    - 并集结果
"""
def boolean_OR(list1, list2):
    # 使用集合去重并合并
    return sorted(list(set(list1) | set(list2)))


"""
计算逻辑非（补集）
参数:
    - list1: 要取补集的文档ID列表
    - all_docIDs: 全部文档ID列表
返回:
    - list1相对于all_docIDs的补集
"""
def boolean_NOT(list1, all_docIDs):
    # 特殊情况处理：如果list1为空或包含特殊标记，则返回所有文档ID
    if not list1:
        return all_docIDs
    
    # 使用集合求差集
    return sorted(list(set(all_docIDs) - set(list1)))


"""
Shunting-yard算法，将中缀表达式转换为后缀表达式
参数:
    - query_tokens: 查询标记列表
返回:
    - 后缀表达式标记列表
"""
def shunting_yard(query_tokens):
    output_queue = []
    operator_stack = []
    operators = {'AND': 2, 'OR': 1, 'NOT': 3}  # 运算符优先级
    
    # 预处理，修正NOT操作符
    processed_tokens = []
    i = 0
    while i < len(query_tokens):
        # 如果当前是NOT操作符且前面有词项且不是操作符
        if query_tokens[i] == 'NOT' and i > 0 and query_tokens[i-1] not in operators and query_tokens[i-1] != '(':
            # 在NOT前插入AND操作符
            processed_tokens.append('AND')
            processed_tokens.append(query_tokens[i])
        # 如果当前是词项且前面是NOT后面的词项
        elif i > 1 and query_tokens[i-1] == 'NOT' and query_tokens[i-2] not in operators and query_tokens[i-2] != '(':
            processed_tokens.append(query_tokens[i])
            # 处理完NOT A之后添加AND操作符
            if i+1 < len(query_tokens) and query_tokens[i+1] not in operators and query_tokens[i+1] != ')':
                processed_tokens.append('AND')
        else:
            processed_tokens.append(query_tokens[i])
        i += 1
    
    print(f"预处理后的查询词项: {processed_tokens}")
    
    # 正常的Shunting-yard算法处理
    for token in processed_tokens:
        if token in operators:
            # 处理操作符
            while (operator_stack and operator_stack[-1] != '(' and 
                   ((token != 'NOT' and operators.get(operator_stack[-1], 0) >= operators.get(token, 0)) or
                    (token == 'NOT' and operators.get(operator_stack[-1], 0) > operators.get(token, 0)))):
                output_queue.append(operator_stack.pop())
            operator_stack.append(token)
        elif token == '(':
            operator_stack.append(token)
        elif token == ')':
            # 处理右括号
            while operator_stack and operator_stack[-1] != '(':
                output_queue.append(operator_stack.pop())
            if operator_stack and operator_stack[-1] == '(':
                operator_stack.pop()  # 弹出左括号
            else:
                raise ValueError("表达式错误：括号不匹配")
        else:
            # 处理操作数
            output_queue.append(token)
    
    # 将剩余的操作符加入输出队列
    while operator_stack:
        if operator_stack[-1] == '(':
            raise ValueError("表达式错误：括号不匹配")
        output_queue.append(operator_stack.pop())
    
    return output_queue


"""
从倒排文件中加载倒排列表
参数:
    - post_file: 倒排索引文件对象
    - length: 列表长度
    - offset: 列表在文件中的偏移量
返回:
    - 倒排列表（文档ID列表）
"""
def load_posting_list(post_file, length, offset):
    """从倒排文件中读取倒排列表"""
    try:
        post_file.seek(offset)
        # 读取倒排列表长度（4字节）
        length_bytes = post_file.read(4)
        if not length_bytes:
            logger.warning(f"在偏移量 {offset} 处无法读取倒排列表长度")
            return []
            
        posting_length = int.from_bytes(length_bytes, byteorder='big')
        logger.debug(f"读取到倒排列表长度: {posting_length}")
        
        if posting_length > 1000000:  # 设置一个合理的上限
            logger.error(f"倒排列表长度异常: {posting_length}")
            return []
        
        # 使用生成器读取倒排列表，减少内存使用
        def read_doc_ids():
            for _ in range(posting_length):
                # 读取文档ID（8字节）
                doc_id_bytes = post_file.read(8)
                if not doc_id_bytes:
                    logger.error(f"读取文档ID时出错，offset={post_file.tell()}")
                    break
                doc_id = int.from_bytes(doc_id_bytes, byteorder='big')
                yield doc_id
        
        # 将生成器转换为列表
        posting_list = list(read_doc_ids())
        
        if len(posting_list) != posting_length:
            logger.warning(f"实际读取的文档ID数量 ({len(posting_list)}) 与预期 ({posting_length}) 不符")
        
        logger.debug(f"成功读取倒排列表，包含 {len(posting_list)} 个文档ID")
        return posting_list
        
    except Exception as e:
        logger.error(f"读取倒排列表时发生错误: {str(e)}", exc_info=True)
        return []


"""
从倒排文件中加载倒排列表及词频
参数:
    - post_file: 倒排索引文件对象
    - length: 列表长度
    - offset: 列表在文件中的偏移量
返回:
    - 倒排列表，每项为(docID, 词频)元组
"""
def load_posting_list_with_tf(post_file, length, offset):
    """从倒排文件中读取带词频的倒排列表"""
    post_file.seek(offset)
    # 读取倒排列表长度（4字节）
    length_bytes = post_file.read(4)
    if not length_bytes:
        return []
    posting_length = int.from_bytes(length_bytes, byteorder='big')
    
    # 读取倒排列表
    posting_list = []
    for _ in range(posting_length):
        # 读取文档ID（8字节）
        doc_id_bytes = post_file.read(8)
        if not doc_id_bytes:
            break
        doc_id = int.from_bytes(doc_id_bytes, byteorder='big')
        
        # 读取词频（4字节）
        tf_bytes = post_file.read(4)
        if not tf_bytes:
            break
        tf = int.from_bytes(tf_bytes, byteorder='big')
        
        posting_list.append((doc_id, tf))
    
    return posting_list


"""显示正确的命令用法"""
def print_usage():
    print("用法: " + sys.argv[0] + " -d 词典文件 -p 倒排文件 -q 查询文件 -o 输出文件")

# 命令行接口
if __name__ == "__main__":
    dictionary_file = postings_file = queries_file = output_file = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)
        
    for o, a in opts:
        if o == '-d':
            dictionary_file = a
        elif o == '-p':
            postings_file = a
        elif o == '-q':
            queries_file = a
        elif o == '-o':
            output_file = a
        else:
            assert False, "未处理的选项"
            
    if dictionary_file == None or postings_file == None or queries_file == None or output_file == None:
        print_usage()
        sys.exit(2)
        
    search(dictionary_file, postings_file, queries_file, output_file, use_file=True)  # 命令行使用批量文件模式