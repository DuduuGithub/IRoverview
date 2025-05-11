#!/usr/bin/python
from ..rank.rank import rank_results, SORT_BY_RELEVANCE, SORT_BY_TIME_DESC, SORT_BY_TIME_ASC, SORT_BY_COMBINED
from Database.config import db
from Database.model import Work, Author, Topic, Concept, Institution, Source, WorkAuthorship, WorkConcept, WorkTopic
import re
import sys
import getopt
import codecs
import struct
import math
import io
import collections
import json
import bisect
from datetime import datetime
import jieba
import subprocess
import jieba

# 导入数据库模型和配置
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# 导入排序模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 常量定义
BYTE_SIZE = 4  # 文档ID使用整型存储，占用4字节

"""
布尔检索主函数，支持从数据库读取数据
如果指定了排序方式，会对结果进行排序处理
"""


def search(
    dictionary_file=None,
    postings_file=None,
    queries_file=None,
    output_file=None,
    sort_method=SORT_BY_RELEVANCE,
    query_text=None,
     use_db=True):
    # 如果使用数据库模式，则忽略文件参数
    if use_db:
        if query_text is None:
            raise ValueError("使用数据库模式时必须提供query_text参数")

        # 处理查询
        return process_query_with_db(query_text, sort_method)
    else:
        # 传统文件模式
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
        for i in range(len(queries_list)):
            query = queries_list[i]

            # 判断是否为布尔查询
            has_boolean_ops = any(op in query for op in [
                                  'AND', 'OR', 'NOT', '(', ')'])

            if has_boolean_ops:
                result = process_boolean_query(
                    query, dictionary, post_file, indexed_docIDs)
            else:
                matched_docs = get_matching_docs(query, dictionary, post_file)
                result = matched_docs

            # 提取查询词项，用于排序
            query_terms = extract_query_terms(query)

            # 对结果进行排序
            if result and not has_boolean_ops and query_terms:
                # 如果结果存在且不是布尔查询，应用排序
                result = rank_results(
                    result, query, dictionary, post_file, doc_metadata, sort_method)

            # 将结果写入输出文件
            for j in range(len(result)):
                docID = str(result[j])
                if (j != len(result) - 1):
                    docID += ' '
                out_file.write(docID)
            if (i != len(queries_list) - 1):
                out_file.write('\n')
            results.append(result)

        # 关闭文件
        post_file.close()
        query_file.close()
        out_file.close()
        return results


"""从数据库处理查询"""


def process_query_with_db(query_text, sort_method=SORT_BY_RELEVANCE):
    # 处理查询前，将查询文本中的布尔操作符转换为大写形式(不区分大小写)
    # 首先替换小写形式的操作符，确保空格分隔
    query_text = re.sub(r'\band\b', ' AND ', query_text, flags=re.IGNORECASE)
    query_text = re.sub(r'\bor\b', ' OR ', query_text, flags=re.IGNORECASE)
    query_text = re.sub(r'\bnot\b', ' NOT ', query_text, flags=re.IGNORECASE)
    
    # 判断是否为布尔查询 - 使用大写形式检查
    has_boolean_ops = any(op in query_text for op in ['AND', 'OR', 'NOT', '(', ')'])

    print(f"查询文本: {query_text}")
    print(f"是否为布尔查询: {has_boolean_ops}")

    if has_boolean_ops:
        # 布尔查询处理
        result = process_boolean_query_with_db(query_text)
    else:
        # 普通查询处理
        result = get_matching_docs_from_db(query_text)

    # 提取查询词项，用于排序
    query_terms = extract_query_terms(query_text)

    # 对结果进行排序
    if result and not has_boolean_ops and query_terms and sort_method:
        # 如果结果存在且不是布尔查询，应用排序
        result = sort_db_results(result, query_text, sort_method)

    return result


"""从数据库获取匹配的文档ID列表"""


def get_matching_docs_from_db(query):
    # 分词
    print(f"正在搜索关键词: {query}")
    query_terms = [term.lower().strip() for term in jieba.cut(query)]
    query_terms = [re.sub(r'[^\w\s]', '', term)
                          for term in query_terms if term.strip()]
    
    print(f"分词结果: {query_terms}")
    
    # 如果分词结果为空，直接使用原始查询词
    if not query_terms:
        query_terms = [query.lower().strip()]
    
    results = []

    # 对每个词项进行查询
    for term in query_terms:
        if not term:
            continue
            
        print(f"搜索词项: {term}")

        # 在多个字段中搜索 - 使用更宽松的匹配条件
        matching_works = Work.query.filter(
            (Work.title.ilike(f'%{term}%')) |
            (Work.display_name.ilike(f'%{term}%')) |
            (Work.abstract_inverted_index.cast(db.String).ilike(f'%{term}%'))  # 在摘要中也搜索
        ).all()
        
        print(f"标题/显示名称/摘要匹配数: {len(matching_works)}")

        # 添加作者匹配
        author_matches = db.session.query(Work).join(
            WorkAuthorship, Work.id == WorkAuthorship.work_id
        ).join(
            Author, Author.id == WorkAuthorship.author_id
        ).filter(
            Author.display_name.ilike(f'%{term}%')
        ).all()
        
        print(f"作者匹配数: {len(author_matches)}")

        # 添加概念/主题匹配
        concept_matches = db.session.query(Work).join(
            WorkConcept, Work.id == WorkConcept.work_id
        ).join(
            Concept, Concept.id == WorkConcept.concept_id
        ).filter(
            Concept.display_name.ilike(f'%{term}%')
        ).all()
        
        print(f"概念匹配数: {len(concept_matches)}")

        topic_matches = db.session.query(Work).join(
            WorkTopic, Work.id == WorkTopic.work_id
        ).join(
            Topic, Topic.id == WorkTopic.topic_id
        ).filter(
            Topic.display_name.ilike(f'%{term}%')
        ).all()
        
        print(f"主题匹配数: {len(topic_matches)}")

        # 合并结果并去重
        term_results = list(set(
            [w.id for w in matching_works + author_matches + concept_matches + topic_matches]))
            
        print(f"词项'{term}'的匹配结果数: {len(term_results)}")

        # 与之前的结果进行OR操作
        if not results:
            results = term_results
        else:
            results = list(set(results + term_results))
    
    print(f"总匹配结果数: {len(results)}")
    return results


"""处理数据库布尔查询"""


def process_boolean_query_with_db(query):
    # 处理查询字符串，确保括号周围有空格
    query = query.replace('(', ' ( ').replace(')', ' ) ')
    
    # 先处理操作符，确保大写并且前后有空格
    # 这样可以防止操作符被jieba分词时被拆分
    query = re.sub(r'\bAND\b', ' AND ', query, flags=re.IGNORECASE)
    query = re.sub(r'\bOR\b', ' OR ', query, flags=re.IGNORECASE)
    query = re.sub(r'\bNOT\b', ' NOT ', query, flags=re.IGNORECASE)

    print(f"处理的布尔查询: {query}")

    # 分离操作符和词项
    operators = ['AND', 'OR', 'NOT', '(', ')']
    query_parts = []

    for part in query.split():
        if part in operators:
            query_parts.append(part)
        else:
            # 对非操作符部分进行分词
            query_parts.extend(list(jieba.cut(part)))

    # 过滤空字符串
    query_tokens = [token for token in query_parts if token.strip()]
    print(f"分词后的查询标记: {query_tokens}")

    # 将查询转换为后缀表达式
    results_stack = []
    postfix_queue = collections.deque(shunting_yard(query_tokens))
    print(f"后缀表达式: {list(postfix_queue)}")

    if not postfix_queue:
        return []

    # 获取所有文档ID（用于NOT操作）
    all_doc_ids = [w.id for w in Work.query.all()]

    # 执行后缀表达式计算
    while postfix_queue:
        token = postfix_queue.popleft()
        result = []

        # 处理操作数
        if token not in ('AND', 'OR', 'NOT'):
            term = token.lower().strip()
            term = re.sub(r'[^\w\s]', '', term)

            if term:
                # 在多个字段中搜索
                matching_works = Work.query.filter(
                    (Work.title.ilike(f'%{term}%')) |
                    (Work.display_name.ilike(f'%{term}%'))
                ).all()

                # 添加作者匹配
                author_matches = db.session.query(Work).join(
                    WorkAuthorship, Work.id == WorkAuthorship.work_id
                ).join(
                    Author, Author.id == WorkAuthorship.author_id
                ).filter(
                    Author.display_name.ilike(f'%{term}%')
                ).all()

                # 添加概念/主题匹配
                concept_matches = db.session.query(Work).join(
                    WorkConcept, Work.id == WorkConcept.work_id
                ).join(
                    Concept, Concept.id == WorkConcept.concept_id
                ).filter(
                    Concept.display_name.ilike(f'%{term}%')
                ).all()

                topic_matches = db.session.query(Work).join(
                    WorkTopic, Work.id == WorkTopic.work_id
                ).join(
                    Topic, Topic.id == WorkTopic.topic_id
                ).filter(
                    Topic.display_name.ilike(f'%{term}%')
                ).all()

                # 合并结果并去重
                result = list(set([w.id for w in matching_works +
                              author_matches + concept_matches + topic_matches]))
                
                print(f"关键词'{term}'匹配结果数: {len(result)}")

        # 处理操作符
        elif token == 'AND':
            right_operand = results_stack.pop() if results_stack else []
            left_operand = results_stack.pop() if results_stack else []
            result = boolean_AND(left_operand, right_operand)
            print(f"执行AND操作: {len(left_operand)} AND {len(right_operand)} = {len(result)}")

        elif token == 'OR':
            right_operand = results_stack.pop() if results_stack else []
            left_operand = results_stack.pop() if results_stack else []
            result = boolean_OR(left_operand, right_operand)
            print(f"执行OR操作: {len(left_operand)} OR {len(right_operand)} = {len(result)}")

        elif token == 'NOT':
            right_operand = results_stack.pop() if results_stack else []
            result = boolean_NOT(right_operand, all_doc_ids)
            print(f"执行NOT操作: NOT {len(right_operand)} = {len(result)}")

        results_stack.append(result)

    return results_stack.pop() if results_stack else []


"""对数据库查询结果进行排序"""


def sort_db_results(result_ids, query_text, sort_method):
    if not result_ids:
        return []

    if sort_method == SORT_BY_TIME_DESC:
        # 按发表年份降序排序
        works = Work.query.filter(Work.id.in_(result_ids)).order_by(
            Work.publication_year.desc()).all()
        return [w.id for w in works]

    elif sort_method == SORT_BY_TIME_ASC:
        # 按发表年份升序排序
        works = Work.query.filter(Work.id.in_(result_ids)).order_by(
            Work.publication_year.asc()).all()
        return [w.id for w in works]

    elif sort_method == SORT_BY_COMBINED:
        # 组合排序（相关性 + 时间）
        # 这里简化处理，先按相关性再按时间排序
        query_terms = extract_query_terms(query_text)
        scored_results = []

        for work_id in result_ids:
            work = Work.query.get(work_id)
            if work:
                # 计算简单的TF得分
                score = 0
                for term in query_terms:
                    if term in work.title.lower():
                        score += 2  # 标题匹配权重更高
                    if work.abstract_inverted_index and term in json.dumps(
                        work.abstract_inverted_index).lower():
                        score += 1  # 摘要匹配

                # 时间因素（最近5年的文章得分较高）
                current_year = datetime.now().year
                if work.publication_year and current_year - work.publication_year <= 5:
                    time_bonus = 1 - \
                        (current_year - work.publication_year) * 0.2
                    score = score * 0.7 + time_bonus * 0.3

                scored_results.append((work_id, score))

        # 按得分降序排序
        sorted_results = [r[0] for r in sorted(
            scored_results, key=lambda x: x[1], reverse=True)]
        return sorted_results

    else:
        # 默认按相关性排序
        query_terms = extract_query_terms(query_text)
        scored_results = []

        for work_id in result_ids:
            work = Work.query.get(work_id)
            if work:
                # 计算简单的TF得分
                score = 0
                for term in query_terms:
                    if term in work.title.lower():
                        score += 2  # 标题匹配权重更高
                    if work.abstract_inverted_index and term in json.dumps(
                        work.abstract_inverted_index).lower():
                        score += 1  # 摘要匹配

                scored_results.append((work_id, score))

        # 按得分降序排序
        sorted_results = [r[0] for r in sorted(
            scored_results, key=lambda x: x[1], reverse=True)]
        return sorted_results


"""从布尔查询中提取查询词项，用于排序"""


def extract_query_terms(query):
    # 移除布尔操作符
    query = re.sub(r'\bAND\b|\bOR\b|\bNOT\b|\(|\)', ' ', query)
    # 分词
    terms = list(jieba.cut(query.strip()))
    # 过滤空词项
    terms = [term.lower().strip() for term in terms if term.strip()]
    return terms


"""解析词典文件并返回词典数据结构"""


def load_dictionary(dict_file):
    dictionary = {}          # 词典映射 {term: (df, idf, offset)}
    indexed_docIDs = []      # 所有已索引的文档ID列表
    doc_lengths = {}         # 文档长度映射 {docID: 长度}
    doc_metadata = {}        # 文档元数据 {docID: {field: value}}

    docIDs_processed = False
    doc_lengths_processed = False
    doc_metadata_processed = False

    # 解析词典文件
    for entry in dict_file.read().split('\n'):
        if not entry:
            continue

        # 处理文档ID列表
        if not docIDs_processed:
            if entry.startswith("all_indexed_docIDs: "):
                docIDs_str = entry[20:]
                indexed_docIDs = [
                    int(docID) for docID in docIDs_str.split(',') if docID.strip()]
                docIDs_processed = True
                continue

        # 处理文档长度信息
        if not doc_lengths_processed:
            if entry.startswith("doc_lengths: "):
                doc_lengths = json.loads(entry[13:])
                doc_lengths = {int(k): v for k, v in doc_lengths.items()}
                doc_lengths_processed = True
                continue

        # 处理文档元数据
        if not doc_metadata_processed:
            if entry.startswith("doc_metadata: "):
                doc_metadata = json.loads(entry[14:])
                doc_metadata = {int(k): v for k, v in doc_metadata.items()}
                doc_metadata_processed = True
                continue

        # 处理普通词条
        token = entry.split()
        if len(token) >= 4:  # term df idf offset
            term = token[0]
            df = int(token[1])
            idf = float(token[2])
            offset = int(token[3])
            dictionary[term] = (df, idf, offset)
    
    return (dictionary, indexed_docIDs, doc_lengths, doc_metadata)

"""处理布尔查询表达式"""
def process_boolean_query(query, dictionary, post_file, indexed_docIDs):
    # 处理查询字符串
    query = query.replace('(', ' ( ').replace(')', ' ) ')
    
    # 分离操作符和词项
    operators = ['AND', 'OR', 'NOT', '(', ')']
    query_parts = []
    
    for part in query.split():
        if part in operators:
            query_parts.append(part)
        else:
            query_parts.extend(list(jieba.cut(part)))
    
    # 过滤空字符串
    query_tokens = [token for token in query_parts if token.strip()]
    
    # 将查询转换为后缀表达式
    results_stack = []
    postfix_queue = collections.deque(shunting_yard(query_tokens))
    
    if not postfix_queue:
        return []

    # 执行后缀表达式计算
    while postfix_queue:
        token = postfix_queue.popleft()
        result = []
        
        # 处理操作数
        if token not in ('AND', 'OR', 'NOT'):
            term = token.lower().strip()
            term = re.sub(r'[^\w\s]', '', term)
            
            if term in dictionary: 
                result = load_posting_list(post_file, dictionary[term][0], dictionary[term][2])
        
        # 处理操作符
        elif token == 'AND':
            right_operand = results_stack.pop() if results_stack else []
            left_operand = results_stack.pop() if results_stack else []
            result = boolean_AND(left_operand, right_operand)
            
        elif token == 'OR':
            right_operand = results_stack.pop() if results_stack else []
            left_operand = results_stack.pop() if results_stack else []
            result = boolean_OR(left_operand, right_operand)
            
        elif token == 'NOT':
            right_operand = results_stack.pop() if results_stack else []
            result = boolean_NOT(right_operand, indexed_docIDs)
            
        results_stack.append(result)

    return results_stack.pop() if results_stack else []

"""获取包含查询词项的所有文档"""
def get_matching_docs(query, dictionary, post_file):
    # 分词
    query_terms = [term.lower().strip() for term in jieba.cut(query)]
    query_terms = [re.sub(r'[^\w\s]', '', term) for term in query_terms if term.strip()]
    
    all_matched_docs = []
    
    # 获取所有匹配文档
    for term in query_terms:
        if term in dictionary:
            df, idf, offset = dictionary[term]
            docs = load_posting_list(post_file, df, offset)
            all_matched_docs = boolean_OR(all_matched_docs, docs)
    
    return all_matched_docs

"""对两个列表执行布尔AND操作"""
def boolean_AND(list1, list2):
    if not list1 or not list2:
        return []
    
    # 如果列表长度相差较大，使用二分查找优化
    if len(list1) > 10 * len(list2):
        return binary_search_AND(list1, list2)
    elif len(list2) > 10 * len(list1):
        return binary_search_AND(list2, list1)
    else:
        # 普通合并算法（使用跳表结构）
        return skip_list_AND(list1, list2)

"""使用二分查找优化的AND操作"""
def binary_search_AND(longer_list, shorter_list):
    result = []
    
    # 确保列表已排序
    longer_list = sorted(longer_list)
    shorter_list = sorted(shorter_list)
    
    # 对较短的列表的每个元素，在较长的列表中二分查找
    for item in shorter_list:
        # 使用Python的bisect模块进行二分查找
        index = bisect.bisect_left(longer_list, item)
        if index < len(longer_list) and longer_list[index] == item:
            result.append(item)
    
    return result

"""使用跳表结构优化的AND操作"""
def skip_list_AND(list1, list2):
    result = []
    
    # 确保列表已排序
    list1 = sorted(list1)
    list2 = sorted(list2)
    
    # 计算跳表步长
    skip1 = int(math.sqrt(len(list1)))
    skip2 = int(math.sqrt(len(list2)))
    
    i = j = 0
    
    while i < len(list1) and j < len(list2):
        if list1[i] == list2[j]:
            result.append(list1[i])
            i += 1
            j += 1
        elif list1[i] < list2[j]:
            # 尝试跳跃
            if skip1 > 1 and i + skip1 < len(list1) and list1[i + skip1] <= list2[j]:
                i = next((i + k for k in range(skip1, 1, -1) if i + k < len(list1) and list1[i + k] <= list2[j]), i + 1)
            else:
                i += 1
        else: # list1[i] > list2[j]
            # 尝试跳跃
            if skip2 > 1 and j + skip2 < len(list2) and list2[j + skip2] <= list1[i]:
                j = next((j + k for k in range(skip2, 1, -1) if j + k < len(list2) and list2[j + k] <= list1[i]), j + 1)
            else:
                j += 1
    
    return result

"""对两个列表执行布尔OR操作"""
def boolean_OR(list1, list2):
    # 使用集合去重
    return sorted(list(set(list1) | set(list2)))

"""对两个列表执行布尔NOT操作"""
def boolean_NOT(list1, all_docIDs):
    # 特殊情况处理：如果list1为空或包含特殊标记，则返回所有文档ID
    if not list1 or (len(list1) == 1 and isinstance(list1[0], str) and list1[0] == "all_docs"):
        return all_docIDs
    
    # 正常情况：使用集合差集找出不在list1中的所有文档ID
    return sorted(list(set(all_docIDs) - set(list1)))

"""将中缀表达式转换为后缀表达式"""
def shunting_yard(query_tokens):
    # 使用Shunting-yard算法将中缀表达式转换为后缀表达式
    precedence = {'NOT': 3, 'AND': 2, 'OR': 1}
    output_queue = []
    operator_stack = []
    
    # 特殊处理NOT操作符在开头的情况
    # 如果NOT是第一个标记，或者前面是左括号或其他操作符，需要特殊处理
    if query_tokens and len(query_tokens) > 0:
        for i in range(len(query_tokens)):
            if query_tokens[i] == 'NOT' and (i == 0 or query_tokens[i-1] in ('AND', 'OR', 'NOT', '(')):
                # 确保NOT后面有操作数
                if i + 1 < len(query_tokens) and query_tokens[i+1] not in ('AND', 'OR', ')', '('):
                    continue  # 正常情况，NOT后面跟着操作数
                else:
                    # 在NOT后面插入一个虚拟的"all"操作数，代表所有文档
                    # 这样可以使NOT操作符正确应用于整个文档集
                    new_tokens = query_tokens.copy()
                    new_tokens.insert(i+1, "all_docs")
                    query_tokens = new_tokens
    
    for token in query_tokens:
        if token in ('AND', 'OR', 'NOT'):
            # 处理操作符
            while (operator_stack and operator_stack[-1] != '(' and
                   ((token != 'NOT' and precedence.get(operator_stack[-1], 0) >= precedence.get(token, 0)) or
                    (token == 'NOT' and precedence.get(operator_stack[-1], 0) > precedence.get(token, 0)))):
                output_queue.append(operator_stack.pop())
            operator_stack.append(token)
        elif token == '(':
            operator_stack.append(token)
        elif token == ')':
            # 弹出所有操作符直到左括号
            while operator_stack and operator_stack[-1] != '(':
                output_queue.append(operator_stack.pop())
            if operator_stack and operator_stack[-1] == '(':
                operator_stack.pop()  # 弹出左括号
        else:
            # 操作数直接输出
            output_queue.append(token)
    
    # 将栈中剩余的操作符加入输出队列
    while operator_stack:
        output_queue.append(operator_stack.pop())
    
    return output_queue

"""获取一个词项的倒排列表（仅文档ID）- 使用跳表结构"""
def load_posting_list(post_file, length, offset):
    post_file.seek(offset)
    posting_list = []
    
    # 计算跳表的步长
    skip_distance = int(math.sqrt(length))
    skip_enabled = skip_distance > 1
    
    for i in range(length):
        # 读取文档ID并跳过词频
        posting = post_file.read(BYTE_SIZE)
        docID = struct.unpack('I', posting)[0]
        
        # 存储文档ID
        posting_list.append(docID)
        
        # 如果启用跳表，并且是跳表节点，则存储跳表指针
        if skip_enabled and i % skip_distance == 0:
            # 在实际实现中，这里应该存储跳表指针
            # 由于这是结构适配，我们实际上没有修改文件格式，
            # 只是在内存中模拟跳表结构
            pass
        
        post_file.read(BYTE_SIZE)  # 跳过词频数据
    
    return posting_list

"""获取一个词项的倒排列表（包含词频）"""
def load_posting_list_with_tf(post_file, length, offset):
    post_file.seek(offset)
    posting_list = []
    for i in range(length):
        # 读取文档ID和词频
        posting = post_file.read(BYTE_SIZE)
        docID = struct.unpack('I', posting)[0]
        
        tf_data = post_file.read(BYTE_SIZE)
        tf = struct.unpack('I', tf_data)[0]
        
        posting_list.append((docID, tf))
    return posting_list

"""显示正确的命令用法"""
def print_usage():
    print("用法: " + sys.argv[0] + " -d 词典文件 -p 倒排文件 -q 查询文件 -o 结果输出文件 [-s 排序方式]")

# 命令行接口
if __name__ == "__main__":
    dictionary_file = postings_file = queries_file = output_file = None
    sort_method = SORT_BY_RELEVANCE
    use_db = False
    query_text = None
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:s:u:t:')
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
        elif o == '-s':
            # 可选的排序方式
            if a in [SORT_BY_RELEVANCE, SORT_BY_TIME_DESC, SORT_BY_TIME_ASC, SORT_BY_COMBINED]:
                sort_method = a
        elif o == '-u':
            # 使用数据库模式
            use_db = (a.lower() == 'true')
        elif o == '-t':
            # 查询文本(数据库模式)
            query_text = a
        else:
            assert False, "未处理的选项"
            
    if use_db:
        if query_text is None:
            print("使用数据库模式时必须提供查询文本 -t 参数")
            sys.exit(2)
        search(use_db=True, query_text=query_text, sort_method=sort_method)
    else:
        if dictionary_file is None or postings_file is None or queries_file is None or output_file is None:
            print_usage()
            sys.exit(2)
        search(dictionary_file, postings_file, queries_file, output_file, sort_method)