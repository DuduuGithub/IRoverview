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
参数:
    - dictionary_file: 词典文件路径（文件模式时使用）
    - postings_file: 倒排索引文件路径（文件模式时使用）
    - queries_file: 查询文件路径（文件模式时使用）
    - output_file: 输出文件路径（文件模式时使用）
    - sort_method: 排序方法，默认按相关性排序
    - query_text: 查询文本（数据库模式时使用）
    - use_db: 是否使用数据库模式，默认为True
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
    use_db=True):
    
    # 数据库模式
    if use_db:
        if query_text is None:
            raise ValueError("使用数据库模式时必须提供query_text参数")
        return process_query_with_db(query_text, sort_method)
    
    # 文件模式
    else:
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


"""
从数据库处理查询
参数:
    - query_text: 查询文本
    - sort_method: 排序方法
返回:
    - 匹配的文档ID列表
"""
def process_query_with_db(query_text, sort_method=SORT_BY_RELEVANCE):
    # 标准化布尔操作符，确保空格分隔
    query_text = re.sub(r'\band\b', ' AND ', query_text, flags=re.IGNORECASE)
    query_text = re.sub(r'\bor\b', ' OR ', query_text, flags=re.IGNORECASE)
    query_text = re.sub(r'\bnot\b', ' NOT ', query_text, flags=re.IGNORECASE)
    
    # 判断是否为布尔查询
    has_boolean_ops = any(op in query_text for op in ['AND', 'OR', 'NOT', '(', ')'])

    print(f"查询文本: {query_text}")
    print(f"是否为布尔查询: {has_boolean_ops}")

    # 根据查询类型选择处理方法
    if has_boolean_ops:
        result = process_boolean_query_with_db(query_text)
    else:
        result = get_matching_docs_from_db(query_text)

    # 提取查询词项，用于排序
    query_terms = extract_query_terms(query_text)

    # 对结果进行排序（非布尔查询）
    if result and not has_boolean_ops and query_terms and sort_method:
        result = sort_db_results(result, query_text, sort_method)

    return result


"""
从数据库获取匹配的文档ID列表
参数:
    - query: 查询文本
返回:
    - 匹配的文档ID列表
"""
def get_matching_docs_from_db(query):
    # 对查询文本进行分词
    print(f"正在搜索关键词: {query}")
    query_terms = [term.lower().strip() for term in jieba.cut(query)]
    query_terms = [re.sub(r'[^\w\s]', '', term) for term in query_terms if term.strip()]
    
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
        
        # 通过 SQL 查询合并所有结果，减少数据库查询次数
        matches = []
        
        # 在标题、显示名称和摘要中搜索
        title_matches = Work.query.filter(
            (Work.title.ilike(f'%{term}%')) |
            (Work.display_name.ilike(f'%{term}%')) |
            (Work.abstract_inverted_index.cast(db.String).ilike(f'%{term}%'))
        ).all()
        matches.extend([w.id for w in title_matches])
        print(f"标题/显示名称/摘要匹配数: {len(title_matches)}")

        # 在作者中搜索
        author_matches = db.session.query(Work.id).join(
            WorkAuthorship, Work.id == WorkAuthorship.work_id
        ).join(
            Author, Author.id == WorkAuthorship.author_id
        ).filter(
            Author.display_name.ilike(f'%{term}%')
        ).all()
        matches.extend([m[0] for m in author_matches])
        print(f"作者匹配数: {len(author_matches)}")

        # 在概念中搜索
        concept_matches = db.session.query(Work.id).join(
            WorkConcept, Work.id == WorkConcept.work_id
        ).join(
            Concept, Concept.id == WorkConcept.concept_id
        ).filter(
            Concept.display_name.ilike(f'%{term}%')
        ).all()
        matches.extend([m[0] for m in concept_matches])
        print(f"概念匹配数: {len(concept_matches)}")

        # 在主题中搜索
        topic_matches = db.session.query(Work.id).join(
            WorkTopic, Work.id == WorkTopic.work_id
        ).join(
            Topic, Topic.id == WorkTopic.topic_id
        ).filter(
            Topic.display_name.ilike(f'%{term}%')
        ).all()
        matches.extend([m[0] for m in topic_matches])
        print(f"主题匹配数: {len(topic_matches)}")

        # 合并结果并去重
        term_results = list(set(matches))
        print(f"词项'{term}'的匹配结果数: {len(term_results)}")

        # 与之前的结果进行OR操作
        if not results:
            results = term_results
        else:
            results = list(set(results + term_results))
    
    print(f"总匹配结果数: {len(results)}")
    return results


"""
处理数据库布尔查询
参数:
    - query: 布尔查询表达式
返回:
    - 匹配的文档ID列表
"""
def process_boolean_query_with_db(query):
    # 处理查询字符串，确保括号周围有空格
    query = query.replace('(', ' ( ').replace(')', ' ) ')
    
    # 确保布尔操作符规范，前后加空格
    query = re.sub(r'\bAND\b', ' AND ', query, flags=re.IGNORECASE)
    query = re.sub(r'\bOR\b', ' OR ', query, flags=re.IGNORECASE)
    query = re.sub(r'\bNOT\b', ' NOT ', query, flags=re.IGNORECASE)
    query = re.sub(r'\s+', ' ', query).strip()  # 规范化多余的空格

    print(f"处理的布尔查询: {query}")

    # 分离操作符和词项
    operators = ['AND', 'OR', 'NOT', '(', ')']
    query_parts = []
    current_term = ""
    
    # 迭代处理查询词项，确保捕获多词项
    for part in query.split():
        if part in operators:
            # 添加之前积累的词项
            if current_term:
                query_parts.append(current_term.strip())
                current_term = ""
            # 添加操作符
            query_parts.append(part)
        else:
            # 连接非操作符部分
            if current_term:
                current_term += " " + part
            else:
                current_term = part
    
    # 添加最后一个词项
    if current_term:
        query_parts.append(current_term.strip())
    
    print(f"解析的查询词项: {query_parts}")
    
    # 规范化查询表达式
    # 如果只有一个词项，直接返回结果
    if len(query_parts) == 1 and query_parts[0] not in operators:
        return get_matching_docs_from_db(query_parts[0])
        
    # 如果是简单的 "A NOT B" 形式，转换为 "A AND NOT B"
    if len(query_parts) == 3 and query_parts[1] == 'NOT':
        print("检测到 'A NOT B' 简单形式，转换为 'A AND NOT B'")
        query_parts = [query_parts[0], 'AND', 'NOT', query_parts[2]]
        print(f"修正后的查询词项: {query_parts}")
    
    # 使用Shunting-yard算法将中缀表达式转换为后缀表达式
    try:
        postfix = shunting_yard(query_parts)
        print(f"生成的后缀表达式: {postfix}")
    except Exception as e:
        print(f"生成后缀表达式失败: {str(e)}")
        raise
    
    try:
        # 计算结果
        return evaluate_postfix_with_db(postfix)
    except Exception as e:
        print(f"表达式评估错误: {str(e)}")
        raise


"""
使用数据库查询评估后缀表达式
参数:
    - postfix: 后缀表达式
返回:
    - 匹配的文档ID列表
"""
def evaluate_postfix_with_db(postfix):
    stack = []
    
    # 延迟初始化，避免不必要的数据库查询
    all_doc_ids = None
    
    print(f"开始评估后缀表达式: {postfix}")
    
    for i, token in enumerate(postfix):
        try:
            if token == 'AND':
                if len(stack) < 2:
                    raise ValueError(f"表达式错误：AND操作符缺少操作数。当前栈: {stack}")
                right = stack.pop()
                left = stack.pop()
                result = boolean_AND(left, right)
                print(f"执行 AND 操作: {len(left)} AND {len(right)} = {len(result)} 条结果")
                stack.append(result)
            elif token == 'OR':
                if len(stack) < 2:
                    raise ValueError(f"表达式错误：OR操作符缺少操作数。当前栈: {stack}")
                right = stack.pop()
                left = stack.pop()
                result = boolean_OR(left, right)
                print(f"执行 OR 操作: {len(left)} OR {len(right)} = {len(result)} 条结果")
                stack.append(result)
            elif token == 'NOT':
                if len(stack) < 1:
                    raise ValueError(f"表达式错误：NOT操作符缺少操作数。当前栈: {stack}")
                operand = stack.pop()
                # 延迟加载所有文档ID
                if all_doc_ids is None:
                    all_doc_ids = [work.id for work in Work.query.with_entities(Work.id).all()]
                    print(f"加载全部文档ID: {len(all_doc_ids)} 条")
                result = boolean_NOT(operand, all_doc_ids)
                print(f"执行 NOT 操作: NOT {len(operand)} = {len(result)} 条结果")
                stack.append(result)
            else:
                # 词项，获取匹配的文档ID
                result = get_matching_docs_from_db(token)
                print(f"查询词项 '{token}': 找到 {len(result)} 条结果")
                stack.append(result)
            
            print(f"处理完 token '{token}' 后栈状态: {[len(x) for x in stack]} 个结果集")
            
        except Exception as e:
            print(f"处理 token '{token}' (位置 {i+1}/{len(postfix)}) 时出错: {str(e)}")
            raise
    
    if len(stack) != 1:
        raise ValueError(f"表达式错误：操作符与操作数不匹配。最终栈: {[len(x) for x in stack]}")
    
    return stack[0]


"""
对数据库查询结果进行排序
参数:
    - result_ids: 结果文档ID列表
    - query_text: 查询文本
    - sort_method: 排序方法
返回:
    - 排序后的文档ID列表
"""
def sort_db_results(result_ids, query_text, sort_method):
    if not result_ids:
        return []
    
    # 从数据库获取文档详情
    docs = {}
    for doc_id in result_ids:
        work = Work.query.get(doc_id)
        if work:
            # 获取作者信息
            authorships = WorkAuthorship.query.filter_by(work_id=work.id).all()
            authors = []
            for authorship in authorships:
                if authorship.author_id:
                    author = Author.query.get(authorship.author_id)
                    if author:
                        authors.append(author.display_name)
                
            # 构建文档对象，包含排序需要的信息
            docs[doc_id] = {
                'id': work.id,
                'openalex': work.openalex or None,
                'doi': work.doi or None,
                'title': work.title or work.display_name or '',
                'abstract': work.abstract_inverted_index or '',
                'authors': authors,
                'year': work.publication_year or 0,
                'cited_by_count': work.cited_by_count or 0,
                'publication_date': work.publication_date or datetime.now()
            }
    
    # 根据排序方法对结果进行排序
    if sort_method == SORT_BY_RELEVANCE:
        # 按相关性排序 - 简单实现基于词项匹配频率
        query_terms = extract_query_terms(query_text)
        scored_docs = []
        
        for doc_id, doc in docs.items():
            score = 0
            text = (doc['title'] + ' ' + doc['abstract']).lower()
            for term in query_terms:
                score += text.count(term.lower())
            scored_docs.append((doc_id, score))
        
        # 按分数降序排序
        return [doc_id for doc_id, score in sorted(scored_docs, key=lambda x: x[1], reverse=True)]
    
    elif sort_method == SORT_BY_TIME_DESC:
        # 按时间降序
        return [doc_id for doc_id, _ in sorted(docs.items(), key=lambda x: x[1]['year'], reverse=True)]
    
    elif sort_method == SORT_BY_TIME_ASC:
        # 按时间升序
        return [doc_id for doc_id, _ in sorted(docs.items(), key=lambda x: x[1]['year'])]
    
    elif sort_method == SORT_BY_COMBINED:
        # 组合排序：结合相关性和引用次数
        query_terms = extract_query_terms(query_text)
        scored_docs = []
        
        for doc_id, doc in docs.items():
            # 相关性分数
            relevance_score = 0
            text = (doc['title'] + ' ' + doc['abstract']).lower()
            for term in query_terms:
                relevance_score += text.count(term.lower())
            
            # 引用分数
            citation_score = doc['cited_by_count']
            
            # 综合分数
            combined_score = relevance_score * 0.7 + (citation_score / 100) * 0.3
            scored_docs.append((doc_id, combined_score))
        
        # 按综合分数降序排序
        return [doc_id for doc_id, score in sorted(scored_docs, key=lambda x: x[1], reverse=True)]
    
    # 默认返回原始顺序
    return result_ids


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
    
    # 使用jieba分词
    return [term.lower().strip() for term in jieba.cut(cleaned_query) if term.strip()]


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
    # 处理查询字符串，确保括号周围有空格
    query = query.replace('(', ' ( ').replace(')', ' ) ')
    query = re.sub(r'\s+', ' ', query).strip()  # 规范化空格
    
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
            # 对非操作符部分进行分词
            for term in jieba.cut(part):
                term = term.lower().strip()
                if term:  # 忽略空词项
                    tokens.append(term)
    
    # 使用Shunting-yard算法转换为后缀表达式
    postfix = shunting_yard(tokens)
    
    # 计算后缀表达式
    return evaluate_postfix(postfix, dictionary, post_file, indexed_docIDs)


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


"""
获取与查询匹配的文档ID列表
参数:
    - query: 查询文本
    - dictionary: 词典
    - post_file: 倒排索引文件对象
返回:
    - 匹配的文档ID列表
"""
def get_matching_docs(query, dictionary, post_file):
    # 分词
    terms = [term.lower().strip() for term in jieba.cut(query) if term.strip()]
    
    # 如果没有有效词项，返回空列表
    if not terms:
        return []
    
    # 对每个词项获取匹配文档，然后取并集
    results = []
    for term in terms:
        if term in dictionary:
            df, _, offset = dictionary[term]
            postings_list = load_posting_list(post_file, df, offset)
            if not results:
                results = postings_list
            else:
                results = boolean_OR(results, postings_list)
    
    return results

"""
计算两个列表的逻辑与（交集）
参数:
    - list1, list2: 两个文档ID列表
返回:
    - 交集结果
"""
def boolean_AND(list1, list2):
    # 优化：根据列表长度决定使用哪种方法
    if not list1 or not list2:
        return []
        
    if len(list1) > 10*len(list2) or len(list2) > 10*len(list1):
        # 长度差异大时使用二分查找
        if len(list1) < len(list2):
            return binary_search_AND(list2, list1)
        else:
            return binary_search_AND(list1, list2)
    elif len(list1) + len(list2) > 1000:
        # 列表较长时使用跳跃表方法
        return skip_list_AND(list1, list2)
    else:
        # 列表较短时使用简单扫描
        return [x for x in list1 if x in list2]


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
    post_file.seek(offset)
    result = []
    
    for i in range(length):
        # 每个记录包含docID(4字节)和词频(4字节)
        docID_bytes = post_file.read(BYTE_SIZE)
        tf_bytes = post_file.read(BYTE_SIZE)  # 读取但不使用词频
        
        # 将字节转换为整数
        docID = struct.unpack('I', docID_bytes)[0]
        result.append(docID)
    
    return result


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
    post_file.seek(offset)
    result = []
    
    for i in range(length):
        # 读取docID和词频
        docID_bytes = post_file.read(BYTE_SIZE)
        tf_bytes = post_file.read(BYTE_SIZE)
        
        # 将字节转换为整数
        docID = struct.unpack('I', docID_bytes)[0]
        tf = struct.unpack('I', tf_bytes)[0]
        
        result.append((docID, tf))
    
    return result


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
        
    search(dictionary_file, postings_file, queries_file, output_file, use_db=False)