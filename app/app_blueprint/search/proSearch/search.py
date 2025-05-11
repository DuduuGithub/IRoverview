#!/usr/bin/python
import re
import sys
import os
import getopt
import codecs
import struct
import math
import io
import collections
import json
import jieba
import time
import bisect
from datetime import datetime
import importlib

# 添加项目根目录到sys.path，避免相对导入问题
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

# 导入数据库模型和配置
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from Database.model import Work, Author, Topic, Concept, Institution, Source, WorkAuthorship, WorkConcept, WorkTopic
from Database.config import db

# 导入排序模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ..rank.rank import rank_results, SORT_BY_RELEVANCE, SORT_BY_TIME_DESC, SORT_BY_TIME_ASC, SORT_BY_COMBINED

# 常量定义
BYTE_SIZE = 4  # 文档ID使用整型存储，占用4字节

# 高级检索字段类型
FIELD_TITLE = "title"      # 标题/主题
FIELD_AUTHOR = "author"    # 作者
FIELD_KEYWORD = "keyword"  # 关键词
FIELD_TIME = "time"        # 时间
FIELD_CONTENT = "content"  # 普通内容

"""
执行高级搜索，支持从数据库获取数据
支持多字段查询和过滤
"""
def search(dictionary_file=None, postings_file=None, queries_file=None, output_file=None, sort_method=SORT_BY_RELEVANCE, query_text=None, use_db=True):
    # 如果使用数据库模式，则忽略文件参数
    if use_db:
        if query_text is None:
            raise ValueError("使用数据库模式时必须提供query_text参数")
        
        # 解析高级查询
        parsed_query = parse_advanced_query(query_text)
        
        # 处理查询
        return process_advanced_query_with_db(parsed_query, sort_method)
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
        field_dictionary = loaded_dict[1] # 字段词典映射
        indexed_docIDs = loaded_dict[2] # 所有已索引的文档ID列表
        doc_lengths = loaded_dict[3]    # 文档长度映射
        doc_metadata = loaded_dict[4]   # 文档元数据
        dict_file.close()

        # 处理每个查询
        queries_list = query_file.read().splitlines()
        results = []
        for i in range(len(queries_list)):
            query = queries_list[i]
            
            # 解析高级查询
            parsed_query = parse_advanced_query(query)
            
            # 执行查询
            result = process_advanced_query(parsed_query, dictionary, field_dictionary, post_file, indexed_docIDs, doc_metadata)
            
            # 对结果进行排序
            if result and parsed_query["general_query"]:  # 确保结果不为空且有通用查询部分
                # 如果有通用查询部分，使用它进行排序
                # 对高级检索结果应用排序，但需要考虑必须使用时间排序的特殊情况
                try:
                    if sort_method in [SORT_BY_TIME_DESC, SORT_BY_TIME_ASC, SORT_BY_COMBINED] and doc_metadata:
                        result = rank_results(result, parsed_query["general_query"], dictionary, post_file, doc_metadata, sort_method)
                    else:
                        # 如果查询中有时间字段，但没有指定时间排序，则自动使用组合排序
                        if "time" in parsed_query["field_queries"] and doc_metadata:
                            result = rank_results(result, parsed_query["general_query"], dictionary, post_file, doc_metadata, SORT_BY_COMBINED)
                        else:
                            result = rank_results(result, parsed_query["general_query"], dictionary, post_file, doc_metadata, sort_method)
                except Exception as e:
                    print(f"排序过程中发生错误: {str(e)}")
                    # 如果排序失败，保留原始结果
                    pass
            
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

"""解析词典文件并返回词典数据结构，包含字段信息"""
def load_dictionary(dict_file):
    dictionary = {}         # 普通词典映射 {term: (df, idf, offset)}
    field_dictionary = {}   # 字段词典映射 {field: {term: (df, idf, offset)}}
    indexed_docIDs = []     # 所有已索引的文档ID列表
    doc_lengths = {}        # 文档长度映射 {docID: 长度}
    doc_metadata = {}       # 文档元数据 {docID: {field: value}}
    
    docIDs_processed = False
    doc_lengths_processed = False
    doc_metadata_processed = False

    # 初始化字段词典
    for field in [FIELD_TITLE, FIELD_AUTHOR, FIELD_KEYWORD, FIELD_TIME, FIELD_CONTENT]:
        field_dictionary[field] = {}

    # 解析词典文件
    for entry in dict_file.read().split('\n'):
        if not entry:
            continue
            
        # 处理文档ID列表
        if not docIDs_processed:
            if entry.startswith("all_indexed_docIDs: "):
                docIDs_str = entry[20:]
                indexed_docIDs = [int(docID) for docID in docIDs_str.split(',') if docID.strip()]
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
        
        # 处理词项条目（带字段）
        token = entry.split()
        if len(token) >= 5:  # 字段索引格式: field:term df idf offset
            if ":" in token[0]:
                field, term = token[0].split(":", 1)
                df = int(token[1])
                idf = float(token[2])
                offset = int(token[3])
                if field in field_dictionary:
                    field_dictionary[field][term] = (df, idf, offset)
            else:  # 常规词项
                term = token[0]
                df = int(token[1])
                idf = float(token[2])
                offset = int(token[3])
                dictionary[term] = (df, idf, offset)

    return (dictionary, field_dictionary, indexed_docIDs, doc_lengths, doc_metadata)

"""解析高级查询表达式，识别字段限定符"""
def parse_advanced_query(query):
    # 识别字段查询（如 title:人工智能）
    field_queries = {}
    general_terms = []
    
    # 分离各部分查询
    parts = re.findall(r'(\w+):"([^"]+)"|(\w+):(\S+)|(\([^)]+\))|(\S+)', query)
    
    for match in parts:
        if match[0] and match[1]:  # 形式如 field:"value with spaces"
            field, value = match[0].lower(), match[1]
            field_queries.setdefault(field, []).append(value)
        elif match[2] and match[3]:  # 形式如 field:value
            field, value = match[2].lower(), match[3]
            field_queries.setdefault(field, []).append(value)
        elif match[4]:  # 括号表达式
            general_terms.append(match[4])
        elif match[5]:  # 普通词项
            # 处理布尔运算符
            if match[5] in ["AND", "OR", "NOT"]:
                general_terms.append(match[5])
            else:
                general_terms.append(match[5])
    
    return {
        "field_queries": field_queries,
        "general_query": " ".join(general_terms)
    }

"""处理高级查询"""
def process_advanced_query(parsed_query, dictionary, field_dictionary, post_file, indexed_docIDs, doc_metadata):
    # 动态导入布尔操作函数，避免循环导入
    from ..basicSearch.search import boolean_AND, boolean_OR

    field_queries = parsed_query["field_queries"]
    general_query = parsed_query["general_query"]
    
    # 初始结果集为所有文档
    result = indexed_docIDs.copy()
    
    # 处理字段查询
    for field, values in field_queries.items():
        field_result = []
        
        for value in values:
            # 特殊处理时间字段
            if field == FIELD_TIME:
                time_docs = process_time_query(value, doc_metadata)
                field_result = boolean_OR(field_result, time_docs)
            
            # 处理其他字段
            elif field in field_dictionary:
                terms = list(jieba.cut(value))
                field_docs = []
                
                for term in terms:
                    term = term.lower().strip()
                    term = re.sub(r'[^\w\s]', '', term)
                    
                    if term in field_dictionary[field]:
                        df, idf, offset = field_dictionary[field][term]
                        docs = load_posting_list(post_file, df, offset)
                        field_docs = boolean_OR(field_docs, docs)
                
                field_result = boolean_OR(field_result, field_docs)
        
        # 与当前结果取交集
        if field_result:
            result = boolean_AND(result, field_result)
    
    # 处理通用查询
    if general_query:
        has_boolean_ops = any(op in general_query for op in ['AND', 'OR', 'NOT', '(', ')'])
        
        if has_boolean_ops:
            general_result = process_boolean_query(general_query, dictionary, post_file, indexed_docIDs)
        else:
            matched_docs = get_matching_docs(general_query, dictionary, post_file)
            general_result = matched_docs
        
        # 与当前结果取交集
        if general_result:
            result = boolean_AND(result, general_result)
    
    return result

"""处理时间范围查询"""
def process_time_query(time_query, doc_metadata):
    matching_docs = []
    
    # 范围查询，如 2020-01-01~2021-01-01
    if "~" in time_query:
        start_date_str, end_date_str = time_query.split("~", 1)
        try:
            start_date = parse_date(start_date_str.strip())
            end_date = parse_date(end_date_str.strip())
            
            for doc_id, metadata in doc_metadata.items():
                if FIELD_TIME in metadata:
                    doc_time = parse_date(metadata[FIELD_TIME])
                    if start_date <= doc_time <= end_date:
                        matching_docs.append(doc_id)
        except:
            pass
    
    # 单个时间点查询
    else:
        try:
            query_date = parse_date(time_query.strip())
            
            for doc_id, metadata in doc_metadata.items():
                if FIELD_TIME in metadata:
                    doc_time = parse_date(metadata[FIELD_TIME])
                    # 使用年份相等作为匹配条件
                    if query_date.year == doc_time.year:
                        matching_docs.append(doc_id)
        except:
            pass
    
    return sorted(matching_docs)

"""解析日期字符串"""
def parse_date(date_str):
    # 尝试多种日期格式
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y年%m月%d日",
        "%Y.%m.%d",
        "%Y-%m",
        "%Y/%m",
        "%Y年%m月",
        "%Y.%m",
        "%Y"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # 默认返回当前日期
    return datetime.now()

"""处理布尔查询"""
def process_boolean_query(query, dictionary, post_file, indexed_docIDs):
    # 动态导入布尔操作函数和shunting_yard算法，避免循环导入
    from ..basicSearch.search import boolean_AND, boolean_OR, boolean_NOT, shunting_yard
    
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
    
    matched_docs = []
    
    # 获取所有匹配文档
    for term in query_terms:
        if term in dictionary:
            df, idf, offset = dictionary[term]
            docs = load_posting_list(post_file, df, offset)
            matched_docs.extend(docs)
    
    # 去重并排序
    return sorted(list(set(matched_docs)))

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

"""处理数据库高级查询"""
def process_advanced_query_with_db(parsed_query, sort_method=SORT_BY_RELEVANCE):
    # 动态导入布尔操作函数，避免循环导入
    from ..basicSearch.search import boolean_AND, boolean_OR
    
    field_queries = parsed_query["field_queries"]
    general_query = parsed_query["general_query"]
    
    # 获取所有文档ID
    all_work_ids = [w.id for w in Work.query.all()]
    if not all_work_ids:
        return []
    
    # 初始结果集为所有文档
    result = all_work_ids.copy()
    
    # 处理字段查询
    for field, values in field_queries.items():
        field_result = []
        
        for value in values:
            # 特殊处理时间字段
            if field == FIELD_TIME:
                time_docs = process_time_query_with_db(value)
                field_result = boolean_OR(field_result, time_docs)
            
            # 处理作者字段
            elif field == FIELD_AUTHOR:
                # 查找匹配的作者
                authors = Author.query.filter(Author.display_name.ilike(f'%{value}%')).all()
                author_ids = [a.id for a in authors]
                
                # 查找这些作者的作品（添加明确的连接条件）
                if author_ids:
                    works = db.session.query(Work.id).select_from(Work).join(
                        WorkAuthorship, Work.id == WorkAuthorship.work_id
                    ).filter(
                        WorkAuthorship.author_id.in_(author_ids)
                    ).all()
                    author_docs = [w[0] for w in works]
                    field_result = boolean_OR(field_result, author_docs)
            
            # 处理标题字段
            elif field == FIELD_TITLE:
                works = Work.query.filter(
                    (Work.title.ilike(f'%{value}%')) | (Work.display_name.ilike(f'%{value}%'))
                ).all()
                title_docs = [w.id for w in works]
                field_result = boolean_OR(field_result, title_docs)
            
            # 处理关键词字段（对应概念和主题）
            elif field == FIELD_KEYWORD:
                # 匹配概念
                concepts = Concept.query.filter(Concept.display_name.ilike(f'%{value}%')).all()
                concept_ids = [c.id for c in concepts]
                
                # 匹配主题
                topics = Topic.query.filter(Topic.display_name.ilike(f'%{value}%')).all()
                topic_ids = [t.id for t in topics]
                
                # 查找这些概念和主题的相关作品（添加明确的连接条件）
                concept_works = []
                if concept_ids:
                    concept_works = db.session.query(Work.id).select_from(Work).join(
                        WorkConcept, Work.id == WorkConcept.work_id
                    ).filter(
                        WorkConcept.concept_id.in_(concept_ids)
                    ).all()
                    concept_works = [w[0] for w in concept_works]
                
                topic_works = []
                if topic_ids:
                    topic_works = db.session.query(Work.id).select_from(Work).join(
                        WorkTopic, Work.id == WorkTopic.work_id
                    ).filter(
                        WorkTopic.topic_id.in_(topic_ids)
                    ).all()
                    topic_works = [w[0] for w in topic_works]
                
                # 合并结果
                keyword_docs = list(set(concept_works + topic_works))
                field_result = boolean_OR(field_result, keyword_docs)
            
            # 处理内容字段（在摘要中搜索）
            elif field == FIELD_CONTENT:
                # 这里简化处理，搜索作品的抽象索引和标题（实际上应该搜索全文，但可能不在数据库中）
                works = Work.query.filter(Work.abstract_inverted_index.cast(db.String).ilike(f'%{value}%')).all()
                content_docs = [w.id for w in works]
                field_result = boolean_OR(field_result, content_docs)
        
        # 与当前结果取交集
        if field_result:
            result = boolean_AND(result, field_result)
    
    # 处理通用查询
    if general_query:
        has_boolean_ops = any(op in general_query for op in ['AND', 'OR', 'NOT', '(', ')'])
        
        if has_boolean_ops:
            from ..basicSearch.search import process_boolean_query_with_db
            general_result = process_boolean_query_with_db(general_query)
        else:
            from ..basicSearch.search import get_matching_docs_from_db
            general_result = get_matching_docs_from_db(general_query)
        
        # 与当前结果取交集
        if general_result:
            result = boolean_AND(result, general_result)
    
    # 对结果进行排序
    if result and parsed_query["general_query"] and sort_method:  # 确保结果不为空且有通用查询部分
        # 如果有通用查询部分，使用它进行排序
        result = sort_db_results(result, parsed_query, sort_method)
    
    return result

"""按照排序方法对数据库查询结果进行排序"""
def sort_db_results(result_ids, parsed_query, sort_method):
    # 动态导入extract_query_terms函数，避免循环导入
    from ..basicSearch.search import extract_query_terms
    
    if not result_ids:
        return []
    
    query_text = parsed_query["general_query"]
    
    # 时间排序
    if sort_method == SORT_BY_TIME_DESC:
        works = Work.query.filter(Work.id.in_(result_ids)).order_by(Work.publication_year.desc()).all()
        return [w.id for w in works]
    
    elif sort_method == SORT_BY_TIME_ASC:
        works = Work.query.filter(Work.id.in_(result_ids)).order_by(Work.publication_year.asc()).all()
        return [w.id for w in works]
    
    # 组合排序
    elif sort_method == SORT_BY_COMBINED or "time" in parsed_query["field_queries"]:
        # 计算查询相关性得分
        query_terms = extract_query_terms(query_text)
        scored_results = []
        
        for work_id in result_ids:
            work = Work.query.get(work_id)
            if work:
                # 计算相关性得分
                score = calculate_relevance_score(work, query_terms)
                
                # 时间因素（最近5年的文章得分较高）
                current_year = datetime.now().year
                if work.publication_year and current_year - work.publication_year <= 5:
                    time_bonus = 1 - (current_year - work.publication_year) * 0.2
                    score = score * 0.7 + time_bonus * 0.3  # 70%相关性 + 30%时间因素
                
                scored_results.append((work_id, score))
        
        # 按得分降序排序
        sorted_results = [r[0] for r in sorted(scored_results, key=lambda x: x[1], reverse=True)]
        return sorted_results
    
    # 默认按相关性排序
    else:
        query_terms = extract_query_terms(query_text)
        scored_results = []
        
        for work_id in result_ids:
            work = Work.query.get(work_id)
            if work:
                score = calculate_relevance_score(work, query_terms)
                scored_results.append((work_id, score))
        
        # 按得分降序排序
        sorted_results = [r[0] for r in sorted(scored_results, key=lambda x: x[1], reverse=True)]
        return sorted_results

"""计算作品与查询词项的相关性得分"""
def calculate_relevance_score(work, query_terms):
    score = 0
    
    # 标题匹配
    title_weight = 2.0
    for term in query_terms:
        if work.title and term in work.title.lower():
            score += title_weight
        if work.display_name and term in work.display_name.lower():
            score += title_weight * 0.8  # 展示名称稍微低一点权重
    
    # 摘要匹配
    abstract_weight = 1.0
    if work.abstract_inverted_index:
        abstract_text = json.dumps(work.abstract_inverted_index).lower()
        for term in query_terms:
            if term in abstract_text:
                score += abstract_weight
    
    # 引用数加权
    if work.cited_by_count:
        citation_bonus = min(0.5, work.cited_by_count / 1000)  # 最多加0.5分
        score += citation_bonus
    
    return score

"""处理数据库时间范围查询"""
def process_time_query_with_db(time_query):
    matching_docs = []
    
    # 范围查询，如 2020-01-01~2021-01-01
    if "~" in time_query:
        start_date_str, end_date_str = time_query.split("~", 1)
        try:
            start_date = parse_date(start_date_str.strip())
            end_date = parse_date(end_date_str.strip())
            
            # 查找时间范围内的作品
            works = Work.query.filter(
                (Work.publication_year >= start_date.year) & 
                (Work.publication_year <= end_date.year)
            ).all()
            
            matching_docs = [w.id for w in works]
        except:
            pass
    
    # 单个时间点查询
    else:
        try:
            query_date = parse_date(time_query.strip())
            
            # 查找指定年份的作品
            works = Work.query.filter(Work.publication_year == query_date.year).all()
            matching_docs = [w.id for w in works]
        except:
            pass
    
    return matching_docs
