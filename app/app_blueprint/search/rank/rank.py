#!/usr/bin/python
import re
import sys
import math
import jieba
import io
import struct
from datetime import datetime
import os
import json

# 导入数据库模型和配置
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from Database.model import Work, Author, Topic, Concept, Institution, Source, WorkAuthorship, WorkConcept, WorkTopic
from Database.config import db

# 排序方法常量
SORT_BY_RELEVANCE = "relevance"    # 按相关性排序
SORT_BY_TIME_DESC = "time_desc"    # 按时间降序（新到旧）
SORT_BY_TIME_ASC = "time_asc"      # 按时间升序（旧到新）
SORT_BY_COMBINED = "combined"      # 结合相关性和时间

# 时间权重
TIME_WEIGHT = 0.3  # 时间因素的权重

"""
对搜索结果进行排序
result: 要排序的文档列表
query: 查询字符串
dictionary: 词典
post_file: 倒排索引文件
doc_metadata: 文档元数据
sort_method: 排序方法
use_db: 是否使用数据库模式
"""
def rank_results(result, query, dictionary=None, post_file=None, doc_metadata=None, 
                 sort_method=SORT_BY_RELEVANCE, use_db=False):
    if not result:
        return []
    
    # 如果使用数据库模式
    if use_db:
        return rank_results_with_db(result, query, sort_method)
    
    # 传统文件模式
    # 如果按时间排序
    if sort_method == SORT_BY_TIME_DESC:
        return sort_by_time(result, doc_metadata, reverse=True)
    elif sort_method == SORT_BY_TIME_ASC:
        return sort_by_time(result, doc_metadata, reverse=False)
    elif sort_method == SORT_BY_COMBINED:
        # 组合排序：相关度 + 时间
        tfidf_ranked = sort_by_tfidf(query, result, dictionary, post_file)
        time_ranked = sort_by_time(result, doc_metadata, reverse=True)
        return combine_rankings(tfidf_ranked, time_ranked)
    else:
        # 默认按TF-IDF相关性排序
        return sort_by_tfidf(query, result, dictionary, post_file)

"""使用数据库进行排序"""
def rank_results_with_db(result_ids, query_text, sort_method=SORT_BY_RELEVANCE):
    if not result_ids:
        return []
    
    # 如果按时间排序
    if sort_method == SORT_BY_TIME_DESC:
        works = Work.query.filter(Work.id.in_(result_ids)).order_by(Work.publication_year.desc()).all()
        return [w.id for w in works]
        
    elif sort_method == SORT_BY_TIME_ASC:
        works = Work.query.filter(Work.id.in_(result_ids)).order_by(Work.publication_year.asc()).all()
        return [w.id for w in works]
        
    elif sort_method == SORT_BY_COMBINED:
        # 计算查询相关性得分
        query_terms = [term.lower().strip() for term in jieba.cut(query_text)]
        query_terms = [re.sub(r'[^\w\s]', '', term) for term in query_terms if term.strip()]
        
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
                    score = score * (1.0 - TIME_WEIGHT) + time_bonus * TIME_WEIGHT
                
                scored_results.append((work_id, score))
        
        # 按得分降序排序
        sorted_results = [r[0] for r in sorted(scored_results, key=lambda x: x[1], reverse=True)]
        return sorted_results
    
    else:
        # 默认按相关性排序
        query_terms = [term.lower().strip() for term in jieba.cut(query_text)]
        query_terms = [re.sub(r'[^\w\s]', '', term) for term in query_terms if term.strip()]
        
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

"""根据TF-IDF对文档列表进行排序"""
def sort_by_tfidf(query, doc_list, dictionary, post_file):
    if not doc_list:
        return []
    
    # 分词
    query_terms = [term.lower().strip() for term in jieba.cut(query)]
    query_terms = [re.sub(r'[^\w\s]', '', term) for term in query_terms if term.strip()]
    
    scores = {docID: 0.0 for docID in doc_list}
    
    # 计算每个查询词项的贡献
    for term in query_terms:
        if term in dictionary:
            df, idf, offset = dictionary[term]
            
            # 获取词项的倒排列表
            posting_list_with_tf = load_posting_list_with_tf(post_file, df, offset)
            
            # 更新文档得分
            for docID, tf in posting_list_with_tf:
                if docID in scores:
                    scores[docID] += tf * idf
    
    # 归一化得分
    max_score = max(scores.values()) if scores else 1.0
    
    # 避免除以零错误
    if max_score > 0:
        for docID in scores:
            scores[docID] /= max_score
    
    # 去除零分文档
    for docID in list(scores.keys()):
        if scores[docID] <= 0:
            scores.pop(docID)
    
    # 按得分降序排序
    ranked_docs = [doc for doc, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    
    # 如果经过排序后没有文档，返回原始列表
    if not ranked_docs:
        return doc_list
    
    return ranked_docs

"""根据时间对文档列表进行排序"""
def sort_by_time(doc_list, doc_metadata, reverse=True):
    if not doc_list or not doc_metadata:
        return doc_list
    
    # 从元数据中提取时间
    doc_times = {}
    
    for docID in doc_list:
        if docID in doc_metadata and 'time' in doc_metadata[docID]:
            try:
                time_str = doc_metadata[docID]['time']
                time_obj = parse_date(time_str)
                doc_times[docID] = time_obj
            except:
                # 设置默认时间
                doc_times[docID] = datetime.min
        else:
            # 如果没有时间信息，设置为最小时间
            doc_times[docID] = datetime.min
    
    # 按时间排序
    ranked_docs = sorted(doc_list, key=lambda docID: doc_times.get(docID, datetime.min), reverse=reverse)
    
    return ranked_docs

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
    
    # 默认返回最小日期
    return datetime.min

"""组合多个排序结果"""
def combine_rankings(tfidf_ranked, time_ranked):
    if not tfidf_ranked:
        return time_ranked
    if not time_ranked:
        return tfidf_ranked
    
    # 为每个文档计算组合得分
    all_docs = set(tfidf_ranked) | set(time_ranked)
    
    # 计算排名得分
    tfidf_scores = {doc: 1.0 - (tfidf_ranked.index(doc) / len(tfidf_ranked) if doc in tfidf_ranked else 1.0) for doc in all_docs}
    time_scores = {doc: 1.0 - (time_ranked.index(doc) / len(time_ranked) if doc in time_ranked else 1.0) for doc in all_docs}
    
    # 组合得分
    combined_scores = {doc: (1.0 - TIME_WEIGHT) * tfidf_scores[doc] + TIME_WEIGHT * time_scores[doc] for doc in all_docs}
    
    # 按组合得分排序
    ranked_docs = [doc for doc, score in sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)]
    
    return ranked_docs

"""获取一个词项的倒排列表（包含词频）"""
def load_posting_list_with_tf(post_file, length, offset):
    post_file.seek(offset)
    posting_list = []
    for i in range(length):
        # 读取文档ID和词频
        posting = post_file.read(4)
        docID = struct.unpack('I', posting)[0]
        
        tf_data = post_file.read(4)
        tf = struct.unpack('I', tf_data)[0]
        
        posting_list.append((docID, tf))
    return posting_list