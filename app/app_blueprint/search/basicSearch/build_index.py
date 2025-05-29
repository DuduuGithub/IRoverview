#!/usr/bin/python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from flask import Flask
from Database.model import Work, Author, Topic, Concept, Institution, Source, WorkAuthorship, WorkConcept, WorkTopic
from Database.config import db
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import math
import re
import time
from datetime import datetime

# 创建 Flask 应用
app = Flask(__name__)
# 配置数据库
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:240921@localhost/iroverview'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# 初始化数据库
db.init_app(app)

# 初始化NLTK组件
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

def process_abstract_inverted_index(abstract_index):
    """处理摘要倒排索引JSON数据"""
    if not abstract_index:
        return ""
    
    # 如果输入是字符串，尝试解析为JSON
    if isinstance(abstract_index, str):
        try:
            abstract_index = json.loads(abstract_index)
        except json.JSONDecodeError:
            return abstract_index  # 如果解析失败，直接返回原始字符串
    
    # 将倒排索引转换为普通文本
    words = []
    for word, positions in abstract_index.items():
        # 根据位置信息重建文本
        for pos in positions:
            words.append((pos, word))
    
    # 按位置排序并连接成文本
    return " ".join(word for _, word in sorted(words))

def build_inverted_index():
    """构建倒排索引"""
    start_time = time.time()
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始构建倒排索引...")
    
    with app.app_context():
        # 获取所有工作文档
        print("正在从数据库获取文档...")
        works = Work.query.all()
        total_works = len(works)
        print(f"共获取到 {total_works} 个文档")
        
        # 初始化数据结构
        dictionary = {}  # 词典 {term: (df, idf, offset)}
        postings = {}    # 倒排列表 {term: [(doc_id, tf), ...]}
        doc_lengths = {} # 文档长度 {doc_id: length}
        doc_metadata = {} # 文档元数据 {doc_id: {field: value}}
        
        # 处理每个文档
        for work in works:
            doc_id = work.id
            
            # 获取文档内容，保持字段分离
            title = work.title or work.display_name or ''
            abstract = process_abstract_inverted_index(work.abstract_inverted_index) if work.abstract_inverted_index else ''
            
            # 获取作者信息
            authorships = WorkAuthorship.query.filter_by(work_id=work.id).all()
            authors = []
            for authorship in authorships:
                if authorship.author_id:
                    author = Author.query.get(authorship.author_id)
                    if author:
                        authors.append(author.display_name)
            
            # 获取概念信息
            concepts = WorkConcept.query.filter_by(work_id=work.id).all()
            concept_names = []
            for concept in concepts:
                if concept.concept_id:
                    concept_obj = db.session.get(Concept, concept.concept_id)
                    if concept_obj:
                        concept_names.append(concept_obj.display_name)
            
            # 获取主题信息
            topics = WorkTopic.query.filter_by(work_id=work.id).all()
            topic_names = []
            for topic in topics:
                if topic.topic_id:
                    topic_obj = db.session.get(Topic, topic.topic_id)
                    if topic_obj:
                        topic_names.append(topic_obj.display_name)
            
            # 分别处理每个字段
            fields = {
                'title': title,
                'abstract': abstract,
                'authors': ' '.join(authors),
                'concepts': ' '.join(concept_names),
                'topics': ' '.join(topic_names)
            }
            
            # 存储文档元数据
            doc_metadata[doc_id] = {
                'id': work.id,
                'openalex': work.openalex,
                'doi': work.doi,
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'concepts': concept_names,
                'topics': topic_names,
                'year': work.publication_year,
                'cited_by_count': work.cited_by_count,
                'publication_date': work.publication_date.isoformat() if work.publication_date else None,
                'type': work.type,
                'language': work.language,
                'is_retracted': work.is_retracted,
                'is_paratext': work.is_paratext
            }
            
            # 处理每个字段
            total_terms = 0
            for field, content in fields.items():
                # 分词
                terms = tokenize_english(content)
                total_terms += len(terms)
                
                # 统计词频
                term_freq = {}
                for term in terms:
                    term_freq[term] = term_freq.get(term, 0) + 1
                
                # 更新倒排列表
                for term, tf in term_freq.items():
                    if term not in postings:
                        postings[term] = []
                    postings[term].append((doc_id, tf))
            
            # 记录文档总词数
            doc_lengths[doc_id] = total_terms
        
        # 计算文档频率和IDF
        N = len(works)  # 文档总数
        for term, posting_list in postings.items():
            df = len(posting_list)  # 文档频率
            idf = math.log(N / df) if df > 0 else 0  # 逆文档频率
            dictionary[term] = (df, idf, 0)  # offset将在写入文件时更新
        
        # 创建索引目录
        index_dir = os.path.join(os.path.dirname(__file__), 'index')
        os.makedirs(index_dir, exist_ok=True)
        
        # 写入词典文件
        dict_file = os.path.join(index_dir, 'dictionary.txt')
        with open(dict_file, 'w', encoding='utf-8') as f:
            # 写入所有已索引的文档ID
            f.write(f"all_indexed_docIDs:{','.join(map(str, sorted(doc_lengths.keys())))}\n")
            # 写入文档长度信息
            f.write(f"doc_lengths:{json.dumps(doc_lengths)}\n")
            # 写入文档元数据
            f.write(f"doc_metadata:{json.dumps(doc_metadata)}\n")
            # 写入词典条目
            for term, (df, idf, _) in sorted(dictionary.items()):
                f.write(f"{term} {df} {idf} 0\n")  # offset将在写入倒排文件后更新
        
        # 写入倒排文件
        postings_file = os.path.join(index_dir, 'postings.bin')
        current_offset = 0
        with open(postings_file, 'wb') as f:
            for term in sorted(dictionary.keys()):
                # 更新词典中的offset
                dictionary[term] = (dictionary[term][0], dictionary[term][1], current_offset)
                
                # 写入倒排列表
                posting_list = sorted(postings[term])
                # 写入倒排列表长度（4字节）
                f.write(len(posting_list).to_bytes(4, byteorder='big'))
                current_offset += 4
                
                for doc_id, tf in posting_list:
                    # 确保 doc_id 是整数类型
                    doc_id_int = int(doc_id)
                    # 写入文档ID（4字节）和词频（4字节）
                    f.write(doc_id_int.to_bytes(4, byteorder='big'))
                    f.write(tf.to_bytes(4, byteorder='big'))
                    current_offset += 8
        
        # 更新词典文件中的offset
        with open(dict_file, 'w', encoding='utf-8') as f:
            # 写入所有已索引的文档ID
            f.write(f"all_indexed_docIDs:{','.join(map(str, sorted(doc_lengths.keys())))}\n")
            # 写入文档长度信息
            f.write(f"doc_lengths:{json.dumps(doc_lengths)}\n")
            # 写入文档元数据
            f.write(f"doc_metadata:{json.dumps(doc_metadata)}\n")
            # 写入词典条目
            for term in sorted(dictionary.keys()):
                df, idf, offset = dictionary[term]
                f.write(f"{term} {df} {idf} {offset}\n")
        
        end_time = time.time()
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 倒排索引构建完成！")
        print(f"总耗时: {end_time - start_time:.2f} 秒")
        print(f"索引文件保存在: {index_dir}")

if __name__ == "__main__":
    build_inverted_index() 