#!/usr/bin/python
import re
import sys
import getopt
import codecs
import struct
import math
import io
import os
import collections
import json
import jieba

# 常量定义
BYTE_SIZE = 4               # 文档ID使用整型存储，占用4字节
IGNORE_STOPWORDS = True     # 是否忽略停用词
IGNORE_NUMBERS = True       # 是否忽略数字
IGNORE_SINGLES = True       # 是否忽略单字符词项

# 字段类型定义
FIELD_TITLE = "title"      # 标题/主题
FIELD_AUTHOR = "author"    # 作者
FIELD_KEYWORD = "keyword"  # 关键词
FIELD_TIME = "time"        # 时间
FIELD_CONTENT = "content"  # 普通内容

"""加载停用词表"""
def load_stopwords(stopwords_file="IRoverview\stopwords_cn.txt"):
    stopwords = set()
    
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                stopwords.add(word)
    print(f"成功从{stopwords_file}加载{len(stopwords)}个停用词")
    return stopwords


"""
构建增强的倒排索引，从document_dir读取文档，生成词典文件和倒排文件
支持按字段索引
"""
def build_index(document_dir, dictionary_file, postings_file):
    # 打开输出文件
    dict_file = codecs.open(dictionary_file, 'w', encoding='utf-8')
    post_file = io.open(postings_file, 'wb')
    
    # 加载停用词表
    stopwords = load_stopwords()
    
    # 获取文档列表并排序
    docIDs = sorted([int(filename) for filename in os.listdir(document_dir) if filename.isdigit()])
    
    # 创建词典结构
    dictionary = {}  # 普通词项到(文档频率, 偏移量)的映射
    field_dictionaries = {} # 字段到词典的映射
    term_freq = {}   # 记录每个词项在每个文档中的频率: {term: {docID: 频率}}
    field_term_freq = {} # 字段词项频率: {field: {term: {docID: 频率}}}
    doc_lengths = {} # 文档长度: {docID: 词项总数}
    doc_metadata = {} # 文档元数据: {docID: {field: value}}
    
    # 初始化字段词典
    for field in [FIELD_TITLE, FIELD_AUTHOR, FIELD_KEYWORD, FIELD_TIME, FIELD_CONTENT]:
        field_dictionaries[field] = {}
        field_term_freq[field] = {}
    
    # 处理所有文档，构建内存中的倒排索引
    for docID in docIDs:
        doc_lengths[docID] = 0  # 初始化文档长度
        doc_metadata[docID] = {}  # 初始化文档元数据
        
        with codecs.open(os.path.join(document_dir, str(docID)), 'r', encoding='utf-8') as doc_file:
            content = doc_file.read()
            
            # 尝试解析JSON格式文档以获取元数据和结构化内容
            try:
                doc_data = json.loads(content)
                # 提取元数据
                if isinstance(doc_data, dict):
                    # 处理标题
                    if 'title' in doc_data:
                        doc_metadata[docID][FIELD_TITLE] = doc_data['title']
                        process_field_content(FIELD_TITLE, doc_data['title'], docID, 
                                            field_dictionaries, field_term_freq, stopwords)
                    
                    # 处理作者
                    if 'author' in doc_data:
                        doc_metadata[docID][FIELD_AUTHOR] = doc_data['author']
                        process_field_content(FIELD_AUTHOR, doc_data['author'], docID, 
                                            field_dictionaries, field_term_freq, stopwords)
                    
                    # 处理关键词
                    if 'keywords' in doc_data:
                        keywords = doc_data['keywords']
                        if isinstance(keywords, list):
                            keywords = ' '.join(keywords)
                        doc_metadata[docID][FIELD_KEYWORD] = keywords
                        process_field_content(FIELD_KEYWORD, keywords, docID, 
                                            field_dictionaries, field_term_freq, stopwords)
                    
                    # 处理时间
                    if 'time' in doc_data or 'date' in doc_data:
                        time_value = doc_data.get('time', doc_data.get('date', ''))
                        doc_metadata[docID][FIELD_TIME] = time_value
                    
                    # 处理内容
                    if 'content' in doc_data:
                        content_text = doc_data['content']
                        doc_metadata[docID][FIELD_CONTENT] = content_text
                        process_field_content(FIELD_CONTENT, content_text, docID, 
                                            field_dictionaries, field_term_freq, stopwords)
                        # 同时处理为普通内容
                        process_content(content_text, docID, dictionary, term_freq, doc_lengths, stopwords)
                else:
                    # 如果不是JSON格式，作为普通内容处理
                    process_content(content, docID, dictionary, term_freq, doc_lengths, stopwords)
            except json.JSONDecodeError:
                # 非JSON格式，作为普通内容处理
                process_content(content, docID, dictionary, term_freq, doc_lengths, stopwords)
    
    # 写入所有索引的文档ID到词典文件第一行
    dict_file.write(f"all_indexed_docIDs: {','.join(map(str, docIDs))}\n")
    
    # 写入文档长度信息到词典文件的第二行
    dict_file.write(f"doc_lengths: {json.dumps(doc_lengths)}\n")
    
    # 写入文档元数据到词典文件的第三行
    dict_file.write(f"doc_metadata: {json.dumps(doc_metadata)}\n")
    
    # 将内存中的倒排索引写入文件
    current_offset = 0  # 当前在倒排文件中的偏移量
    
    # 首先写入普通词项
    for term in sorted(dictionary.keys()):
        # 获取包含该词项的文档列表并排序
        postings_list = sorted(list(dictionary[term]))
        df = len(postings_list)  # 文档频率
        
        # 计算该词项的IDF值
        n_docs = len(docIDs)
        idf = math.log10(n_docs / df) if df > 0 and n_docs > df else 0
        
        # 将词项、文档频率、IDF和偏移量写入词典文件
        dict_file.write(f"{term} {df} {idf} {current_offset}\n")
        
        # 将倒排列表写入倒排文件
        for docID in postings_list:
            # 写入文档ID和词频
            post_file.write(struct.pack('I', docID))
            post_file.write(struct.pack('I', term_freq[term][docID]))
        
        # 更新偏移量（每条记录占8字节：4字节docID + 4字节词频）
        current_offset += df * (BYTE_SIZE * 2)
    
    # 然后写入字段词项
    for field in [FIELD_TITLE, FIELD_AUTHOR, FIELD_KEYWORD, FIELD_CONTENT]:
        for term in sorted(field_dictionaries[field].keys()):
            # 获取包含该字段词项的文档列表并排序
            postings_list = sorted(list(field_dictionaries[field][term]))
            df = len(postings_list)  # 文档频率
            
            # 计算该词项的IDF值
            n_docs = len(docIDs)
            idf = math.log10(n_docs / df) if df > 0 and n_docs > df else 0
            
            # 将字段:词项、文档频率、IDF和偏移量写入词典文件
            dict_file.write(f"{field}:{term} {df} {idf} {current_offset}\n")
            
            # 将倒排列表写入倒排文件
            for docID in postings_list:
                # 写入文档ID和词频
                post_file.write(struct.pack('I', docID))
                post_file.write(struct.pack('I', field_term_freq[field][term][docID]))
            
            # 更新偏移量（每条记录占8字节：4字节docID + 4字节词频）
            current_offset += df * (BYTE_SIZE * 2)
    
    # 关闭文件
    dict_file.close()
    post_file.close()

"""处理文档的通用内容，更新词典和文档长度"""
def process_content(content, docID, dictionary, term_freq, doc_lengths, stopwords):
    # 使用结巴分词
    tokens = list(jieba.cut(content))
    
    # 处理每个词项
    for token in tokens:
        term = token.lower().strip()  # 全部转为小写并去除空白
        
        # 过滤条件
        if not term:  # 跳过空字符串
            continue
        if IGNORE_STOPWORDS and term in stopwords:
            continue
        if IGNORE_NUMBERS and is_number(term):
            continue
        if IGNORE_SINGLES and len(term) <= 1:
            continue
        
        # 移除标点符号
        term = re.sub(r'[^\w\s]', '', term)
        if not term:  # 再次检查是否为空
            continue
        
        # 更新词典和词频
        if term not in dictionary:
            dictionary[term] = set()
            term_freq[term] = {}
        
        dictionary[term].add(docID)
        
        # 更新词频统计
        if docID not in term_freq[term]:
            term_freq[term][docID] = 0
        term_freq[term][docID] += 1
        
        # 更新文档长度
        doc_lengths[docID] += 1

"""处理文档的特定字段内容，更新字段词典"""
def process_field_content(field, content, docID, field_dictionaries, field_term_freq, stopwords):
    if not content:
        return
        
    # 使用结巴分词
    tokens = list(jieba.cut(str(content)))
    
    # 处理每个词项
    for token in tokens:
        term = token.lower().strip()  # 全部转为小写并去除空白
        
        # 过滤条件
        if not term:  # 跳过空字符串
            continue
        if IGNORE_STOPWORDS and term in stopwords:
            continue
        if IGNORE_NUMBERS and is_number(term):
            continue
        if IGNORE_SINGLES and len(term) <= 1:
            continue
        
        # 移除标点符号
        term = re.sub(r'[^\w\s]', '', term)
        if not term:  # 再次检查是否为空
            continue
        
        # 更新词典和词频
        if term not in field_dictionaries[field]:
            field_dictionaries[field][term] = set()
            field_term_freq[field][term] = {}
        
        field_dictionaries[field][term].add(docID)
        
        # 更新词频统计
        if docID not in field_term_freq[field][term]:
            field_term_freq[field][term][docID] = 0
        field_term_freq[field][term][docID] += 1

"""判断词项是否为数字"""
def is_number(token):
    token = token.replace(",", "")
    try:
        float(token)
        return True
    except ValueError:
        return False

"""显示正确的命令用法"""
def print_usage():
    print("用法: " + sys.argv[0] + " -i 文档目录 -d 词典文件 -p 倒排文件")

# 命令行接口
if __name__ == "__main__":
    document_dir = dictionary_file = postings_file = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)
        
    for o, a in opts:
        if o == '-i':
            document_dir = a
        elif o == '-d':
            dictionary_file = a
        elif o == '-p':
            postings_file = a
        else:
            assert False, "未处理的选项"
            
    if document_dir == None or dictionary_file == None or postings_file == None:
        print_usage()
        sys.exit(2)

    build_index(document_dir, dictionary_file, postings_file) 