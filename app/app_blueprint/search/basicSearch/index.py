#!/usr/bin/python
import re
import nltk
import sys
import getopt
import codecs
import struct
import math
import io
import os
import collections
import timeit
import json
import jieba

# 配置参数
BYTE_SIZE = 4               # 文档ID使用整型存储，占用4字节
IGNORE_STOPWORDS = True     # 是否忽略停用词
IGNORE_NUMBERS = True       # 是否忽略数字
IGNORE_SINGLES = True       # 是否忽略单字符词项

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
构建倒排索引，从document_dir读取文档，生成词典文件和倒排文件
"""
def build_index(document_dir, dictionary_file, postings_file):
    # 打开输出文件
    dict_file = codecs.open(dictionary_file, 'w', encoding='utf-8')
    post_file = io.open(postings_file, 'wb')
    
    # 加载停用词表
    stopwords = load_stopwords()
    
    # 获取文档列表并排序
    docIDs = sorted([int(filename) for filename in os.listdir(document_dir) if filename.isdigit()])
    dictionary = {}  # 词项到(文档频率, 偏移量)的映射
    term_freq = {}   # 记录每个词项在每个文档中的频率: {term: {docID: 频率}}
    doc_lengths = {} # 文档长度: {docID: 词项总数}
    
    # 处理所有文档，构建内存中的倒排索引
    for docID in docIDs:
        doc_lengths[docID] = 0  # 初始化文档长度
        with codecs.open(os.path.join(document_dir, str(docID)), 'r', encoding='utf-8') as doc_file:
            # 读取文档并分词
            content = doc_file.read()
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
    
    # 写入所有索引的文档ID到词典文件第一行
    dict_file.write(f"all_indexed_docIDs: {','.join(map(str, docIDs))}\n")
    
    # 写入文档长度信息到词典文件的第二行
    dict_file.write(f"doc_lengths: {json.dumps(doc_lengths)}\n")
    
    # 将内存中的倒排索引写入文件
    current_offset = 0  # 当前在倒排文件中的偏移量
    
    # 按词项字母顺序排序
    for term in sorted(dictionary.keys()):
        # 获取包含该词项的文档列表并排序
        postings_list = sorted(list(dictionary[term]))
        df = len(postings_list)  # 文档频率
        
        # 计算该词项的IDF值
        idf = math.log10(len(docIDs) / df) if df < len(docIDs) else 0
        
        # 将词项、文档频率、IDF和偏移量写入词典文件
        dict_file.write(f"{term} {df} {idf} {current_offset}\n")
        
        # 将倒排列表写入倒排文件
        for docID in postings_list:
            # 写入文档ID和词频
            post_file.write(struct.pack('I', docID))
            post_file.write(struct.pack('I', term_freq[term][docID]))
        
        # 更新偏移量（每条记录占8字节：4字节docID + 4字节词频）
        current_offset += df * (BYTE_SIZE * 2)
    
    # 关闭文件
    dict_file.close()
    post_file.close()

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