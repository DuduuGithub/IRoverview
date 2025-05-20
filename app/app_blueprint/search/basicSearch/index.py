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
# import jieba  # 移除jieba导入

# 添加英文分词工具
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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
    # 只保留字母、数字和空格
    text = re.sub(r'[^\w\s]', ' ', text)
    # 分词
    tokens = word_tokenize(text)
    # 移除停用词、单个字符的词，并进行词干提取
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and len(word) > 1]
    return tokens

# 配置参数
BYTE_SIZE = 4               # 文档ID使用整型存储，占用4字节
IGNORE_STOPWORDS = True     # 是否忽略停用词
IGNORE_NUMBERS = True       # 是否忽略数字
IGNORE_SINGLES = True       # 是否忽略单字符词项

"""
加载停用词表
参数:
    - stopwords_file: 停用词文件路径
返回:
    - 停用词集合
"""
def load_stopwords(stopwords_file=None):
    """加载英文停用词"""
    # 使用NLTK的英文停用词列表
    try:
        nltk_stopwords = set(stopwords.words('english'))
        print(f"成功加载NLTK的{len(nltk_stopwords)}个英文停用词")
        return nltk_stopwords
    except:
        # 如果NLTK停用词加载失败，尝试从文件加载
        if stopwords_file:
            stopwords = set()
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        stopwords.add(word)
            print(f"成功从{stopwords_file}加载{len(stopwords)}个停用词")
            return stopwords
        else:
            print("警告：无法加载停用词")
            return set()

"""
构建倒排索引，从document_dir读取文档，生成词典文件和倒排文件
参数:
    - document_dir: 文档目录
    - dictionary_file: 输出的词典文件
    - postings_file: 输出的倒排索引文件
"""
def build_index(document_dir, dictionary_file, postings_file):
    print(f"开始构建索引，文档目录: {document_dir}")
    start_time = timeit.default_timer()
    
    # 打开输出文件
    dict_file = codecs.open(dictionary_file, 'w', encoding='utf-8')
    post_file = io.open(postings_file, 'wb')
    
    # 加载停用词表
    stopwords = load_stopwords()
    
    # 获取文档列表并排序
    docIDs = sorted([int(filename) for filename in os.listdir(document_dir) if filename.isdigit()])
    print(f"发现{len(docIDs)}个文档")
    
    # 初始化索引数据结构
    dictionary = {}  # 词项到文档ID集合的映射
    term_freq = {}   # 记录每个词项在每个文档中的频率
    doc_lengths = {} # 记录每个文档的词项总数
    
    # 第一步：处理所有文档，构建内存中的索引
    for docID in docIDs:
        doc_lengths[docID] = 0  # 初始化文档长度
        
        # 读取文档内容
        with codecs.open(os.path.join(document_dir, str(docID)), 'r', encoding='utf-8') as doc_file:
            content = doc_file.read()
            tokens = tokenize_english(content)  # 使用英文分词
            
            # 处理每个词项
            for term in tokens:
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
    
    # 第二步：写入元数据到词典文件
    # 写入所有索引的文档ID
    dict_file.write(f"all_indexed_docIDs: {','.join(map(str, docIDs))}\n")
    
    # 写入文档长度信息
    dict_file.write(f"doc_lengths: {json.dumps(doc_lengths)}\n")
    
    # 第三步：将内存中的倒排索引写入文件
    current_offset = 0  # 当前在倒排文件中的偏移量
    term_count = 0      # 已处理的词项数
    
    # 按词项字母顺序排序以便二分查找
    sorted_terms = sorted(dictionary.keys())
    
    for term in sorted_terms:
        term_count += 1
        if term_count % 1000 == 0:
            print(f"处理词项进度: {term_count}/{len(sorted_terms)}")
            
        # 获取包含该词项的文档列表并排序
        postings_list = sorted(list(dictionary[term]))
        df = len(postings_list)  # 文档频率
        
        # 计算该词项的IDF值
        idf = math.log10(len(docIDs) / df) if df > 0 else 0
        
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
    
    end_time = timeit.default_timer()
    
    print(f"索引构建完成:")
    print(f"- 文档数: {len(docIDs)}")
    print(f"- 词项数: {len(dictionary)}")
    print(f"- 词典文件: {dictionary_file}")
    print(f"- 倒排文件: {postings_file}")
    print(f"- 耗时: {end_time - start_time:.2f}秒")

"""
判断词项是否为数字
参数:
    - token: 要检查的词项
返回:
    - 布尔值，表示词项是否为数字
"""
def is_number(token):
    # 移除数字中的逗号
    token = token.replace(",", "")
    try:
        float(token)
        return True
    except ValueError:
        return False

"""
显示正确的命令用法
"""
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