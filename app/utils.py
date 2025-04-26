# 连接数据库的基本功能
import sys
import os

from flask import jsonify
from sqlalchemy import text
# 将项目根目录添加到 sys.path,Python默认从当前文件所在的目录开始找，也就是app文件夹开始找
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Database.config import db
from Database.model import *
from sqlalchemy import or_
from flask import session
from flask_login import current_user
import json
from yearToyearTools.run import convert_to_gregorian



#全文搜索
def db_context_query(query, doc_type=None, date_from=None, date_to=None):
    """
    实现全文检索，查询文书标题或原文中包含关键字的记录，并支持高级搜索。
    支持部分匹配和多个关键词。
    """
    try:
        # 处理搜索关键词，添加通配符和 + 操作符
        search_terms = query.split()
        formatted_query = ' '.join([f'+*{term}*' for term in search_terms])
        
        # 构建基本的全文检索 SQL 查询
        sql = """
            SELECT Doc_id FROM Documents
            WHERE MATCH(Doc_title, Doc_simplifiedText, Doc_originalText) 
            AGAINST(:query IN BOOLEAN MODE)
            OR Doc_title LIKE :like_query
            OR Doc_simplifiedText LIKE :like_query
            OR Doc_originalText LIKE :like_query
        """
        params = {
            'query': formatted_query,
            'like_query': f'%{query}%'  # 添加 LIKE 查询作为备选
        }
        
        # 添加文档类型筛选条件（如果提供）
        if doc_type:
            sql += " AND Doc_type = :doc_type"
            params['doc_type'] = doc_type
        
        # 添加日期范围筛选条件（如果提供）
        if date_from:
            sql += " AND Doc_createdAt >= :date_from"
            params['date_from'] = date_from
        if date_to:
            sql += " AND Doc_createdAt <= :date_to"
            params['date_to'] = date_to
        
        # 添加排序条件
        sql += """ 
            ORDER BY MATCH(Doc_title, Doc_simplifiedText, Doc_originalText) 
            AGAINST(:query IN BOOLEAN MODE) DESC
        """
        
        print(f"Executing search with query: {formatted_query}")
        print(f"SQL: {sql}")
        print(f"Params: {params}")
        
        # 执行查询获取文档ID
        result = db.session.execute(text(sql), params)
        doc_ids = [row[0] for row in result.fetchall()]
        
        if not doc_ids:
            return []
            
        # 使用找到的文档ID从 DocumentDisplayView 中获取完整信息
        display_results = DocumentDisplayView.query.filter(
            DocumentDisplayView.Doc_id.in_(doc_ids)
        ).all()
        
        return display_results
        
    except Exception as e:
        print(f"全文搜索出错: {str(e)}")
        return []



def get_document_info(text: str):
    """调用星火大模型API获取文书信息"""
    try:
        # 配置星火认知大模型
        spark = ChatSparkLLM(
            spark_api_url='wss://spark-api.xf-yun.com/v4.0/chat',
            spark_app_id='ce9ffe63',
            spark_api_key='37a6c3241c800fc455c445176efddc0d',
            spark_api_secret='ODhjMzViODcwODU4Njk0ZTgxZWI1ZGVh',
            spark_llm_domain='4.0Ultra',
            streaming=False
        )
        
        # 构建提示词
        prompt_base = """分析下面这份清代契约文书，提取以下信息：
        1. 文书标题（根据内容生成一个合适的标题）
        2. 文书类型（借钱契、租赁契、抵押契、赋税契、诉状、判决书、祭祀契约、祠堂契、劳役契、其他）
        3. 文书大意（200字以内）
        4. 签订时间（从文书中提取具体的时间）
        5. 更改时间（如果有）
        6. 关键词（3-5个）
        7. 契约双方（两个人的姓名）
        8. 契约双方关系（如叔侄、父子等，如果有）
        9. 参与人及其身份（如见证人、代书等，可能有多人）
        10.对文书内容进行断句，如果有字体仍为繁体则将其转换为简体，返回断句且再次简体化的文书内容

        请用JSON格式返回结果，格式如下：
        {
            "title": "文书标题",
            "type": "文书类型",
            "summary": "文书大意",
            "created_time": "签订时间",
            "updated_time": "更��时间（如果有则填写，没有则为null）",
            "keywords": ["关键词1", "关键词2", "关键词3"],
            "contractors": [
                {"name": "第一个契约人姓名"},
                {"name": "第二个契约人姓名"}
            ],
            "relation": "契约双方关系（如果有则填写，没有则为null）",
            "participants": [
                {"name": "参与人1姓名", "role": "参与人1身份"},
                {"name": "参与人2姓名", "role": "参与人2身份"}
            ],
            "simple_text": "文书内容断句且简体化"
        }

        请严格按照上述JSON格式返回结果。以下是文书内容：
        """
        
        # 创建消息
        messages = [ChatMessage(role="user", content=prompt_base+text)]
        
        # 调用API
        handler = ChunkPrintHandler()
        response = spark.generate([messages], callbacks=[handler])
        print("问答完成\n")
        
        # 获取并清理JSON字符串
        if isinstance(response.generations[0], list):
            json_str = response.generations[0][0].text
        else:
            json_str = response.generations[0].text
            
        json_str = json_str.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()
        
        # 解析返回的JSON
        try:
            result = json.loads(json_str)
            # 验证返回的数据是否包含所有必要字段
            required_fields = ['title', 'type', 'summary', 'created_time', 'keywords', 
                             'contractors', 'participants', 'simple_text']
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"API返回的数据缺少必要字段: {field}")
            return result
        except Exception as e:
            print(f"解析文书信息失败: {e}")
            print(f"尝试解析的字符串: {json_str}")
            raise ValueError(f"解析文书信息失败: {str(e)}")
            
    except Exception as e:
        print(f"调用星火API失败: {e}")
        raise


    

