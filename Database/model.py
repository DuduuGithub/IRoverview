# 表设计 留了一个时间表的示例，文书展示视图可以在这次项目中直接套用，主要是包含了一些文献的基本信息，比如作者、标题、关键词等。
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Database.config import db  # 导入 Database/config.py 中的 db 实例
from datetime import datetime

# 用户文书表：uid,编号,标记、笔记等

# 日志表 记录对文书数据表的更改日志

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String, Text, Date, Enum, TIMESTAMP, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from flask_login import UserMixin

#1、时间表
class TimeRecord(db.Model):
    __tablename__ = 'TimeRecord'

    Time_id = Column(Integer, primary_key=True, autoincrement=True)
    createdData = Column(String(50), nullable=False)  # 原始时间（"康熙三年"）
    Standard_createdData = Column(TIMESTAMP, nullable=False)  # 标准化时间（公历时间）
    __table_args__ = (
        db.Index('idx_createdData', 'createdData'),
        db.Index('idx_Standard_createdData', 'Standard_createdData'),
        {'mysql_charset': 'utf8mb4', 'mysql_collate': 'utf8mb4_unicode_ci'}
    )

class Document(db.Model):
    """文献数据模型"""
    __tablename__ = 'documents'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    author = db.Column(db.String(255))
    publish_date = db.Column(db.DateTime)
    content = db.Column(db.Text)
    keywords = db.Column(db.String(255))
    
    # 定义引用关系
    citations = db.relationship(
        'Document',
        secondary='citation_network',
        primaryjoin='Document.id == citation_network.c.citing_doc_id',
        secondaryjoin='Document.id == citation_network.c.cited_doc_id',
        backref=db.backref('cited_by', lazy='dynamic')
    )

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'author': self.author,
            'publish_date': self.publish_date.strftime('%Y-%m-%d') if self.publish_date else None,
            'content': self.content,
            'keywords': self.keywords
        }

# 引用网络关系表
citation_network = db.Table('citation_network',
    db.Column('citing_doc_id', db.Integer, db.ForeignKey('documents.id'), primary_key=True),
    db.Column('cited_doc_id', db.Integer, db.ForeignKey('documents.id'), primary_key=True),
    db.Column('citation_date', db.DateTime, default=datetime.utcnow)
)

# 搜索会话记录
class SearchSession(db.Model):
    __tablename__ = 'search_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(50), nullable=False)  # 前端生成的会话ID
    search_time = db.Column(db.DateTime, default=datetime.utcnow)
    keyword = db.Column(db.String(255))  # 基本搜索关键词
    # 高级搜索字段
    title_query = db.Column(db.String(255))
    author_query = db.Column(db.String(255))
    date_from = db.Column(db.DateTime)
    date_to = db.Column(db.DateTime)
    search_type = db.Column(db.String(20))  # 'basic', 'advanced', 'query'
    total_results = db.Column(db.Integer)  # 搜索结果总数

    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'search_time': self.search_time.strftime('%Y-%m-%d %H:%M:%S'),
            'keyword': self.keyword,
            'title_query': self.title_query,
            'author_query': self.author_query,
            'date_from': self.date_from.strftime('%Y-%m-%d') if self.date_from else None,
            'date_to': self.date_to.strftime('%Y-%m-%d') if self.date_to else None,
            'search_type': self.search_type,
            'total_results': self.total_results
        }

# 搜索结果记录
class SearchResult(db.Model):
    __tablename__ = 'search_results'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(50), db.ForeignKey('search_sessions.session_id'))
    document_id = db.Column(db.Integer, db.ForeignKey('documents.id'))
    rank_position = db.Column(db.Integer)  # 在结果列表中的位置
    is_clicked = db.Column(db.Boolean, default=False)  # 是否被点击
    click_time = db.Column(db.DateTime)  # 点击时间
    click_order = db.Column(db.Integer)  # 在当前会话中的点击顺序
    dwell_time = db.Column(db.Integer)  # 停留时间（秒）

    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'document_id': self.document_id,
            'rank_position': self.rank_position,
            'is_clicked': self.is_clicked,
            'click_time': self.click_time.strftime('%Y-%m-%d %H:%M:%S') if self.click_time else None,
            'click_order': self.click_order,
            'dwell_time': self.dwell_time
        }