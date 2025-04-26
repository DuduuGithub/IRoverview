# 表设计 留了一个时间表的示例，文书展示视图可以在这次项目中直接套用，主要是包含了一些文献的基本信息，比如作者、标题、关键词等。
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Database.config import db  # 导入 Database/config.py 中的 db 实例


# 用户文书表：uid,编号,标记、笔记等

# 日志表 记录对文书数据表的更改日志

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String, Text, Date, Enum, TIMESTAMP, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from Database.config import db
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
# 文书展示视图模型
class DocumentDisplayView(db.Model):
    """
    文书展示视图
    用于在页面中展示文书的基本信息，同时支持搜索功能
    """
    __tablename__ = 'DocumentDisplayView'
    __table_args__ = {'info': {'is_view': True}}
    
    Doc_id = Column(String(20), primary_key=True)
    Doc_title = Column(String(255))         # 文书标题
    Doc_type = Column(Enum('借钱契', '租赁契', '抵押契','赋税契','诉状','判决书','祭祀契约','祠堂契','劳役契','其他'))  # 文书类型
    Doc_summary = Column(Text)              # 文书大意
    Doc_image_path = Column(String(255))         # 文书图片路径
    Doc_time = Column(String(50))           # 签约时间（原始格式）
    Doc_standardTime = Column(TIMESTAMP)    # 签约时间（公历格式）
    ContractorInfo = Column(Text)           # 格式：《张三》《李四》（叔侄）
    ParticipantInfo = Column(Text)          # 格式：《王五》（见证人）《赵六》（代书）

    def __repr__(self):
        return f'<DocumentDisplay {self.Doc_title}>'
    
    def to_dict(self):
        """转换为字典格式，方便JSON序列化"""
        return {
            'doc_id': self.Doc_id,
            'title': self.Doc_title,
            'type': self.Doc_type,
            'time': self.Doc_time,
            'image': self.Doc_image,
            'summary': self.Doc_summary[:200] + '...' if self.Doc_summary and len(self.Doc_summary) > 200 else self.Doc_summary,
            'contractors': self.ContractorInfo,
            'participants': self.ParticipantInfo
        }
