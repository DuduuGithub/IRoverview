# 表设计 留了一个时间表的示例，文书展示视图可以在这次项目中直接套用，主要是包含了一些文献的基本信息，比如作者、标题、关键词等。
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Database.config import db  # 导入 Database/config.py 中的 db 实例
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Date, Enum, TIMESTAMP, ForeignKey, DateTime, Float
from datetime import datetime
import json
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String, Text, Date, Enum, TIMESTAMP, ForeignKey, DateTime, Boolean, Float, JSON, DECIMAL
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from flask_login import UserMixin

# 作者表
class Author(db.Model):
    __tablename__ = 'authors'
    
    id = Column(String(255), primary_key=True)
    orcid = Column(String(255))
    display_name = Column(String(255), nullable=False)
    display_name_alternatives = Column(JSON)
    works_count = Column(Integer, default=0)
    cited_by_count = Column(Integer, default=0)
    last_known_institution = Column(String(255))
    works_api_url = Column(String(255))
    updated_date = Column(DateTime)

# 概念表
class Concept(db.Model):
    __tablename__ = 'concepts'
    
    id = Column(String(255), primary_key=True)
    wikidata = Column(String(255))
    display_name = Column(String(255), nullable=False)
    level = Column(Integer)
    description = Column(Text)
    works_count = Column(Integer, default=0)
    cited_by_count = Column(Integer, default=0)
    image_url = Column(String(255))
    image_thumbnail_url = Column(String(255))
    works_api_url = Column(String(255))
    updated_date = Column(Date)

# 机构表
class Institution(db.Model):
    __tablename__ = 'institutions'
    
    id = Column(String(255), primary_key=True)
    ror = Column(String(255))
    display_name = Column(String(255))
    country_code = Column(String(2))
    type = Column(String(255))
    homepage_url = Column(String(255))
    image_url = Column(String(255))
    image_thumbnail_url = Column(String(255))
    display_name_acronyms = Column(JSON)
    display_name_alternatives = Column(JSON)
    works_count = Column(Integer, default=0)
    cited_by_count = Column(Integer, default=0)
    works_api_url = Column(String(255))
    updated_date = Column(DateTime)

# 来源表
class Source(db.Model):
    __tablename__ = 'sources'
    
    id = Column(String(255), primary_key=True)
    issn_l = Column(String(255))
    issn = Column(JSON)
    display_name = Column(String(255))
    works_count = Column(Integer, default=0)
    cited_by_count = Column(Integer, default=0)
    is_oa = Column(Boolean, default=False)
    is_in_doaj = Column(Boolean, default=False)
    homepage_url = Column(String(255))
    works_api_url = Column(String(255))
    updated_date = Column(DateTime)

# 主题表
class Topic(db.Model):
    __tablename__ = 'topics'
    
    id = Column(String(255), primary_key=True)
    display_name = Column(String(255))
    subfield_id = Column(String(255))
    subfield_display_name = Column(String(255))
    field_id = Column(String(255))
    field_display_name = Column(String(255))
    domain_id = Column(String(255))
    domain_display_name = Column(String(255))
    description = Column(Text)
    keywords = Column(Text)
    works_api_url = Column(String(255))
    wikipedia_id = Column(String(255))
    works_count = Column(Integer, default=0)
    cited_by_count = Column(Integer, default=0)
    updated_date = Column(DateTime)

# 作品表
class Work(db.Model):
    __tablename__ = 'works'
    
    id = Column(String(255), primary_key=True)
    doi = Column(String(255))
    title = Column(Text)
    display_name = Column(Text)
    publication_year = Column(Integer)
    publication_date = Column(Date)
    type = Column(String(50))
    cited_by_count = Column(Integer, default=0)
    is_retracted = Column(Boolean, default=False)
    is_paratext = Column(Boolean, default=False)
    cited_by_api_url = Column(String(255))
    abstract_inverted_index = Column(JSON)
    language = Column(String(10))
    openalex = Column(String(255))
    mag = Column(String(255))
    pmid = Column(String(255))
    pmcid = Column(String(255))
    volume = Column(String(50))
    issue = Column(String(50))
    first_page = Column(String(50))
    last_page = Column(String(50))

# 作者ID映射表
class AuthorId(db.Model):
    __tablename__ = 'authors_ids'
    
    author_id = Column(String(255), primary_key=True)
    openalex = Column(String(255))
    orcid = Column(String(255))
    scopus = Column(String(255))
    twitter = Column(String(255))
    wikipedia = Column(String(255))
    mag = Column(String(255))

# 概念ID映射表
class ConceptId(db.Model):
    __tablename__ = 'concepts_ids'
    
    concept_id = Column(String(255), primary_key=True)
    openalex = Column(String(255))
    wikidata = Column(String(255))
    wikipedia = Column(String(255))
    umls_aui = Column(String(255))
    umls_cui = Column(String(255))
    mag = Column(String(255))

# 概念层级关系表
class ConceptAncestor(db.Model):
    __tablename__ = 'concepts_ancestors'
    
    concept_id = Column(String(255), primary_key=True)
    ancestor_id = Column(String(255), primary_key=True)

# 概念相关性关系表
class ConceptRelatedConcept(db.Model):
    __tablename__ = 'concepts_related_concepts'
    
    concept_id = Column(String(255), primary_key=True)
    related_concept_id = Column(String(255), primary_key=True)
    score = Column(Float)

# 机构ID映射表
class InstitutionId(db.Model):
    __tablename__ = 'institutions_ids'
    
    institution_id = Column(String(255), primary_key=True)
    openalex = Column(String(255))
    ror = Column(String(255))
    grid = Column(String(255))
    wikipedia = Column(String(255))
    wikidata = Column(String(255))
    mag = Column(String(255))

# 机构地理位置表
class InstitutionGeo(db.Model):
    __tablename__ = 'institutions_geo'
    
    institution_id = Column(String(255), primary_key=True)
    city = Column(String(255))
    geonames_city_id = Column(String(255))
    region = Column(String(255))
    country_code = Column(String(2))
    country = Column(String(255))
    latitude = Column(DECIMAL(10,6))
    longitude = Column(DECIMAL(10,6))

# 机构关联关系表
class InstitutionAssociatedInstitution(db.Model):
    __tablename__ = 'institutions_associated_institutions'
    
    institution_id = Column(String(255), primary_key=True)
    associated_institution_id = Column(String(255), primary_key=True)
    relationship = Column(String(255))

# 来源ID映射表
class SourceId(db.Model):
    __tablename__ = 'sources_ids'
    
    source_id = Column(String(255), primary_key=True)
    openalex = Column(String(255))
    issn_l = Column(String(255))
    issn = Column(JSON)
    mag = Column(String(255))
    wikidata = Column(String(255))
    fatcat = Column(String(255))

# 作品-主题关联表
class WorkTopic(db.Model):
    __tablename__ = 'works_topics'
    
    work_id = Column(String(255), primary_key=True)
    topic_id = Column(String(255), primary_key=True)
    score = Column(Float)

# 作品-相关作品关联表
class WorkRelatedWork(db.Model):
    __tablename__ = 'works_related_works'
    
    work_id = Column(String(255), primary_key=True)
    related_work_id = Column(String(255), primary_key=True)

# 作品-引用作品关联表
class WorkReferencedWork(db.Model):
    __tablename__ = 'works_referenced_works'
    
    work_id = Column(String(255), primary_key=True)
    referenced_work_id = Column(String(255), primary_key=True)

# 作品-位置关联表
class WorkLocation(db.Model):
    __tablename__ = 'works_locations'
    
    work_id = Column(String(255), primary_key=True)
    source_id = Column(String(255), primary_key=True)
    location_type = Column(Enum('primary', 'best_oa', 'other'))
    landing_page_url = Column(String(255))
    pdf_url = Column(String(255))
    is_oa = Column(Boolean, default=False)
    oa_status = Column(Enum('gold', 'bronze', 'green', 'hybrid', 'closed'))
    oa_url = Column(String(255))
    any_repository_has_fulltext = Column(Boolean, default=False)
    version = Column(String(50))
    license = Column(String(255))

# 作品-MeSH主题关联表
class WorkMesh(db.Model):
    __tablename__ = 'works_mesh'
    
    work_id = Column(String(255), primary_key=True)
    descriptor_ui = Column(String(255), primary_key=True)
    descriptor_name = Column(String(255))
    qualifier_ui = Column(String(255), primary_key=True)
    qualifier_name = Column(String(255))
    is_major_topic = Column(Boolean, default=False)

# 作品-概念关联表
class WorkConcept(db.Model):
    __tablename__ = 'works_concepts'
    
    work_id = Column(String(255), primary_key=True)
    concept_id = Column(String(255), primary_key=True)
    score = Column(Float)

# 作品-作者署名表
class WorkAuthorship(db.Model):
    __tablename__ = 'works_authorships'
    
    work_id = Column(String(255), primary_key=True)
    author_position = Column(String(50), primary_key=True)
    author_id = Column(String(255))
    institution_id = Column(String(255))
    raw_affiliation_string = Column(Text)

# 年度统计表
class YearlyStat(db.Model):
    __tablename__ = 'yearly_stats'
    
    entity_id = Column(String(255), primary_key=True)
    entity_type = Column(Enum('author', 'concept', 'institution', 'source'), primary_key=True)
    year = Column(Integer, primary_key=True)
    works_count = Column(Integer, default=0)
    cited_by_count = Column(Integer, default=0)
    oa_works_count = Column(Integer, default=0)

# 操作记录表
class OperationLog(db.Model):
    __tablename__ = 'operation_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    operation_type = Column(Enum('import', 'update', 'delete', 'export'))
    entity_type = Column(Enum('author', 'concept', 'institution', 'source', 'work', 'topic'))
    entity_id = Column(String(255))
    operation_time = Column(DateTime, default=datetime.utcnow)
    operator = Column(String(50))
    operation_status = Column(Enum('success', 'failed'))
    operation_details = Column(Text)
    error_message = Column(Text)
    affected_rows = Column(Integer)

# 数据导入类
class DataImporter:
    def __init__(self, db):
        self.db = db
        self.logger = OperationLog()
    
    def import_csv(self, file_path, table_name, skip_duplicates=True):
        """
        从CSV文件导入数据到指定表
        
        功能说明：
        1. 支持从CSV文件批量导入数据到指定的数据库表
        2. 自动处理重复数据（可配置是否跳过）
        3. 记录操作日志到operation_logs表
        4. 支持事务处理和错误回滚
        5. 自动处理特殊数据类型（如JSON）
        
        参数说明：
        file_path (str): CSV文件路径
        table_name (str): 目标表名，必须是已定义的模型表名之一
        skip_duplicates (bool): 是否跳过重复数据，默认为True
        
        使用要求：
        1. CSV文件要求：
           - 文件编码必须是UTF-8
           - 列名必须与数据库表的字段名完全匹配
           - 对于JSON类型的字段，CSV中的值必须是有效的JSON字符串
           - 对于枚举类型的字段，值必须是预定义的枚举值之一
        
        2. 环境要求：
           - 需要安装pandas库：pip install pandas
           - 需要足够的数据库连接权限
           - 需要足够的磁盘空间
        
        3. 数据量限制：
           - 建议单次导入不超过1000条记录
           - 如果数据量较大，建议分批导入
        
        错误处理：
        1. 如果发生错误：
           - 会自动回滚事务
           - 记录错误信息到operation_logs表
           - 抛出异常，包含详细的错误信息
        
        2. 可能出现的错误：
           - 文件不存在或无法读取
           - 文件格式不正确
           - 数据类型不匹配
           - 违反唯一约束
           - 数据库连接错误
        
        使用示例：
        ```python
        importer = DataImporter(db)
        try:
            importer.import_csv('path/to/your/file.csv', 'authors', skip_duplicates=True)
        except Exception as e:
            print(f"导入失败: {str(e)}")
        ```
        
        Returns:
            None
        
        Raises:
            ValueError: 当表名不存在时
            FileNotFoundError: 当CSV文件不存在时
            IntegrityError: 当违反数据库约束时
            Exception: 其他可能的错误
        """
        try:
            import pandas as pd
            from sqlalchemy.exc import IntegrityError
            
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 获取对应的模型类
            model_class = self._get_model_class(table_name)
            if not model_class:
                raise ValueError(f"未知的表名: {table_name}")
            
            # 记录开始导入
            self.logger.operation_type = 'import'
            self.logger.entity_type = table_name
            self.logger.operation_time = datetime.utcnow()
            self.logger.operator = 'system'
            self.logger.operation_status = 'success'
            self.logger.operation_details = f"开始导入文件: {file_path}"
            self.logger.affected_rows = 0
            
            # 开始事务
            with self.db.session.begin():
                # 遍历数据行
                for _, row in df.iterrows():
                    try:
                        # 创建模型实例
                        instance = model_class()
                        for column in model_class.__table__.columns:
                            if column.name in df.columns:
                                value = row[column.name]
                                # 处理特殊类型
                                if isinstance(column.type, (JSON,)):
                                    if pd.isna(value):
                                        value = None
                                    elif isinstance(value, str):
                                        try:
                                            value = json.loads(value)
                                        except:
                                            value = None
                                setattr(instance, column.name, value)
                        
                        # 检查是否存在重复数据
                        if skip_duplicates:
                            primary_keys = [key.name for key in model_class.__table__.primary_key]
                            filters = {key: getattr(instance, key) for key in primary_keys}
                            exists = self.db.session.query(model_class).filter_by(**filters).first()
                            if exists:
                                continue
                        
                        # 添加到会话
                        self.db.session.add(instance)
                        self.logger.affected_rows += 1
                        
                    except IntegrityError as e:
                        if skip_duplicates:
                            continue
                        raise e
                    except Exception as e:
                        self.logger.error_message = str(e)
                        raise e
                
                # 提交事务
                self.db.session.commit()
                
        except Exception as e:
            # 回滚事务
            self.db.session.rollback()
            self.logger.operation_status = 'failed'
            self.logger.error_message = str(e)
            raise e
        finally:
            # 记录操作日志
            self.db.session.add(self.logger)
            self.db.session.commit()
    
    def _get_model_class(self, table_name):
        """根据表名获取对应的模型类"""
        model_map = {
            'authors': Author,
            'concepts': Concept,
            'institutions': Institution,
            'sources': Source,
            'topics': Topic,
            'works': Work,
            'authors_ids': AuthorId,
            'concepts_ids': ConceptId,
            'concepts_ancestors': ConceptAncestor,
            'concepts_related_concepts': ConceptRelatedConcept,
            'institutions_ids': InstitutionId,
            'institutions_geo': InstitutionGeo,
            'institutions_associated_institutions': InstitutionAssociatedInstitution,
            'sources_ids': SourceId,
            'works_topics': WorkTopic,
            'works_related_works': WorkRelatedWork,
            'works_referenced_works': WorkReferencedWork,
            'works_locations': WorkLocation,
            'works_mesh': WorkMesh,
            'works_concepts': WorkConcept,
            'works_authorships': WorkAuthorship,
            'yearly_stats': YearlyStat
        }
        return model_map.get(table_name)

# 搜索会话记录
class SearchSession(db.Model):
    __tablename__ = 'search_sessions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(50), nullable=False, unique=True)
    search_time = Column(DateTime, default=datetime.utcnow)
    keyword = Column(String(255))
    title_query = Column(String(255))
    author_query = Column(String(255))
    concept_query = Column(String(255))
    institution_query = Column(String(255))
    date_from = Column(DateTime)
    date_to = Column(DateTime)
    search_type = Column(Enum('keyword', 'semantic', 'advanced', 'citation'))
    total_results = Column(Integer, default=0)
    search_filters = Column(JSON)
    user_id = Column(String(50))
    client_info = Column(JSON)
    
    # 关系
    results = relationship('SearchResult', back_populates='session')
    
    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'search_time': self.search_time.isoformat() if self.search_time else None,
            'keyword': self.keyword,
            'title_query': self.title_query,
            'author_query': self.author_query,
            'concept_query': self.concept_query,
            'institution_query': self.institution_query,
            'date_from': self.date_from.isoformat() if self.date_from else None,
            'date_to': self.date_to.isoformat() if self.date_to else None,
            'search_type': self.search_type,
            'total_results': self.total_results,
            'search_filters': self.search_filters,
            'user_id': self.user_id,
            'client_info': self.client_info
        }

# 搜索结果记录
class SearchResult(db.Model):
    __tablename__ = 'search_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(50), ForeignKey('search_sessions.session_id'))
    entity_type = Column(Enum('work', 'author', 'concept', 'institution', 'source', 'topic'))
    entity_id = Column(String(255))
    rank_position = Column(Integer)
    relevance_score = Column(Float)
    is_clicked = Column(Boolean, default=False)
    click_time = Column(DateTime)
    click_order = Column(Integer)
    dwell_time = Column(Integer)
    user_feedback = Column(JSON)
    
    # 关系
    session = relationship('SearchSession', back_populates='results')
    
    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'entity_type': self.entity_type,
            'entity_id': self.entity_id,
            'rank_position': self.rank_position,
            'relevance_score': self.relevance_score,
            'is_clicked': self.is_clicked,
            'click_time': self.click_time.isoformat() if self.click_time else None,
            'click_order': self.click_order,
            'dwell_time': self.dwell_time,
            'user_feedback': self.user_feedback
        }