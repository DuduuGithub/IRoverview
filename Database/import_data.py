import os
import sys
import json
import pandas as pd
import logging

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 完全关闭SQLAlchemy的日志输出
logging.getLogger('sqlalchemy.engine').setLevel(logging.ERROR)
logging.getLogger('sqlalchemy.pool').setLevel(logging.ERROR)
logging.getLogger('sqlalchemy.dialects').setLevel(logging.ERROR)
logging.getLogger('sqlalchemy.orm').setLevel(logging.ERROR)

from Database.model import DataImporter, db
from app import create_app  # 导入create_app函数

# 导入所有数据库模型类
from Database.model import Author, Concept, Institution, Source, Topic, Work, AuthorId, ConceptId, ConceptAncestor, ConceptRelatedConcept, InstitutionId, InstitutionGeo, InstitutionAssociatedInstitution, SourceId, WorkTopic, WorkRelatedWork, WorkReferencedWork, WorkLocation, WorkMesh, WorkConcept, WorkAuthorship, YearlyStat, OperationLog

# 创建Flask应用实例
app = create_app()

# 定义CSV文件名到数据库表名的映射
csv_to_table_mapping = {
    'authors.csv': 'authors',
    'concepts.csv': 'concepts',
    'institutions.csv': 'institutions',
    'sources.csv': 'sources',
    'topics.csv': 'topics',
    'works.csv': 'works',
    'authors_ids.csv': 'authors_ids',
    'concepts_ids.csv': 'concepts_ids',
    'concepts_ancestors.csv': 'concepts_ancestors',
    'concepts_related_concepts.csv': 'concepts_related_concepts',
    'institutions_ids.csv': 'institutions_ids',
    'institutions_geo.csv': 'institutions_geo',
    'institutions_associated_institutions.csv': 'institutions_associated_institutions',
    'sources_ids.csv': 'sources_ids',
    'works_topics.csv': 'works_topics',
    'works_related_works.csv': 'works_related_works',
    'works_referenced_works.csv': 'works_referenced_works',
    'works_locations.csv': 'works_locations',
    'works_mesh.csv': 'works_mesh',
    'works_concepts.csv': 'works_concepts',
    'works_authorships.csv': 'works_authorships',
    'yearly_stats.csv': 'yearly_stats'
}

# 表名到实体类型（单数英文）映射
table_to_entity_type = {
    'authors': 'author',
    'concepts': 'concept',
    'institutions': 'institution',
    'sources': 'source',
    'topics': 'topic',
    'works': 'work',
    # 其他表可按需补充
}

# 定义字段类型映射
field_type_mapping = {
    'id': 'String',
    'orcid': 'String',
    'display_name': 'String',
    'display_name_alternatives': 'JSON',
    'works_count': 'Integer',
    'cited_by_count': 'Integer',
    'last_known_institution': 'String',
    'works_api_url': 'String',
    'updated_date': 'DateTime',
    'wikidata': 'String',
    'level': 'Integer',
    'description': 'Text',
    'image_url': 'String',
    'image_thumbnail_url': 'String',
    'country_code': 'String',
    'type': 'String',
    'homepage_url': 'String',
    'display_name_acronyms': 'JSON',
    'issn_l': 'String',
    'issn': 'JSON',
    'is_oa': 'Boolean',
    'is_in_doaj': 'Boolean',
    'subfield_id': 'String',
    'subfield_display_name': 'String',
    'field_id': 'String',
    'field_display_name': 'String',
    'domain_id': 'String',
    'domain_display_name': 'String',
    'keywords': 'Text',
    'wikipedia_id': 'String',
    'doi': 'String',
    'title': 'Text',
    'publication_year': 'Integer',
    'publication_date': 'Date',
    'is_retracted': 'Boolean',
    'is_paratext': 'Boolean',
    'cited_by_api_url': 'String',
    'abstract_inverted_index': 'JSON',
    'language': 'String',
    'openalex': 'String',
    'mag': 'String',
    'pmid': 'String',
    'pmcid': 'String',
    'volume': 'String',
    'issue': 'String',
    'first_page': 'String',
    'last_page': 'String',
    'author_id': 'String',
    'scopus': 'String',
    'twitter': 'String',
    'wikipedia': 'String',
    'concept_id': 'String',
    'umls_aui': 'String',
    'umls_cui': 'String',
    'ancestor_id': 'String',
    'related_concept_id': 'String',
    'score': 'Float',
    'institution_id': 'String',
    'ror': 'String',
    'grid': 'String',
    'city': 'String',
    'geonames_city_id': 'String',
    'region': 'String',
    'country': 'String',
    'latitude': 'DECIMAL',
    'longitude': 'DECIMAL',
    'associated_institution_id': 'String',
    'relationship': 'String',
    'source_id': 'String',
    'fatcat': 'String',
    'work_id': 'String',
    'topic_id': 'String',
    'related_work_id': 'String',
    'referenced_work_id': 'String',
    'source_id': 'String',
    'location_type': 'Enum',
    'landing_page_url': 'String',
    'pdf_url': 'String',
    'oa_status': 'Enum',
    'oa_url': 'String',
    'any_repository_has_fulltext': 'Boolean',
    'version': 'String',
    'license': 'String',
    'descriptor_ui': 'String',
    'descriptor_name': 'String',
    'qualifier_ui': 'String',
    'qualifier_name': 'String',
    'is_major_topic': 'Boolean',
    'author_position': 'String',
    'raw_affiliation_string': 'Text',
    'entity_id': 'String',
    'entity_type': 'Enum',
    'year': 'Integer',
    'oa_works_count': 'Integer'
}

def get_table_field_types(table_name):
    """根据表名获取字段类型映射"""
    # 这里需要根据你的实际数据库模型来返回每个表字段的类型
    # 这是一个示例，你需要根据你的模型定义来完善它
    # 例如，你可以有一个函数或类来管理这些映射
    # 或者从SQLAlchemy的模型定义中动态生成
    # 由于我没有完整的模型定义，这里只能提供一个框架
    # 你需要补充具体的字段类型映射逻辑
    
    # 示例：根据table_name返回对应的字段类型字典
    # 这是一个简化的示例，实际应用中可能需要更复杂的逻辑
    model = get_table_model(table_name)
    if model:
        return {c.name: c.type.__class__.__name__ for c in model.__table__.columns}
    return {}

def get_table_model(table_name):
    """根据表名获取对应的SQLAlchemy模型"""
    # 这里需要根据你的实际模型定义来返回对应的模型类
    # 这是一个示例，你需要根据你的模型定义来完善它
    # 例如，你可以有一个字典或函数来映射表名到模型类
    
    # 示例：根据table_name返回对应的模型类
    model_mapping = {
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
        'yearly_stats': YearlyStat,
        'operation_logs': OperationLog # 添加operation_logs模型映射
    }
    return model_mapping.get(table_name)

def convert_field_type(value, field_type):
    # 处理 nan 值
    if pd.isna(value):
        if field_type in ['String', 'Text', 'JSON']:
            return None
        elif field_type in ['Integer', 'Float', 'DECIMAL']:
            return 0
        elif field_type == 'Boolean':
            return False
        elif field_type in ['Date', 'DateTime']:
            return None
        else:
            return None

    # 如果是字符串类型，先去除首尾空格
    if isinstance(value, str):
        value = value.strip()
        # 如果是空字符串，返回None
        if not value:
            return None

    try:
        if field_type == 'String':
            return str(value)
        elif field_type == 'Integer':
            # 尝试转换为整数
            try:
                return int(float(value))
            except (ValueError, TypeError):
                return 0
        elif field_type == 'Float':
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0
        elif field_type == 'Boolean':
            if isinstance(value, str):
                value = value.lower()
                if value in ['true', '1', 'yes', 'y']:
                    return True
                elif value in ['false', '0', 'no', 'n']:
                    return False
            return bool(value)
        elif field_type == 'Date':
            try:
                return pd.to_datetime(value).date()
            except:
                return None
        elif field_type == 'DateTime':
            try:
                return pd.to_datetime(value)
            except:
                return None
        elif field_type == 'JSON':
            try:
                if isinstance(value, str):
                    return json.loads(value)
                return value
            except:
                return value
        elif field_type == 'Text':
            return str(value)
        elif field_type == 'DECIMAL':
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0
        elif field_type == 'Enum':
            return str(value)
        else:
            return value
    except Exception as e:
        print(f"转换值 {value} 到类型 {field_type} 时出错: {str(e)}")
        return None

def import_csv_to_table(csv_file, table_name):
    """导入CSV文件到指定表"""
    print(f"开始导入 {csv_file} 到表 {table_name}")

    try:
        # 检查数据库连接
        try:
            from sqlalchemy import text
            db.session.execute(text("SELECT 1"))
        except Exception as e:
            print(f"数据库连接失败: {str(e)}")
            return False

        # 清空表
        try:
            # 禁用外键检查
            db.session.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
            # 清空表
            db.session.execute(text(f"TRUNCATE TABLE {table_name}"))
            # 启用外键检查
            db.session.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
            db.session.commit()
            print(f"表 {table_name} 已清空")
        except Exception as e:
            print(f"清空表失败: {str(e)}")
            # 确保外键检查被重新启用
            db.session.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
            db.session.commit()
            return False

        # 读取CSV文件
        try:
            df = pd.read_csv(csv_file, header=0)
            print(f"读取到 {len(df)} 条记录")
            
            # 特别处理works表
            if table_name == 'works':
                print("\nWorks表数据预览:")
                print(df.head())
                print("\nWorks表列名:", df.columns.tolist())
                print("\nWorks表数据类型:")
                print(df.dtypes)
        except Exception as e:
            print(f"读取CSV文件失败: {str(e)}")
            return False

        # 获取表的字段类型映射
        field_types = get_table_field_types(table_name)
        if table_name == 'works':
            print("\nWorks表字段类型:", field_types)

        # 转换数据类型并处理NaN
        for column in df.columns:
            if column in field_types:
                try:
                    df[column] = df[column].apply(lambda x: convert_field_type(x, field_types[column]))
                except Exception as e:
                    print(f"转换列 {column} 时出错: {str(e)}")
                    if table_name == 'works':
                        print(f"错误数据示例: {df[column].head()}")
            else:
                if column in df.columns:
                    print(f"警告: 列 {column} 不在表 {table_name} 的字段定义中，将被忽略")
                    df = df.drop(columns=[column])

        # 获取表模型
        model = get_table_model(table_name)
        if not model:
            raise ValueError(f"未知的表名: {table_name}")

        # 确保所有模型需要的列都存在于DataFrame中
        model_columns = [c.name for c in model.__table__.columns]
        if table_name == 'works':
            print("\nWorks表需要的列:", model_columns)
            print("DataFrame中的列:", df.columns.tolist())
        
        for col in model_columns:
            if col not in df.columns:
                print(f"警告: 列 {col} 在CSV中不存在，将被设置为NULL")
                df[col] = None

        # 确保DataFrame的列顺序与模型一致
        df = df[model_columns]

        # 开始事务
        success_count = 0
        error_count = 0
        
        try:
            # 批量插入数据
            data_to_insert = df.to_dict(orient='records')
            
            for row_dict in data_to_insert:
                try:
                    # 特别处理works_mesh的qualifier_ui字段
                    if table_name == 'works_mesh':
                        if 'qualifier_ui' not in row_dict or row_dict['qualifier_ui'] is None or str(row_dict['qualifier_ui']).strip() == '':
                            row_dict['qualifier_ui'] = ''

                    # 实例化模型并添加会话
                    record = model(**row_dict)
                    db.session.add(record)
                    success_count += 1
                    
                    # # 每100条记录提交一次
                    # if success_count % 1000 == 0:
                    #     db.session.commit()
                    #     print(f"已处理 {success_count} 条记录")

                except Exception as e:
                    error_count += 1
                    if table_name == 'works':
                        print(f"\n插入works记录失败: {str(e)}")
                        print(f"失败的数据: {row_dict}")
                    db.session.rollback()
                    continue

            # 最后提交剩余的事务
            db.session.commit()
            print(f"表 {table_name} 导入完成: 成功 {success_count} 条, 失败 {error_count} 条")
            return True

        except Exception as e:
            print(f"事务处理失败: {str(e)}")
            db.session.rollback()
            return False

    except Exception as e:
        print(f"导入失败: {str(e)}")
        db.session.rollback()
        return False

def import_all_csvs(csv_dir):
    """导入所有CSV文件"""
    print(f"开始导入目录: {csv_dir}")
    
    # 检查目录是否存在
    if not os.path.exists(csv_dir):
        print(f"目录不存在: {csv_dir}")
        return
        
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    for filename in csv_files:
        file_path = os.path.join(csv_dir, filename)
        table_name = csv_to_table_mapping.get(filename, os.path.splitext(filename)[0].rstrip('s'))
        entity_type = table_to_entity_type.get(table_name, table_name.rstrip('s'))[:30]
        
        print(f"正在导入 {filename} 到表 {table_name}...")
        try:
            if import_csv_to_table(file_path, table_name):
                print(f"文件 {filename} 处理完成")
            else:
                print(f"文件 {filename} 处理失败")
        except Exception as e:
            print(f"处理文件 {filename} 时发生未预期错误: {str(e)}")

if __name__ == '__main__':
    print("--- Running the updated import_data.py script ---")
    csv_dir = os.path.join(os.path.dirname(__file__), '..', '当下所有')
    
    with app.app_context():
        try:
            # 初始化数据库
            print("初始化数据库...")
            db.drop_all()
            db.create_all()
            db.session.commit()
            print("数据库初始化完成")
            
            # 导入数据
            import_all_csvs(csv_dir)
        except Exception as e:
            print(f"发生错误: {str(e)}")
            db.session.rollback() 