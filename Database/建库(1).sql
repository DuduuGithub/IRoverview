-- 删除数据库（如果存在）
DROP DATABASE IF EXISTS iroverview;

-- 创建数据库
CREATE DATABASE iroverview
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

-- 使用数据库
USE iroverview;

-- 设置SQL模式
SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";

-- 设置字符集
SET NAMES utf8mb4;

-- 表1：作者详细信息表（核心表）
CREATE TABLE authors (
    id VARCHAR(255) COMMENT 'OpenAlex系统中的作者唯一标识符',
    orcid VARCHAR(255) COMMENT 'ORCID系统中的作者ID',
    display_name VARCHAR(255) NOT NULL COMMENT '作者的标准显示名称',
    display_name_alternatives JSON COMMENT '作者的其他名称变体，JSON格式存储',
    works_count INT DEFAULT 0 COMMENT '作者发表的作品总数',
    cited_by_count INT DEFAULT 0 COMMENT '作者所有作品被引用的总次数',
    last_known_institution VARCHAR(255) COMMENT '作者最后已知的所属机构',
    works_api_url VARCHAR(255) COMMENT '获取作者作品的API URL',
    updated_date DATETIME COMMENT '作者信息最后更新时间',
    PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='表1：作者详细信息表，存储作者的基本信息和统计信息';

-- 表2：概念详细信息表（核心表）
CREATE TABLE concepts (
    id VARCHAR(255) COMMENT 'OpenAlex系统中的概念唯一标识符',
    wikidata VARCHAR(255) COMMENT 'Wikidata中的概念ID',
    display_name VARCHAR(255) NOT NULL COMMENT '概念的标准显示名称',
    level INT COMMENT '概念层级',
    description TEXT COMMENT '概念描述',
    works_count INT DEFAULT 0 COMMENT '相关作品总数',
    cited_by_count INT DEFAULT 0 COMMENT '被引用总次数',
    image_url VARCHAR(255) COMMENT '概念图片URL',
    image_thumbnail_url VARCHAR(255) COMMENT '概念缩略图URL',
    works_api_url VARCHAR(255) COMMENT '获取相关作品的API URL',
    updated_date DATE COMMENT '概念信息最后更新时间',
    PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='表2：概念详细信息表，存储研究领域的基本信息和统计信息';

-- 表3：机构详细信息表（核心表）
CREATE TABLE institutions (
    id VARCHAR(255) COMMENT 'OpenAlex系统中的机构唯一标识符',
    ror VARCHAR(255) COMMENT 'ROR系统中的机构ID',
    display_name VARCHAR(255) COMMENT '机构的标准显示名称',
    country_code VARCHAR(2) COMMENT '机构所在国家代码',
    type VARCHAR(255) COMMENT '机构类型',
    homepage_url VARCHAR(255) COMMENT '机构主页URL',
    image_url VARCHAR(255) COMMENT '机构图片URL',
    image_thumbnail_url VARCHAR(255) COMMENT '机构缩略图URL',
    display_name_acronyms JSON COMMENT '机构名称缩写',
    display_name_alternatives JSON COMMENT '机构的其他名称',
    works_count INT DEFAULT 0 COMMENT '相关作品总数',
    cited_by_count INT DEFAULT 0 COMMENT '被引用总次数',
    works_api_url VARCHAR(255) COMMENT '获取相关作品的API URL',
    updated_date DATETIME COMMENT '机构信息最后更新时间',
    PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='表3：机构详细信息表，存储机构的基本信息和统计信息';

-- 表4：来源详细信息表（核心表）
CREATE TABLE sources (
    id VARCHAR(255) COMMENT 'OpenAlex系统中的来源唯一标识符',
    issn_l VARCHAR(255) COMMENT 'ISSN-L标识符',
    issn JSON COMMENT 'ISSN列表',
    display_name VARCHAR(255) COMMENT '来源的标准显示名称',
    works_count INT DEFAULT 0 COMMENT '相关作品总数',
    cited_by_count INT DEFAULT 0 COMMENT '被引用总次数',
    is_oa BOOLEAN DEFAULT FALSE COMMENT '是否为开放获取',
    is_in_doaj BOOLEAN DEFAULT FALSE COMMENT '是否在DOAJ中',
    homepage_url VARCHAR(255) COMMENT '来源主页URL',
    works_api_url VARCHAR(255) COMMENT '获取相关作品的API URL',
    updated_date DATETIME COMMENT '来源信息最后更新时间',
    PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='表4：来源详细信息表，存储来源的基本信息和统计信息';

-- 表5：主题详细信息表（核心表）
CREATE TABLE topics (
    id VARCHAR(255) COMMENT 'OpenAlex系统中的主题唯一标识符',
    display_name VARCHAR(255) COMMENT '主题的标准显示名称',
    subfield_id VARCHAR(255) COMMENT '子领域ID',
    subfield_display_name VARCHAR(255) COMMENT '子领域名称',
    field_id VARCHAR(255) COMMENT '领域ID',
    field_display_name VARCHAR(255) COMMENT '领域名称',
    domain_id VARCHAR(255) COMMENT '学科领域ID',
    domain_display_name VARCHAR(255) COMMENT '学科领域名称',
    description TEXT COMMENT '主题描述',
    keywords TEXT COMMENT '关键词列表',
    works_api_url VARCHAR(255) COMMENT '获取相关作品的API URL',
    wikipedia_id VARCHAR(255) COMMENT 'Wikipedia页面ID',
    works_count INT DEFAULT 0 COMMENT '相关作品总数',
    cited_by_count INT DEFAULT 0 COMMENT '被引用总次数',
    updated_date DATETIME COMMENT '主题信息最后更新时间',
    PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='表5：主题详细信息表，存储主题的基本信息和统计信息';

-- 表6：作品主表（核心表）
CREATE TABLE works (
    id VARCHAR(255) COMMENT 'OpenAlex系统中的作品唯一标识符',
    doi VARCHAR(255) COMMENT '数字对象唯一标识符',
    title TEXT COMMENT '作品标题',
    display_name TEXT COMMENT '作品的标准显示名称',
    publication_year INT COMMENT '出版年份',
    publication_date DATE COMMENT '出版日期',
    type VARCHAR(50) COMMENT '作品类型',
    cited_by_count INT DEFAULT 0 COMMENT '被引用次数',
    is_retracted BOOLEAN DEFAULT FALSE COMMENT '是否被撤回',
    is_paratext BOOLEAN DEFAULT FALSE COMMENT '是否为辅助文本',
    cited_by_api_url VARCHAR(255) COMMENT '获取引用作品的API URL',
    abstract_inverted_index JSON COMMENT '摘要倒排索引',
    language VARCHAR(10) COMMENT '作品语言',
    -- 合并works_ids表的字段
    openalex VARCHAR(255) COMMENT 'OpenAlex系统中的作品ID',
    mag VARCHAR(255) COMMENT 'Microsoft Academic Graph中的作品ID',
    pmid VARCHAR(255) COMMENT 'PubMed中的作品ID',
    pmcid VARCHAR(255) COMMENT 'PubMed Central中的作品ID',
    -- 合并works_biblio表的字段
    volume VARCHAR(50) COMMENT '卷号',
    issue VARCHAR(50) COMMENT '期号',
    first_page VARCHAR(50) COMMENT '起始页码',
    last_page VARCHAR(50) COMMENT '结束页码',
    PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='表6：作品主表，存储作品的基本信息';

-- 表7：作者ID映射表（从属表）
CREATE TABLE authors_ids (
    author_id VARCHAR(255) COMMENT '作者唯一标识符，关联authors表',
    openalex VARCHAR(255) COMMENT 'OpenAlex系统中的作者ID',
    orcid VARCHAR(255) COMMENT 'ORCID系统中的作者ID',
    scopus VARCHAR(255) COMMENT 'Scopus系统中的作者ID',
    twitter VARCHAR(255) COMMENT 'Twitter账号ID',
    wikipedia VARCHAR(255) COMMENT 'Wikipedia页面ID',
    mag VARCHAR(255) COMMENT 'Microsoft Academic Graph中的作者ID',
    PRIMARY KEY (author_id),
    FOREIGN KEY (author_id) REFERENCES authors(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='表7：作者ID映射表，存储作者在不同系统中的ID';

-- 表8：概念ID映射表（从属表）
CREATE TABLE concepts_ids (
    concept_id VARCHAR(255) COMMENT '概念唯一标识符，关联concepts表',
    openalex VARCHAR(255) COMMENT 'OpenAlex系统中的概念ID',
    wikidata VARCHAR(255) COMMENT 'Wikidata中的概念ID',
    wikipedia VARCHAR(255) COMMENT 'Wikipedia页面ID',
    umls_aui VARCHAR(255) COMMENT 'UMLS AUI标识符',
    umls_cui VARCHAR(255) COMMENT 'UMLS CUI标识符',
    mag VARCHAR(255) COMMENT 'Microsoft Academic Graph中的概念ID',
    PRIMARY KEY (concept_id),
    FOREIGN KEY (concept_id) REFERENCES concepts(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='表8：概念ID映射表，存储研究领域在不同系统中的ID';

-- 表9：概念层级关系表（从属表）
CREATE TABLE concepts_ancestors (
    concept_id VARCHAR(255) COMMENT '子概念ID，关联concepts表',
    ancestor_id VARCHAR(255) COMMENT '父概念ID，关联concepts表',
    PRIMARY KEY (concept_id, ancestor_id),
    FOREIGN KEY (concept_id) REFERENCES concepts(id) ON DELETE CASCADE,
    FOREIGN KEY (ancestor_id) REFERENCES concepts(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='表9：概念层级关系表，记录研究领域之间的层级关系';

-- 表10：概念相关性关系表（从属表）
CREATE TABLE concepts_related_concepts (
    concept_id VARCHAR(255) COMMENT '概念ID，关联concepts表',
    related_concept_id VARCHAR(255) COMMENT '相关概念ID，关联concepts表',
    score FLOAT COMMENT '相关性得分',
    PRIMARY KEY (concept_id, related_concept_id),
    FOREIGN KEY (concept_id) REFERENCES concepts(id) ON DELETE CASCADE,
    FOREIGN KEY (related_concept_id) REFERENCES concepts(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='表10：概念相关性关系表，记录研究领域之间的相关性关系';

-- 表11：机构ID映射表（从属表）
CREATE TABLE institutions_ids (
    institution_id VARCHAR(255) COMMENT '机构唯一标识符',
    openalex VARCHAR(255) COMMENT 'OpenAlex系统中的机构ID',
    ror VARCHAR(255) COMMENT 'ROR系统中的机构ID',
    grid VARCHAR(255) COMMENT 'GRID系统中的机构ID',
    wikipedia VARCHAR(255) COMMENT 'Wikipedia中的机构ID',
    wikidata VARCHAR(255) COMMENT 'Wikidata中的机构ID',
    mag VARCHAR(255) COMMENT 'Microsoft Academic Graph中的机构ID',
    PRIMARY KEY (institution_id),
    FOREIGN KEY (institution_id) REFERENCES institutions(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='表11：机构ID映射表，存储机构在不同系统中的ID';

-- 表12：机构地理位置表（从属表）
CREATE TABLE institutions_geo (
    institution_id VARCHAR(255) COMMENT '机构唯一标识符',
    city VARCHAR(255) COMMENT '城市名称',
    geonames_city_id VARCHAR(255) COMMENT 'GeoNames中的城市ID',
    region VARCHAR(255) COMMENT '地区名称',
    country_code VARCHAR(2) COMMENT '国家代码',
    country VARCHAR(255) COMMENT '国家名称',
    latitude DECIMAL(10,6) COMMENT '纬度',
    longitude DECIMAL(10,6) COMMENT '经度',
    PRIMARY KEY (institution_id),
    FOREIGN KEY (institution_id) REFERENCES institutions(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='表12：机构地理位置表，存储机构的地理位置信息';

-- 表13：机构关联关系表（从属表）
CREATE TABLE institutions_associated_institutions (
    institution_id VARCHAR(255) COMMENT '机构ID',
    associated_institution_id VARCHAR(255) COMMENT '关联机构ID',
    relationship VARCHAR(255) COMMENT '关联关系类型',
    PRIMARY KEY (institution_id, associated_institution_id),
    FOREIGN KEY (institution_id) REFERENCES institutions(id) ON DELETE CASCADE,
    FOREIGN KEY (associated_institution_id) REFERENCES institutions(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='表13：机构关联关系表，记录机构之间的关联关系';

-- 表14：来源ID映射表（从属表）
CREATE TABLE sources_ids (
    source_id VARCHAR(255) COMMENT '来源唯一标识符',
    openalex VARCHAR(255) COMMENT 'OpenAlex系统中的来源ID',
    issn_l VARCHAR(255) COMMENT 'ISSN-L标识符',
    issn JSON COMMENT 'ISSN列表',
    mag VARCHAR(255) COMMENT 'Microsoft Academic Graph中的来源ID',
    wikidata VARCHAR(255) COMMENT 'Wikidata中的来源ID',
    fatcat VARCHAR(255) COMMENT 'Fatcat中的来源ID',
    PRIMARY KEY (source_id),
    FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='表14：来源ID映射表，存储来源在不同系统中的ID';

-- 表15：作品-主题关联表（从属表）
CREATE TABLE works_topics (
    work_id VARCHAR(255) COMMENT '作品ID',
    topic_id VARCHAR(255) COMMENT '主题ID',
    score FLOAT COMMENT '关联强度得分',
    PRIMARY KEY (work_id, topic_id),
    FOREIGN KEY (work_id) REFERENCES works(id) ON DELETE CASCADE,
    FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='表15：作品-主题关联表，记录作品与主题之间的关联关系';

-- 表16：作品-相关作品关联表（从属表）
CREATE TABLE works_related_works (
    work_id VARCHAR(255) COMMENT '作品ID',
    related_work_id VARCHAR(255) COMMENT '相关作品ID',
    PRIMARY KEY (work_id, related_work_id),
    FOREIGN KEY (work_id) REFERENCES works(id) ON DELETE CASCADE,
    FOREIGN KEY (related_work_id) REFERENCES works(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='表16：作品-相关作品关联表，记录作品之间的关联关系';

-- 表17：作品-引用作品关联表（从属表）
CREATE TABLE works_referenced_works (
    work_id VARCHAR(255) COMMENT '引用作品ID',
    referenced_work_id VARCHAR(255) COMMENT '被引用作品ID',
    PRIMARY KEY (work_id, referenced_work_id),
    FOREIGN KEY (work_id) REFERENCES works(id) ON DELETE CASCADE,
    FOREIGN KEY (referenced_work_id) REFERENCES works(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='表17：作品-引用作品关联表，记录作品之间的引用关系';

-- 表18：作品-位置关联表（从属表）
CREATE TABLE works_locations (
    work_id VARCHAR(255) COMMENT '作品ID，关联works表',
    source_id VARCHAR(255) COMMENT '来源ID，关联sources表',
    location_type ENUM('primary', 'best_oa', 'other') COMMENT '位置类型',
    landing_page_url VARCHAR(255) COMMENT '作品落地页URL',
    pdf_url VARCHAR(255) COMMENT 'PDF文件URL',
    is_oa BOOLEAN DEFAULT FALSE COMMENT '是否为开放获取',
    oa_status VARCHAR(50) COMMENT '开放获取状态',
    oa_url VARCHAR(255) COMMENT '开放获取URL',
    any_repository_has_fulltext BOOLEAN DEFAULT FALSE COMMENT '是否有任何存储库包含全文',
    version VARCHAR(50) COMMENT '版本信息',
    license VARCHAR(255) COMMENT '许可信息',
    PRIMARY KEY (work_id, source_id, location_type),
    FOREIGN KEY (work_id) REFERENCES works(id) ON DELETE CASCADE,
    FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='表18：作品-位置关联表，记录作品在不同来源的发布位置信息';

-- 表19：作品-MeSH主题关联表（从属表）
CREATE TABLE works_mesh (
    work_id VARCHAR(255) COMMENT '作品ID，关联works表',
    descriptor_ui VARCHAR(255) COMMENT 'MeSH主题唯一标识符（Descriptor UI）',
    descriptor_name VARCHAR(255) COMMENT 'MeSH主题名称',
    qualifier_ui VARCHAR(255) COMMENT 'MeSH限定词唯一标识符（Qualifier UI）',
    qualifier_name VARCHAR(255) COMMENT 'MeSH限定词名称',
    is_major_topic BOOLEAN COMMENT '是否为主要主题',
    PRIMARY KEY (work_id, descriptor_ui, qualifier_ui),
    FOREIGN KEY (work_id) REFERENCES works(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='表19：作品-MeSH主题关联表，记录作品与MeSH主题及限定词的对应关系';

-- 表20：作品-概念关联表（从属表）
CREATE TABLE works_concepts (
    work_id VARCHAR(255) COMMENT '作品ID，关联works表',
    concept_id VARCHAR(255) COMMENT '概念ID，关联concepts表',
    score FLOAT COMMENT '关联强度得分',
    PRIMARY KEY (work_id, concept_id),
    FOREIGN KEY (work_id) REFERENCES works(id) ON DELETE CASCADE,
    FOREIGN KEY (concept_id) REFERENCES concepts(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='表20：作品-概念关联表，记录作品与研究领域概念的对应关系';

-- 表21：作品-作者署名表（从属表）
CREATE TABLE works_authorships (
    work_id VARCHAR(255) COMMENT '作品ID，关联works表',
    author_position VARCHAR(50) COMMENT '作者署名顺序',
    author_id VARCHAR(255) COMMENT '作者ID，关联authors表',
    institution_id VARCHAR(255) COMMENT '机构ID，关联institutions表',
    raw_affiliation_string TEXT COMMENT '原始机构隶属关系字符串',
    PRIMARY KEY (work_id, author_position),
    FOREIGN KEY (work_id) REFERENCES works(id) ON DELETE CASCADE,
    FOREIGN KEY (author_id) REFERENCES authors(id) ON DELETE CASCADE,
    FOREIGN KEY (institution_id) REFERENCES institutions(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='表21：作品-作者署名表，记录作品的作者署名信息及机构隶属关系';

-- 表22：年度统计表（从属表）
CREATE TABLE yearly_stats (
    entity_id VARCHAR(255) COMMENT '实体ID',
    entity_type ENUM('author', 'concept', 'institution', 'source') COMMENT '实体类型',
    year INT COMMENT '统计年份',
    works_count INT DEFAULT 0 COMMENT '该年度相关作品数量',
    cited_by_count INT DEFAULT 0 COMMENT '该年度作品被引用的总次数',
    oa_works_count INT DEFAULT 0 COMMENT '该年度开放获取的作品数量',
    PRIMARY KEY (entity_id, entity_type, year),
    INDEX idx_entity (entity_type, entity_id),
    INDEX idx_year (year)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='表22：年度统计表，记录各类实体的年度统计信息';

-- 表23：操作记录表（从属表）
CREATE TABLE operation_logs (
    id BIGINT AUTO_INCREMENT COMMENT '记录ID',
    operation_type ENUM('import', 'update', 'delete', 'export', 'search') COMMENT '操作类型',
    entity_type ENUM('author', 'concept', 'institution', 'source', 'work', 'topic', 'search_session', 'search_result') COMMENT '实体类型',
    entity_id VARCHAR(255) COMMENT '实体ID',
    operation_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '操作时间',
    operator VARCHAR(50) COMMENT '操作者',
    operation_status ENUM('success', 'failed') COMMENT '操作状态',
    operation_details TEXT COMMENT '操作详情',
    error_message TEXT COMMENT '错误信息',
    affected_rows INT COMMENT '影响行数',
    PRIMARY KEY (id),
    INDEX idx_operation_time (operation_time),
    INDEX idx_entity (entity_type, entity_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='表23：操作记录表，记录数据导入、更新、删除等操作历史';

-- 表24：搜索会话表（核心表）
CREATE TABLE search_sessions (
    session_id VARCHAR(100) PRIMARY KEY COMMENT '会话唯一标识符',
    search_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '搜索时间',
    query_text TEXT NOT NULL COMMENT '检索式',
    total_results INT DEFAULT 0 COMMENT '搜索结果总数',
    INDEX idx_search_time (search_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='表24：搜索会话表，记录用户搜索行为和条件';

-- 表25：搜索结果表（核心表）
CREATE TABLE search_results (
    session_id VARCHAR(100) COMMENT '关联的会话ID',
    entity_type ENUM('work', 'author', 'concept', 'institution', 'source', 'topic') COMMENT '结果实体类型',
    entity_id VARCHAR(255) COMMENT '结果实体ID',
    rank_position INT COMMENT '搜索结果排名位置',
    relevance_score FLOAT COMMENT '相关性得分',
    query_text TEXT COMMENT '检索式',
    result_page INT COMMENT '结果所在页码',
    result_position INT COMMENT '结果在页面中的位置',
    is_clicked BOOLEAN DEFAULT FALSE COMMENT '是否被点击',
    click_time DATETIME COMMENT '点击时间',
    dwell_time INT DEFAULT 0 COMMENT '停留时间（秒）',
    PRIMARY KEY (session_id, entity_id),
    INDEX idx_entity (entity_type, entity_id),
    INDEX idx_click_time (click_time),
    FOREIGN KEY (session_id) REFERENCES search_sessions(session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='表25：搜索结果表，记录搜索结果及用户交互行为';

-- 表26：重排序会话表（核心表）
CREATE TABLE rerank_sessions (
    session_id VARCHAR(100) PRIMARY KEY COMMENT '重排序会话ID',
    search_session_id VARCHAR(100) COMMENT '关联的搜索会话ID',
    rerank_query TEXT COMMENT '重排序查询文本',
    rerank_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '重排序时间',
    INDEX idx_search_session (search_session_id),
    FOREIGN KEY (search_session_id) REFERENCES search_sessions(session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='表26：重排序会话表，记录用户重排序行为';

-- 表27：用户行为表（核心表）
CREATE TABLE user_behaviors (
    session_id VARCHAR(100) COMMENT '关联的搜索会话ID',
    rerank_session_id VARCHAR(100) COMMENT '关联的重排序会话ID',
    document_id VARCHAR(255) COMMENT '文档ID',
    rank_position INT COMMENT '排名位置',
    is_clicked BOOLEAN DEFAULT FALSE COMMENT '是否被点击',
    click_time DATETIME COMMENT '点击时间',
    dwell_time INT DEFAULT 0 COMMENT '停留时间（秒）',
    behavior_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '行为发生时间',
    PRIMARY KEY (session_id, document_id),
    INDEX idx_rerank_session (rerank_session_id),
    INDEX idx_document (document_id),
    INDEX idx_rank (rank_position),
    FOREIGN KEY (session_id) REFERENCES search_sessions(session_id),
    FOREIGN KEY (rerank_session_id) REFERENCES rerank_sessions(session_id),
    FOREIGN KEY (document_id) REFERENCES works(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='表27：用户行为表，记录用户与文档的交互行为';

-- 创建触发器函数
-- 为五个主要表（authors、works、concepts、institutions、sources、topics）都创建了触发器，每个表都有三个触发器：
-- 1. AFTER INSERT 触发器：记录新增操作，设置操作类型为'import'
-- 2. AFTER UPDATE 触发器：记录更新操作，设置操作类型为'update'
-- 3. AFTER DELETE 触发器：记录删除操作，设置操作类型为'delete'
-- 每个触发器都会自动记录操作时间、操作状态、影响行数和操作详情
DELIMITER //

-- 作者表触发器
CREATE TRIGGER authors_after_insert
AFTER INSERT ON authors
FOR EACH ROW
BEGIN
    INSERT INTO operation_logs (
        operation_type,
        entity_type,
        entity_id,
        operation_status,
        operation_details,
        affected_rows
    ) VALUES (
        'import',
        'author',
        NEW.id,
        'success',
        CONCAT('新增作者: ', NEW.display_name),
        1
    );
END//

CREATE TRIGGER authors_after_update
AFTER UPDATE ON authors
FOR EACH ROW
BEGIN
    INSERT INTO operation_logs (
        operation_type,
        entity_type,
        entity_id,
        operation_status,
        operation_details,
        affected_rows
    ) VALUES (
        'update',
        'author',
        NEW.id,
        'success',
        CONCAT('更新作者信息: ', NEW.display_name),
        1
    );
END//

CREATE TRIGGER authors_after_delete
AFTER DELETE ON authors
FOR EACH ROW
BEGIN
    INSERT INTO operation_logs (
        operation_type,
        entity_type,
        entity_id,
        operation_status,
        operation_details,
        affected_rows
    ) VALUES (
        'delete',
        'author',
        OLD.id,
        'success',
        CONCAT('删除作者: ', OLD.display_name),
        1
    );
END//

-- 作品表触发器
CREATE TRIGGER works_after_insert
AFTER INSERT ON works
FOR EACH ROW
BEGIN
    INSERT INTO operation_logs (
        operation_type,
        entity_type,
        entity_id,
        operation_status,
        operation_details,
        affected_rows
    ) VALUES (
        'import',
        'work',
        NEW.id,
        'success',
        CONCAT('新增作品: ', NEW.title),
        1
    );
END//

CREATE TRIGGER works_after_update
AFTER UPDATE ON works
FOR EACH ROW
BEGIN
    INSERT INTO operation_logs (
        operation_type,
        entity_type,
        entity_id,
        operation_status,
        operation_details,
        affected_rows
    ) VALUES (
        'update',
        'work',
        NEW.id,
        'success',
        CONCAT('更新作品信息: ', NEW.title),
        1
    );
END//

CREATE TRIGGER works_after_delete
AFTER DELETE ON works
FOR EACH ROW
BEGIN
    INSERT INTO operation_logs (
        operation_type,
        entity_type,
        entity_id,
        operation_status,
        operation_details,
        affected_rows
    ) VALUES (
        'delete',
        'work',
        OLD.id,
        'success',
        CONCAT('删除作品: ', OLD.title),
        1
    );
END//

-- 概念表触发器
CREATE TRIGGER concepts_after_insert
AFTER INSERT ON concepts
FOR EACH ROW
BEGIN
    INSERT INTO operation_logs (
        operation_type,
        entity_type,
        entity_id,
        operation_status,
        operation_details,
        affected_rows
    ) VALUES (
        'import',
        'concept',
        NEW.id,
        'success',
        CONCAT('新增概念: ', NEW.display_name),
        1
    );
END//

CREATE TRIGGER concepts_after_update
AFTER UPDATE ON concepts
FOR EACH ROW
BEGIN
    INSERT INTO operation_logs (
        operation_type,
        entity_type,
        entity_id,
        operation_status,
        operation_details,
        affected_rows
    ) VALUES (
        'update',
        'concept',
        NEW.id,
        'success',
        CONCAT('更新概念信息: ', NEW.display_name),
        1
    );
END//

CREATE TRIGGER concepts_after_delete
AFTER DELETE ON concepts
FOR EACH ROW
BEGIN
    INSERT INTO operation_logs (
        operation_type,
        entity_type,
        entity_id,
        operation_status,
        operation_details,
        affected_rows
    ) VALUES (
        'delete',
        'concept',
        OLD.id,
        'success',
        CONCAT('删除概念: ', OLD.display_name),
        1
    );
END//

-- 机构表触发器
CREATE TRIGGER institutions_after_insert
AFTER INSERT ON institutions
FOR EACH ROW
BEGIN
    INSERT INTO operation_logs (
        operation_type,
        entity_type,
        entity_id,
        operation_status,
        operation_details,
        affected_rows
    ) VALUES (
        'import',
        'institution',
        NEW.id,
        'success',
        CONCAT('新增机构: ', NEW.display_name),
        1
    );
END//

CREATE TRIGGER institutions_after_update
AFTER UPDATE ON institutions
FOR EACH ROW
BEGIN
    INSERT INTO operation_logs (
        operation_type,
        entity_type,
        entity_id,
        operation_status,
        operation_details,
        affected_rows
    ) VALUES (
        'update',
        'institution',
        NEW.id,
        'success',
        CONCAT('更新机构信息: ', NEW.display_name),
        1
    );
END//

CREATE TRIGGER institutions_after_delete
AFTER DELETE ON institutions
FOR EACH ROW
BEGIN
    INSERT INTO operation_logs (
        operation_type,
        entity_type,
        entity_id,
        operation_status,
        operation_details,
        affected_rows
    ) VALUES (
        'delete',
        'institution',
        OLD.id,
        'success',
        CONCAT('删除机构: ', OLD.display_name),
        1
    );
END//

-- 来源表触发器
CREATE TRIGGER sources_after_insert
AFTER INSERT ON sources
FOR EACH ROW
BEGIN
    INSERT INTO operation_logs (
        operation_type,
        entity_type,
        entity_id,
        operation_status,
        operation_details,
        affected_rows
    ) VALUES (
        'import',
        'source',
        NEW.id,
        'success',
        CONCAT('新增来源: ', NEW.display_name),
        1
    );
END//

CREATE TRIGGER sources_after_update
AFTER UPDATE ON sources
FOR EACH ROW
BEGIN
    INSERT INTO operation_logs (
        operation_type,
        entity_type,
        entity_id,
        operation_status,
        operation_details,
        affected_rows
    ) VALUES (
        'update',
        'source',
        NEW.id,
        'success',
        CONCAT('更新来源信息: ', NEW.display_name),
        1
    );
END//

CREATE TRIGGER sources_after_delete
AFTER DELETE ON sources
FOR EACH ROW
BEGIN
    INSERT INTO operation_logs (
        operation_type,
        entity_type,
        entity_id,
        operation_status,
        operation_details,
        affected_rows
    ) VALUES (
        'delete',
        'source',
        OLD.id,
        'success',
        CONCAT('删除来源: ', OLD.display_name),
        1
    );
END//

-- 主题表触发器
CREATE TRIGGER topics_after_insert
AFTER INSERT ON topics
FOR EACH ROW
BEGIN
    INSERT INTO operation_logs (
        operation_type,
        entity_type,
        entity_id,
        operation_status,
        operation_details,
        affected_rows
    ) VALUES (
        'import',
        'topic',
        NEW.id,
        'success',
        CONCAT('新增主题: ', NEW.display_name),
        1
    );
END//

CREATE TRIGGER topics_after_update
AFTER UPDATE ON topics
FOR EACH ROW
BEGIN
    INSERT INTO operation_logs (
        operation_type,
        entity_type,
        entity_id,
        operation_status,
        operation_details,
        affected_rows
    ) VALUES (
        'update',
        'topic',
        NEW.id,
        'success',
        CONCAT('更新主题信息: ', NEW.display_name),
        1
    );
END//

CREATE TRIGGER topics_after_delete
AFTER DELETE ON topics
FOR EACH ROW
BEGIN
    INSERT INTO operation_logs (
        operation_type,
        entity_type,
        entity_id,
        operation_status,
        operation_details,
        affected_rows
    ) VALUES (
        'delete',
        'topic',
        OLD.id,
        'success',
        CONCAT('删除主题: ', OLD.display_name),
        1
    );
END//

-- 实体验证触发器
CREATE TRIGGER yearly_stats_before_insert
BEFORE INSERT ON yearly_stats
FOR EACH ROW
BEGIN
    DECLARE entity_exists INT;
    
    CASE NEW.entity_type
        WHEN 'author' THEN
            SELECT COUNT(*) INTO entity_exists FROM authors WHERE id = NEW.entity_id;
        WHEN 'concept' THEN
            SELECT COUNT(*) INTO entity_exists FROM concepts WHERE id = NEW.entity_id;
        WHEN 'institution' THEN
            SELECT COUNT(*) INTO entity_exists FROM institutions WHERE id = NEW.entity_id;
        WHEN 'source' THEN
            SELECT COUNT(*) INTO entity_exists FROM sources WHERE id = NEW.entity_id;
    END CASE;
    
    IF entity_exists = 0 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Invalid entity reference in yearly_stats';
    END IF;
END//

CREATE TRIGGER yearly_stats_before_update
BEFORE UPDATE ON yearly_stats
FOR EACH ROW
BEGIN
    DECLARE entity_exists INT;
    
    CASE NEW.entity_type
        WHEN 'author' THEN
            SELECT COUNT(*) INTO entity_exists FROM authors WHERE id = NEW.entity_id;
        WHEN 'concept' THEN
            SELECT COUNT(*) INTO entity_exists FROM concepts WHERE id = NEW.entity_id;
        WHEN 'institution' THEN
            SELECT COUNT(*) INTO entity_exists FROM institutions WHERE id = NEW.entity_id;
        WHEN 'source' THEN
            SELECT COUNT(*) INTO entity_exists FROM sources WHERE id = NEW.entity_id;
    END CASE;
    
    IF entity_exists = 0 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Invalid entity reference in yearly_stats';
    END IF;
END//

CREATE TRIGGER search_results_before_insert
BEFORE INSERT ON search_results
FOR EACH ROW
BEGIN
    DECLARE entity_exists INT;
    
    CASE NEW.entity_type
        WHEN 'work' THEN
            SELECT COUNT(*) INTO entity_exists FROM works WHERE id = NEW.entity_id;
        WHEN 'author' THEN
            SELECT COUNT(*) INTO entity_exists FROM authors WHERE id = NEW.entity_id;
        WHEN 'concept' THEN
            SELECT COUNT(*) INTO entity_exists FROM concepts WHERE id = NEW.entity_id;
        WHEN 'institution' THEN
            SELECT COUNT(*) INTO entity_exists FROM institutions WHERE id = NEW.entity_id;
        WHEN 'source' THEN
            SELECT COUNT(*) INTO entity_exists FROM sources WHERE id = NEW.entity_id;
        WHEN 'topic' THEN
            SELECT COUNT(*) INTO entity_exists FROM topics WHERE id = NEW.entity_id;
    END CASE;
    
    IF entity_exists = 0 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Invalid entity reference in search_results';
    END IF;
END//

CREATE TRIGGER search_results_before_update
BEFORE UPDATE ON search_results
FOR EACH ROW
BEGIN
    DECLARE entity_exists INT;
    
    CASE NEW.entity_type
        WHEN 'work' THEN
            SELECT COUNT(*) INTO entity_exists FROM works WHERE id = NEW.entity_id;
        WHEN 'author' THEN
            SELECT COUNT(*) INTO entity_exists FROM authors WHERE id = NEW.entity_id;
        WHEN 'concept' THEN
            SELECT COUNT(*) INTO entity_exists FROM concepts WHERE id = NEW.entity_id;
        WHEN 'institution' THEN
            SELECT COUNT(*) INTO entity_exists FROM institutions WHERE id = NEW.entity_id;
        WHEN 'source' THEN
            SELECT COUNT(*) INTO entity_exists FROM sources WHERE id = NEW.entity_id;
        WHEN 'topic' THEN
            SELECT COUNT(*) INTO entity_exists FROM topics WHERE id = NEW.entity_id;
    END CASE;
    
    IF entity_exists = 0 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Invalid entity reference in search_results';
    END IF;
END//

-- 搜索会话表触发器
CREATE TRIGGER search_sessions_after_insert
AFTER INSERT ON search_sessions
FOR EACH ROW
BEGIN
    INSERT INTO operation_logs (
        operation_type,
        entity_type,
        entity_id,
        operation_status,
        operation_details,
        affected_rows
    ) VALUES (
        'search',
        'search_session',
        NEW.session_id,
        'success',
        CONCAT('新增搜索会话: ', NEW.query_text),
        1
    );
END//

-- 搜索结果表触发器
CREATE TRIGGER search_results_after_insert
AFTER INSERT ON search_results
FOR EACH ROW
BEGIN
    INSERT INTO operation_logs (
        operation_type,
        entity_type,
        entity_id,
        operation_status,
        operation_details,
        affected_rows
    ) VALUES (
        'search',
        'search_result',
        NEW.entity_id,
        'success',
        CONCAT('新增搜索结果: 会话ID=', NEW.session_id, ', 实体ID=', NEW.entity_id),
        1
    );
END//

CREATE TRIGGER search_results_after_update
AFTER UPDATE ON search_results
FOR EACH ROW
BEGIN
    -- 记录点击行为
    IF NEW.is_clicked != OLD.is_clicked AND NEW.is_clicked = TRUE THEN
        INSERT INTO user_behaviors (
            session_id,
            document_id,
            dwell_time,
            behavior_time
        ) VALUES (
            NEW.session_id,
            NEW.entity_id,
            NEW.dwell_time,
            NEW.click_time  -- 用点击时间
        )
        ON DUPLICATE KEY UPDATE
            dwell_time = VALUES(dwell_time),
            behavior_time = VALUES(behavior_time);
    END IF;
    
    -- 记录停留时间
    IF NEW.dwell_time != OLD.dwell_time AND NEW.dwell_time > 0 THEN
        INSERT INTO user_behaviors (
            session_id,
            document_id,
            dwell_time,
            behavior_time
        ) VALUES (
            NEW.session_id,
            NEW.entity_id,
            NEW.dwell_time,
            NOW()  -- 用当前时间
        )
        ON DUPLICATE KEY UPDATE
            dwell_time = VALUES(dwell_time),
            behavior_time = VALUES(behavior_time);
    END IF;
END//

DELIMITER ;