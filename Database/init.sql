-- 删除数据库（如果存在）
DROP DATABASE IF EXISTS iroverview;

-- 创建数据库
CREATE DATABASE iroverview
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

-- 使用数据库
USE iroverview;

-- 创建文献表
CREATE TABLE documents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    author VARCHAR(255),
    publish_date DATETIME,
    content TEXT,
    keywords VARCHAR(255)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 创建引用网络表
CREATE TABLE citation_network (
    citing_doc_id INT,
    cited_doc_id INT,
    citation_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (citing_doc_id, cited_doc_id),
    FOREIGN KEY (citing_doc_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY (cited_doc_id) REFERENCES documents(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 创建搜索会话表
CREATE TABLE search_sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    search_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    keyword VARCHAR(255),
    title_query VARCHAR(255),
    author_query VARCHAR(255),
    date_from DATETIME,
    date_to DATETIME,
    search_type VARCHAR(20),
    total_results INT,
    INDEX idx_session_id (session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 创建搜索结果表
CREATE TABLE search_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(50),
    document_id INT,
    rank_position INT,
    is_clicked BOOLEAN DEFAULT FALSE,
    click_time DATETIME,
    click_order INT,
    dwell_time INT,
    FOREIGN KEY (session_id) REFERENCES search_sessions(session_id),
    FOREIGN KEY (document_id) REFERENCES documents(id),
    INDEX idx_session_document (session_id, document_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci; 