# 搜索系统数据库集成

## 概述

本项目实现了一个将信息检索系统与数据库集成的解决方案，将原有基于文件系统的倒排索引检索改造为支持数据库检索的系统。主要修改了以下三个核心模块：

1. **基本搜索模块** (`basicSearch/search.py`)
2. **高级搜索模块** (`proSearch/search.py`) 
3. **排序模块** (`rank/rank.py`)

## 主要功能

### 基本搜索模块

基本搜索模块现在支持以下功能：

- 从数据库中检索文档
- 支持基于Term的简单查询
- 支持布尔查询 (AND, OR, NOT)
- 保留了原有的二分查找和跳表优化
- 支持多种排序方式

### 高级搜索模块

高级搜索模块支持以下功能：

- 支持多字段查询，包括标题、作者、关键词、内容和时间
- 支持时间范围查询
- 支持字段查询和普通查询的组合
- 支持自动根据查询类型选择合适的排序方式

### 排序模块

排序模块提供以下功能：

- 相关性排序 (TF-IDF)
- 时间排序（升序和降序）
- 组合排序（结合相关性和时间因素）
- 支持数据库模式下的排序

## 数据库集成

系统集成了以下数据库模型：

- `Work`: 作品/文献
- `Author`: 作者
- `Concept`: 概念
- `Topic`: 主题
- `WorkAuthorship`: 作品-作者关联
- `WorkConcept`: 作品-概念关联
- `WorkTopic`: 作品-主题关联

搜索过程充分利用了数据库的关联查询能力，例如：

- 通过作者查询相关文献
- 通过概念和主题查询相关文献
- 通过时间范围过滤文献

## 如何使用

### 基本搜索

```python
from basicSearch.search import search as basic_search

# 使用数据库模式的基本搜索
results = basic_search(query_text="人工智能", use_db=True, sort_method="relevance")
```

### 高级搜索

```python
from proSearch.search import search as advanced_search

# 使用字段查询
results = advanced_search(query_text="title:人工智能 author:李德毅", use_db=True)

# 使用时间范围
results = advanced_search(query_text="time:2020~2023 深度学习", use_db=True)
```

### 测试脚本

提供了测试脚本 `test_db_search.py` 用于测试数据库搜索功能，包含：

- 示例数据生成
- 基本搜索测试
- 高级搜索测试
- 排序功能测试

运行测试脚本：

```bash
cd IRoverview/app/app_blueprint/search
python test_db_search.py
```

## 扩展性

系统设计保持了高扩展性：

1. 同时支持基于文件和基于数据库的搜索
2. 易于添加新的排序方式
3. 可以扩展查询字段和类型
4. 可以优化数据库查询性能

## 未来改进

1. 添加全文搜索引擎支持 (如 Elasticsearch)
2. 提高大规模数据的查询效率
3. 添加更多语义搜索特性
4. 实现更高级的排序算法
5. 添加用户个性化搜索结果支持 