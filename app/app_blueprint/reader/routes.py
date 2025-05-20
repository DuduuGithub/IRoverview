from flask import Blueprint, render_template, request, jsonify, redirect, url_for
from Database.model import Work, Author, WorkAuthorship, SearchResult, YearlyStat, WorkReferencedWork, WorkLocation, Source, WorkConcept, Concept
from Database.config import db
from ..search.search_utils import (
    record_search_session,
    record_search_results,
    record_document_click,
    record_dwell_time,
    calculate_relevance_score,
    update_search_result_score
)
import sys
# 创建蓝图
reader_bp = Blueprint('reader', __name__,
                     template_folder='templates',
                     static_folder='static')

@reader_bp.route('/document/<doc_id>')
def document_detail(doc_id):
    try:
        # 获取session_id
        session_id = request.args.get('session_id')
        print(f"[INFO] 访问文档详情页: doc_id={doc_id}, session_id={session_id}")
        
        work = Work.query.get(doc_id)
        if work:
            print(f"[INFO] 查到文档: {work.id} - {work.title}")
            message = f"查到文档: {work.id} - {work.title}"
        else:
            print(f"[WARN] 没查到文档: {doc_id}")
            message = f"没有查到文档: {doc_id}"
            
        # 获取作者
        authorships = WorkAuthorship.query.filter_by(work_id=doc_id).all()
        authors = []
        for authorship in authorships:
            if authorship.author_id:
                author = Author.query.get(authorship.author_id)
                if author:
                    authors.append(author.display_name)
        
        # 获取论文来源信息
        venue_name = "未知来源"
        publisher = ""
        work_location = WorkLocation.query.filter_by(work_id=doc_id).first()
        if work_location and work_location.source_id:
            source = Source.query.get(work_location.source_id)
            if source and source.display_name:
                venue_name = source.display_name
                # 尝试从来源名称中提取出版商信息
                if "," in source.display_name:
                    publisher = source.display_name.split(",")[-1].strip()
        else:
            print(f"[INFO] 未找到文档来源信息")
            
        # 获取论文相关概念
        work_concepts = WorkConcept.query.filter_by(work_id=doc_id).all()
        concepts_data = []
        for work_concept in work_concepts:
            concept = Concept.query.get(work_concept.concept_id)
            # 只保留level小于等于3的概念
            if concept and (concept.level is None or concept.level <= 3):
                concepts_data.append({
                    'concept_id': concept.id,
                    'name': concept.display_name,
                    'score': work_concept.score
                })
        
        # 按score从大到小排序
        concepts_data.sort(key=lambda x: x['score'] if x['score'] is not None else 0, reverse=True)
        
        # 获取排序后的概念名称列表
        concept_names = [item['name'] for item in concepts_data]
        
        print(f"[INFO] 文档相关概念(level<=3): {concept_names}")
        
        # 获取论文参考文献
        referenced_works_ids = [item.referenced_work_id for item in WorkReferencedWork.query.filter_by(work_id=doc_id).all()]
        referenced_works = []
        
        print(f"[INFO] 找到{len(referenced_works_ids)}篇参考文献")
        
        # 从Work表中获取参考文献的详细信息
        for ref_id in referenced_works_ids:
            ref_work = Work.query.get(ref_id)
            if ref_work:
                # 构建参考文献信息字典
                ref_info = {
                    'id': ref_work.id,
                    'title': ref_work.title,
                    'abstract': ref_work.abstract_inverted_index,
                    'publication_year': ref_work.publication_year,
                    'cited_by_count': ref_work.cited_by_count,
                    'doi': ref_work.doi,
                    'language': ref_work.language,
                    'type': ref_work.type
                }
                
                # 获取参考文献的作者
                ref_authors = []
                ref_authorships = WorkAuthorship.query.filter_by(work_id=ref_id).all()
                for ref_authorship in ref_authorships:
                    if ref_authorship.author_id:
                        ref_author = Author.query.get(ref_authorship.author_id)
                        if ref_author:
                            ref_authors.append(ref_author.display_name)
                
                ref_info['authors'] = ref_authors
                
                # 获取参考文献的来源信息
                ref_venue = "未知来源"
                ref_location = WorkLocation.query.filter_by(work_id=ref_id).first()
                if ref_location and ref_location.source_id:
                    ref_source = Source.query.get(ref_location.source_id)
                    if ref_source and ref_source.display_name:
                        ref_venue = ref_source.display_name
                
                ref_info['venue'] = ref_venue
                
                # 添加到参考文献列表
                referenced_works.append(ref_info)
        
        print(f"[INFO] 成功获取到{len(referenced_works)}篇参考文献的详细信息")
        
        # 构建文档数据，包含用于生成引用格式的所有信息
        document_data = {
            'id': work.id if work else None,
            'title': work.title if work else None,
            'authors': authors,
            'session_id': session_id,
            'venue_name': venue_name,
            'publication_year': work.publication_year if work else None,
            'cited_by_count': work.cited_by_count if work and hasattr(work, 'cited_by_count') else 0,
            'type': work.type if work else "article",
            'doi': work.doi if work and work.doi else "",
            'publisher': publisher,
            'volume': work.volume if work and hasattr(work, 'volume') else "",
            'issue': work.issue if work and hasattr(work, 'issue') else "",
            'pages': f"{work.first_page}-{work.last_page}" if work and hasattr(work, 'first_page') and hasattr(work, 'last_page') and work.first_page and work.last_page else ""
        }

        # 获取年度引用统计数据
        yearly_citations = YearlyStat.query.filter_by(
            entity_id=doc_id#,
            # entity_type='work'
        ).order_by(YearlyStat.year).all()
        
        print(f"[DEBUG] 查询到的年度引用数据数量: {len(yearly_citations)}")
        for citation in yearly_citations:
            print(f"[DEBUG] 年份: {citation.year}, 引用次数: {citation.cited_by_count}")
        
        # 处理年度引用数据，补全缺失年份
        if yearly_citations:
            min_year = min(stat.year for stat in yearly_citations)
            max_year = max(stat.year for stat in yearly_citations)
            
            # 创建完整的年份序列和对应的引用数
            complete_years = list(range(min_year, max_year + 1))
            existing_citations = {stat.year: stat.cited_by_count for stat in yearly_citations}
            
            citation_data = {
                'years': complete_years,
                'citations': [existing_citations.get(year, 0) for year in complete_years]
            }
            print(f"[DEBUG] 处理后的数据: {citation_data}")
        else:
            citation_data = {
                'years': [],
                'citations': []
            }
            print("[DEBUG] 没有找到年度引用数据")
                    
        return render_template('reader/document_detail.html', 
                             work=work,
                             authors=authors,
                             message=message,
                             document_data=document_data,
                             citation_data=citation_data,
                             venue_name=venue_name,
                             concept_names=concept_names,
                             referenced_works=referenced_works)
    except Exception as e:
        print(f"[ERROR] 查询文档失败: {e}")
        return jsonify({'error': str(e)}), 500

@reader_bp.route('/api/record-dwell-time', methods=['POST'])
def record_dwell_time():
    """记录文档停留时间"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        document_id = data.get('document_id')
        dwell_time = data.get('dwell_time')
        
        if not all([session_id, document_id, dwell_time]):
            return jsonify({'error': '缺少必要参数'}), 400
            
        # 记录停留时间
        if record_dwell_time(session_id, document_id, dwell_time):
            return jsonify({'message': '停留时间记录成功'})
        else:
            return jsonify({'error': '记录停留时间失败'}), 500
        
    except Exception as e:
        print(f"记录停留时间出错: {str(e)}")
        return jsonify({'error': '记录停留时间失败'}), 500

@reader_bp.route('/citation-network/<doc_id>')
def citation_network(doc_id):
    """渲染论文引用关系图"""
    try:
        # 获取当前论文
        work = Work.query.get(doc_id)
        if not work:
            return jsonify({'error': '未找到文档'}), 404
            
        # 从数据库中查询引用关系
        # 获取当前论文引用的论文（从属引用关系）
        referenced_works_ids = WorkReferencedWork.query.filter_by(work_id=doc_id).all()
        referenced_works_ids = [row.referenced_work_id for row in referenced_works_ids]
        
        # 获取引用当前论文的论文（被引用关系）
        citing_works_ids = WorkReferencedWork.query.filter_by(referenced_work_id=doc_id).all()
        citing_works_ids = [row.work_id for row in citing_works_ids]
        
        # 准备节点数据
        papers = []
        paper_ids = set()
        
        # 添加当前论文
        papers.append({
            'id': work.id,
            'title': work.title,
            'citations': work.cited_by_count if hasattr(work, 'cited_by_count') else 10
        })
        paper_ids.add(work.id)
        
        # 添加当前论文引用的论文
        for ref_id in referenced_works_ids:
            if ref_id not in paper_ids:
                ref_work = Work.query.get(ref_id)
                if ref_work:
                    papers.append({
                        'id': ref_work.id,
                        'title': ref_work.title,
                        'citations': ref_work.cited_by_count if hasattr(ref_work, 'cited_by_count') else 5
                    })
                    paper_ids.add(ref_work.id)
        
        # 添加引用当前论文的论文
        for citing_id in citing_works_ids:
            if citing_id not in paper_ids:
                citing_work = Work.query.get(citing_id)
                if citing_work:
                    papers.append({
                        'id': citing_work.id,
                        'title': citing_work.title,
                        'citations': citing_work.cited_by_count if hasattr(citing_work, 'cited_by_count') else 5
                    })
                    paper_ids.add(citing_work.id)
        
        # 准备边数据（引用关系）
        citations = []
        
        # 当前论文 -> 引用的论文
        for ref_id in referenced_works_ids:
            citations.append({
                'source': work.id,
                'target': ref_id
            })
        
        # 引用当前论文的论文 -> 当前论文
        for citing_id in citing_works_ids:
            citations.append({
                'source': citing_id,
                'target': work.id
            })
        
        # 添加二级引用关系（可选：如果图太简单可以添加更多连接）
        # 在已有节点之间查找额外的引用关系
        for paper1_id in paper_ids:
            for paper2_id in paper_ids:
                if paper1_id != paper2_id and paper1_id != work.id and paper2_id != work.id:
                    # 检查paper1是否引用了paper2
                    ref_exists = WorkReferencedWork.query.filter_by(work_id=paper1_id, 
                                                                 referenced_work_id=paper2_id).first()
                    if ref_exists:
                        citations.append({
                            'source': paper1_id,
                            'target': paper2_id
                        })
        
        print(f"[INFO] 引用关系图: 找到 {len(papers)} 个节点和 {len(citations)} 条边")
        
        return render_template('reader/citation_network.html', 
                              papers=papers,
                              citations=citations,
                              current_doc_id=doc_id)  # 传递当前文档ID
    except Exception as e:
        import traceback
        print(f"[ERROR] 生成引用关系图失败: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500



def format_authors(authors):
    if not authors:
        return ""
    formatted = []
    for author in authors:
        parts = author.strip().split()
        if len(parts) >= 2:
            last = parts[-1]
            initials = " ".join([p[0] + "." for p in parts[:-1]])
            formatted.append(f"{last}, {initials}")
        else:
            formatted.append(author)
    if len(formatted) == 1:
        return formatted[0]
    elif len(formatted) <= 20:
        return ", ".join(formatted[:-1]) + ", & " + formatted[-1]
    else:
        return ", ".join(formatted[:19]) + ", ... " + formatted[-1]

def generate_apa_citation(info):
    type_ = info.get("type", "other")
    authors = format_authors(info.get("authors", []))
    year = info.get("publication_year", "n.d.")
    title = info.get("title", "[No title]")
    container = info.get("container", "")
    publisher = info.get("publisher", "")
    volume = info.get("volume", "")
    issue = info.get("issue", "")
    pages = info.get("pages", "")
    doi = info.get("doi", "")
    url = info.get("url", "")

    citation = ""

    if type_ in ['article', 'erratum', 'letter', 'editorial', 'peer-review']:
        # 期刊文章或类似文章
        citation = f"{authors} ({year}). {title}. *{container}*"
        if volume:
            citation += f", {volume}"
            if issue:
                citation += f"({issue})"
        if pages:
            citation += f", {pages}"
        citation += "."
    
    elif type_ == 'book':
        citation = f"{authors} ({year}). *{title}*. {publisher}."

    elif type_ == 'book-chapter':
        editors = format_authors(info.get("editors", []))
        book_title = info.get("book_title", "")
        chapter_pages = f"(pp. {pages})" if pages else ""
        citation = (
            f"{authors} ({year}). {title}. In {editors} (Ed.), *{book_title}* {chapter_pages}. {publisher}."
        )

    elif type_ == 'report':
        citation = f"{authors} ({year}). *{title}* (Report). {publisher}."

    elif type_ == 'dissertation':
        degree = info.get("degree", "Doctoral dissertation")
        institution = publisher or info.get("institution", "Unknown institution")
        citation = f"{authors} ({year}). *{title}* ({degree}). {institution}."

    elif type_ == 'dataset':
        citation = f"{authors} ({year}). *{title}* [Data set]. {publisher}."

    elif type_ == 'reference-entry':
        citation = f"{authors} ({year}). {title}. In *{container}*. {publisher}."

    elif type_ == 'standard':
        citation = f"{authors} ({year}). *{title}* (Standard). {publisher}."

    elif type_ == 'grant':
        citation = f"{authors} ({year}). *{title}* [Grant description]. {publisher}."

    elif type_ == 'paratext':
        citation = f"{authors} ({year}). *{title}*. {publisher}."

    else:  # fallback
        citation = f"{authors} ({year}). *{title}*. {publisher}."

    if doi:
        citation += f" https://doi.org/{doi.split('/')[-1]}"
    elif url:
        citation += f" {url}"

    return citation

def generate_apa_citation_agency(work_id):
    """根据work_id从数据库中获取信息并生成APA格式引用"""
    try:
        # 从数据库获取论文信息
        work = Work.query.get(work_id)
        if not work:
            return "引用信息不可用：找不到指定的文献"
        
        # 获取作者列表
        authors = []
        authorships = WorkAuthorship.query.filter_by(work_id=work_id).all()
        for authorship in authorships:
            if authorship.author_id:
                author = Author.query.get(authorship.author_id)
                if author:
                    authors.append(author.display_name)
        
        # 获取期刊/来源信息
        venue_name = "未知来源"
        publisher = ""
        work_location = WorkLocation.query.filter_by(work_id=work_id).first()
        if work_location and work_location.source_id:
            source = Source.query.get(work_location.source_id)
            if source and source.display_name:
                venue_name = source.display_name
                # 尝试从来源名称中提取出版商信息
                if "," in source.display_name:
                    publisher = source.display_name.split(",")[-1].strip()
        
        # 构建引用信息字典
        info = {
            "type": work.type if work.type else "article",
            "authors": authors,
            "title": work.title if work.title else "[无标题]",
            "publication_year": work.publication_year,
            "container": venue_name,
            "publisher": publisher,
            "volume": work.volume if hasattr(work, 'volume') and work.volume else "",
            "issue": work.issue if hasattr(work, 'issue') and work.issue else "",
            "pages": f"{work.first_page}-{work.last_page}" if hasattr(work, 'first_page') and hasattr(work, 'last_page') and work.first_page and work.last_page else "",
            "doi": work.doi if work.doi else ""
        }
        
        # 调用generate_apa_citation生成引用
        citation = generate_apa_citation(info)
        # 清理引用字符串，移除前后空白
        citation = citation.strip() if citation else ""
        return citation
        
    except Exception as e:
        print(f"[ERROR] 生成APA引用出错: {e}")
        return "生成引用时发生错误"

# example = {
#     "type": "book-chapter",
#     "authors": ["Alice Smith", "Bob Johnson"],
#     "editors": ["Jane Editor"],
#     "title": "Deep learning in genomics",
#     "book_title": "Advances in Genomic Research",
#     "publication_year": 2022,
#     "pages": "100-115",
#     "publisher": "Springer",
#     "doi": "10.1234/abcd.2022.001"
# }

# print(generate_apa_citation(example))
