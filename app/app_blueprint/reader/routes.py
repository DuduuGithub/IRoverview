from flask import Blueprint, render_template, request, jsonify, redirect, url_for
from Database.model import Work, Author, WorkAuthorship, SearchResult, YearlyStat, WorkReferencedWork, WorkLocation, Source, WorkConcept, Concept, WorkRelatedWork
from Database.config import db
from ..search.search_utils import (
    record_search_session,
    record_search_results,
    record_document_click,
    record_dwell_time,
    calculate_relevance_score,
    update_search_result_score
)
from ..search.searcher import convert_abstract_to_text
import sys
import json
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
        landing_page_url = ""
        pdf_url = ""
        work_location = WorkLocation.query.filter_by(work_id=doc_id).first()
        if work_location and work_location.source_id:
            source = Source.query.get(work_location.source_id)
            if source and source.display_name:
                venue_name = source.display_name
                # 尝试从来源名称中提取出版商信息
                if "," in source.display_name:
                    publisher = source.display_name.split(",")[-1].strip()
            # 获取落地页URL
            if work_location.landing_page_url:
                landing_page_url = work_location.landing_page_url
            # 获取PDF URL
            if work_location.pdf_url:
                pdf_url = work_location.pdf_url
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
                    'abstract': convert_abstract_to_text(ref_work.abstract_inverted_index),
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
        
        # 获取论文相关文献
        related_works_ids = [item.related_work_id for item in WorkRelatedWork.query.filter_by(work_id=doc_id).all()]
        related_works = []
        
        print(f"[INFO] 找到{len(related_works_ids)}篇相关文献")
        
        # 从Work表中获取相关文献的详细信息
        for rel_id in related_works_ids:
            rel_work = Work.query.get(rel_id)
            if rel_work:
                # 构建相关文献信息字典
                rel_info = {
                    'id': rel_work.id,
                    'title': rel_work.title,
                    'abstract': convert_abstract_to_text(rel_work.abstract_inverted_index),
                    'publication_year': rel_work.publication_year,
                    'cited_by_count': rel_work.cited_by_count,
                    'doi': rel_work.doi,
                    'language': rel_work.language,
                    'type': rel_work.type
                }
                
                # 获取相关文献的作者
                rel_authors = []
                rel_authorships = WorkAuthorship.query.filter_by(work_id=rel_id).all()
                for rel_authorship in rel_authorships:
                    if rel_authorship.author_id:
                        rel_author = Author.query.get(rel_authorship.author_id)
                        if rel_author:
                            rel_authors.append(rel_author.display_name)
                
                rel_info['authors'] = rel_authors
                
                # 获取相关文献的来源信息
                rel_venue = "未知来源"
                rel_location = WorkLocation.query.filter_by(work_id=rel_id).first()
                if rel_location and rel_location.source_id:
                    rel_source = Source.query.get(rel_location.source_id)
                    if rel_source and rel_source.display_name:
                        rel_venue = rel_source.display_name
                
                rel_info['venue'] = rel_venue
                
                # 添加到相关文献列表
                related_works.append(rel_info)
        
        print(f"[INFO] 成功获取到{len(related_works)}篇相关文献的详细信息")
        
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
            'pages': f"{work.first_page}-{work.last_page}" if work and hasattr(work, 'first_page') and hasattr(work, 'last_page') and work.first_page and work.last_page else "",
            'landing_page_url': landing_page_url,
            'pdf_url': pdf_url
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
                    
        # 处理work对象的摘要
        if work and work.abstract_inverted_index:
            work.abstract_inverted_index = convert_abstract_to_text(work.abstract_inverted_index)
                    
        return render_template('reader/document_detail.html', 
                             work=work,
                             authors=authors,
                             message=message,
                             document_data=document_data,
                             citation_data=citation_data,
                             venue_name=venue_name,
                             concept_names=concept_names,
                             referenced_works=referenced_works,
                             related_works=related_works)
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
            
        # 设置最大节点数限制
        MAX_NODES = 50
            
        # 使用一次查询获取直接引用关系
        direct_citations = db.session.query(
            WorkReferencedWork.work_id,
            WorkReferencedWork.referenced_work_id
        ).filter(
            db.or_(
                WorkReferencedWork.work_id == doc_id,
                WorkReferencedWork.referenced_work_id == doc_id
            )
        ).all()
        
        # 收集相关论文ID
        paper_ids = set([doc_id])
        reference_ids = set()  # 存储焦点论文的参考文献ID
        citing_paper_ids = set()  # 存储引用焦点论文的论文ID（b集合）
        
        for work_id, ref_id in direct_citations:
            paper_ids.add(work_id)
            paper_ids.add(ref_id)
            # 如果这是焦点论文的参考文献，加入到reference_ids
            if work_id == doc_id:
                reference_ids.add(ref_id)
            # 如果这是引用焦点论文的论文，加入到citing_paper_ids
            if ref_id == doc_id:
                citing_paper_ids.add(work_id)
                
        # 查找所有引用参考文献的论文
        if reference_ids:
            citations_to_references = db.session.query(
                WorkReferencedWork.work_id,
                WorkReferencedWork.referenced_work_id
            ).filter(
                WorkReferencedWork.referenced_work_id.in_(reference_ids)
            ).all()
            
            # 添加引用参考文献的论文
            for work_id, ref_id in citations_to_references:
                paper_ids.add(work_id)
                
        # 查找b集合中论文的所有参考文献
        if citing_paper_ids:
            references_of_citing = db.session.query(
                WorkReferencedWork.work_id,
                WorkReferencedWork.referenced_work_id
            ).filter(
                WorkReferencedWork.work_id.in_(citing_paper_ids)
            ).all()
            
            # 添加b集合论文的参考文献
            for work_id, ref_id in references_of_citing:
                paper_ids.add(ref_id)
        
        # 限制节点数量
        if len(paper_ids) > MAX_NODES:
            # 获取引用次数最多的论文
            top_papers = db.session.query(Work.id)\
                .filter(Work.id.in_(paper_ids))\
                .order_by(Work.cited_by_count.desc())\
                .limit(MAX_NODES)\
                .all()
            paper_ids = set([p[0] for p in top_papers])
            # 确保当前论文在集合中
            paper_ids.add(doc_id)
        
        # 批量获取论文信息
        papers_data = db.session.query(
            Work.id,
            Work.title,
            Work.cited_by_count
        ).filter(
            Work.id.in_(paper_ids)
        ).all()
        
        # 准备节点数据
        papers = [{
            'id': p.id,
            'title': p.title,
            'citations': p.cited_by_count if p.cited_by_count else 5
        } for p in papers_data]
        
        # 批量获取这些论文之间的引用关系
        citations_data = db.session.query(
            WorkReferencedWork.work_id,
            WorkReferencedWork.referenced_work_id
        ).filter(
            WorkReferencedWork.work_id.in_(paper_ids),
            WorkReferencedWork.referenced_work_id.in_(paper_ids)
        ).all()
        
        # 准备边数据
        citations = [{
            'source': work_id,
            'target': ref_id
        } for work_id, ref_id in citations_data]
        
        print(f"[INFO] 引用关系图: 找到 {len(papers)} 个节点和 {len(citations)} 条边")
        
        return render_template('reader/citation_network.html', 
                              papers=papers,
                              citations=citations,
                              current_doc_id=doc_id)
                              
    except Exception as e:
        import traceback
        print(f"[ERROR] 生成引用关系图失败: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500