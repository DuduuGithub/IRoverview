# 文献详情页 留下了一个与页面前端的数据传输页

from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for, abort
import sys
import os
import requests
import json
from flask import current_app
from openai import OpenAI

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Database.model import (
    DocumentDisplayView
)
from Database.config import db
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

reader_bp = Blueprint('reader', __name__, url_prefix='/reader')

def validate_document_access(doc_id):
    """验证文档访问权限"""
    doc = DocumentDisplayView.query.get_or_404(doc_id)
    if not doc:
        abort(404, description="Document not found")
    return doc


@reader_bp.route('/document/<doc_id>')
def document(doc_id):
    try:
        doc = validate_document_access(doc_id)
        
         # 检查图片文件是否存在
        png_file = os.path.join(current_app.static_folder, 'images', 'documents', f'doc_img_{doc_id}.png')
        jpg_file = os.path.join(current_app.static_folder, 'images', 'documents', f'doc_img_{doc_id}.jpg')
        
        png_exists = os.path.exists(png_file)
        jpg_exists = os.path.exists(jpg_file)
        
        picture_path=None
        if png_exists:
            picture_path='images/documents/doc_img_'+doc_id+'.png'
        elif jpg_exists:
            picture_path='images/documents/doc_img_'+doc_id+'.jpg'
        
        # 获取关键词
        keywords = DocKeywords.query.filter_by(Doc_id=doc_id).all()
        
        # 获取契约人信息
        contractors_info = Contractors.query.filter_by(Doc_id=doc_id).first()
        contractors = []
        relation = None
        if contractors_info:
            alice = People.query.get(contractors_info.Alice_id)
            bob = People.query.get(contractors_info.Bob_id)
            if alice and bob:
                contractors = [
                    {"name": alice.Person_name},
                    {"name": bob.Person_name}
                ]
                # 获取关系
                relation = Relations.query.filter_by(
                    Alice_id=contractors_info.Alice_id,
                    Bob_id=contractors_info.Bob_id
                ).first()
                if relation:
                    relation = relation.Relation_type
        
        # 获取参人信息
        participants_query = db.session.query(
            Participants, People
        ).join(
            People, Participants.Person_id == People.Person_id
        ).filter(
            Participants.Doc_id == doc_id
        ).all()
        
        participants = [
            {"name": person.Person_name, "role": participant.Part_role}
            for participant, person in participants_query
        ]
        
        # 从 URL 参数获取 from_page 和 folder_id
        from_page = request.args.get('from_page', default=None)
        folder_id = request.args.get('folder_id', default=None)
        # 准备模板数据
        data = {
            'document': doc,
            'content': doc.Doc_originalText,
            'keywords': [{"KeyWord": kw.KeyWord} for kw in keywords],
            'contractors': contractors,
            'relation': relation,
            'participants': participants,
            'highlights': [],
            'notes': [],
            'comments': [],
            'evernotes': [],
            'corrections': [],  # 添加纠错记录列表
            'from_page': from_page,  # 添加 from_page 参数
            'folder_id': folder_id,   # 添加 folder_id 参数（如果存在）
            'picture_path':picture_path
        }
        
        if current_user.is_authenticated:
            # 获取用户相关数据
            data['highlights'] = Highlights.query.filter_by(
                Doc_id=doc_id,
                User_id=current_user.User_id
            ).all()
            
            # 获取批注
            data['notes'] = [{
                'id': note.Note_id,
                'content': note.Note_annotationText,
                'created_at': note.Note_createdAt.strftime('%Y-%m-%d %H:%M:%S')
            } for note in Notes.query.filter_by(
                Doc_id=doc_id,
                User_id=current_user.User_id
            ).all()]
            
            # 获取笔记
            evernotes = Evernote.query.filter_by(
                Doc_id=doc_id,
                User_id=current_user.User_id
            ).order_by(Evernote.Evernote_viewedAt.desc()).all()
            
            data['evernotes'] = [{
                'Evernote_id': note.Evernote_id,
                'Evernote_text': note.Evernote_text,
                'created_at': note.Evernote_viewedAt.strftime('%Y-%m-%d %H:%M:%S')
            } for note in evernotes]
            
            # 获取评论
            comments = Comments.query.filter_by(
                Doc_id=doc_id
            ).order_by(Comments.Comment_createdAt.desc()).all()
            
            data['comments'] = [{
                'id': comment.Comment_id,
                'content': comment.Comment_text,
                'user_name': Users.query.get(comment.User_id).User_name if comment.User_id else '匿名用户',
                'created_at': comment.Comment_createdAt.strftime('%Y-%m-%d %H:%M:%S'),
                'is_mine': comment.User_id == current_user.User_id
            } for comment in comments]
            
            # 获取错记录
            corrections = Corrections.query.filter_by(Doc_id=doc_id).all()
            data['corrections'] = [{
                'correction_id': c.Correction_id,
                'user_id': str(c.User_id),
                'user_name': Users.query.get(c.User_id).User_name if c.User_id else '系统',
                'text': c.Correction_text,
                'created_at': c.Correction_createdAt.strftime('%Y-%m-%d %H:%M:%S')
            } for c in corrections]
        
        return render_template('reader/document.html', **data)
        
    except Exception as e:
        logger.error(f"Error in document route: {str(e)}")
        print("*"*40)
        print(f"Error in document route: {str(e)}")
        flash('获取文档失败，请稍后再试', 'error')
        return redirect(url_for('searcher.index'))