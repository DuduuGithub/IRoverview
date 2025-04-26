# 分析页面 暂定的基本功能中不需要用到此页面


from flask import Blueprint, jsonify, render_template, request, make_response
from utils import *

from flask_login import current_user
from sqlalchemy.sql import func
from io import StringIO
import csv
import logging

analysis_bp = Blueprint('analysis', __name__, url_prefix='/analysis')

@analysis_bp.route('/')
def index():
    """
    渲染分析主页面
    """
    try:
        return render_template('analysis/index.html')
    except Exception as e:
        logging.error(f"Error rendering analysis template: {str(e)}")
        return f"Error: {str(e)}", 500
