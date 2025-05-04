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
from Database.config import db
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

reader_bp = Blueprint('reader', __name__, url_prefix='/reader')

