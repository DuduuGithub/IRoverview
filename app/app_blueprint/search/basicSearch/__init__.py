from flask import Blueprint

# 创建搜索蓝图
search_bp = Blueprint('search', __name__,
                      template_folder='templates',
                      static_folder='static')

from . import routes  # 导入路由 