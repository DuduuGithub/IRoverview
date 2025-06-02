import os
import sys

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from app import create_app # 导入create_app函数
from Database.model import db, Work # 导入db实例和Work模型

# 创建Flask应用实例
app = create_app()

if __name__ == '__main__':
    with app.app_context():
        try:
            # 查询works表中的记录数量
            work_count = db.session.query(Work).count()
            print(f"works 表中的记录数量: {work_count}")
            
            # 您也可以添加其他表来检查
            # from Database.model import Author
            # author_count = db.session.query(Author).count()
            # print(f"authors 表中的记录数量: {author_count}")
            
        except Exception as e:
            print(f"查询数据库时出错: {str(e)}") 