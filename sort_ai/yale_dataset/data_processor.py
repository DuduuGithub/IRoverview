import os
import sys
import json
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils import generate_search_query, get_search_session_data

class DataProcessor:
    def __init__(self, data_dir='data'):
        """
        初始化数据处理器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')
        
        # 创建必要的目录
        for dir_path in [self.data_dir, self.raw_dir, self.processed_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def collect_session_data(self, min_dwell_time=30):
        """
        收集会话数据并保存到原始数据目录
        
        Args:
            min_dwell_time: 最小停留时间（秒）
        """
        # 获取会话数据
        session_data = get_search_session_data(min_dwell_time)
        
        if not session_data:
            print("未找到符合条件的会话数据")
            return
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(self.raw_dir, f'session_data_{timestamp}.json')
        
        # 保存原始数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        
        print(f"原始会话数据已保存到: {output_file}")
        return output_file

    def generate_training_data(self, input_file, api_key=None):
        """
        处理原始数据并生成训练数据
        
        Args:
            input_file: 输入文件路径
            api_key: OpenAI API密钥（可选）
        """
        # 读取原始数据
        with open(input_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        training_data = []
        
        # 处理每个会话
        for session in session_data:
            session_id = session['session_id']
            original_query = session['original_query']
            
            # 处理每个点击
            for click in session['clicks']:
                # 使用AI生成检索式
                generated_query = generate_search_query(click['content'], api_key)
                if not generated_query:
                    continue
                
                # 构建训练样本
                training_sample = {
                    'session_id': session_id,
                    'original_query': original_query,
                    'generated_query': generated_query,
                    'document_id': click['document_id'],
                    'rank_position': click['rank_position'],
                    'dwell_time': click['dwell_time'],
                    'click_order': click['click_order']
                }
                
                training_data.append(training_sample)
        
        if not training_data:
            print("未生成任何训练数据")
            return
        
        # 生成输出文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(self.processed_dir, f'training_data_{timestamp}.json')
        
        # 保存训练数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        print(f"训练数据已保存到: {output_file}")
        return output_file

def main():
    """
    主函数，用于生成训练数据
    """
    # 配置参数
    api_key = os.getenv('OPENAI_API_KEY')
    min_dwell_time = 30
    
    try:
        # 创建数据处理器
        processor = DataProcessor()
        
        # 收集会话数据
        raw_data_file = processor.collect_session_data(min_dwell_time)
        if not raw_data_file:
            return
        
        # 生成训练数据
        training_data_file = processor.generate_training_data(raw_data_file, api_key)
        if training_data_file:
            print("数据处理完成！")
    
    except Exception as e:
        print(f"数据处理出错: {str(e)}")

if __name__ == '__main__':
    main() 