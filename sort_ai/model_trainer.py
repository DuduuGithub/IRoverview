import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datetime import datetime
import logging
from typing import List, Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import time
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sort_ai.model import IRRankingModel

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 减少其他库的日志输出
logging.getLogger('transformers').setLevel(logging.WARNING)

# 检查CUDA是否可用
logger.info(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA设备数量: {torch.cuda.device_count()}")
    logger.info(f"当前CUDA设备: {torch.cuda.current_device()}")
    logger.info(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
    # 强制使用CUDA
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# 设置随机种子以确保可重复性
def set_seed(seed=42):
    """设置所有随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 确保CUDA的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class SearchSessionDataset(Dataset):
    """搜索会话数据集"""
    
    def __init__(self, sessions: List[Dict]):
        """
        初始化数据集
        
        Args:
            sessions: 会话数据列表，每个会话包含查询和点击信息
        """
        self.sessions = sessions
        
    def __len__(self):
        return len(self.sessions)
        
    def __getitem__(self, idx):
        session = self.sessions[idx]
        
        # 将结构化查询转换为字符串表示
        query = session['query']
        
        return {
            'query': query,
            'pos_docs': session['pos_docs'],  # 现在是多个正样本的列表
            'neg_docs': session['neg_docs'],
            'session_id': session['session_id'],
            'pos_doc_ids': session['pos_doc_ids'],  # 现在是多个正样本ID的列表
            'neg_doc_ids': session['neg_doc_ids']
        }

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, bert_path=None):
        """
        初始化训练器
        
        Args:
            bert_path: BERT模型路径，如果为None则使用默认的本地bert_uncased路径
        """
        # 设置随机种子
        set_seed()
        
        # 检查是否可以使用CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 如果没有指定bert_path，使用本地bert_uncased路径
        if bert_path is None:
            # 获取sort_ai目录的路径
            sort_ai_dir = os.path.dirname(os.path.abspath(__file__))
            bert_path = os.path.join(sort_ai_dir, 'bert', 'D:/郭如璇的文件/A##Visual Studio Code/bert-base-uncased')
            logger.info(f"使用本地BERT模型: {bert_path}")
        
        self.model = IRRankingModel(bert_path=bert_path)
        # 将模型移动到GPU（如果可用）
        self.model = self.model.to(self.device)
        self.criterion = nn.MarginRankingLoss(margin=0.5)
        
        # 降低初始学习率
        self.initial_lr = 1e-5
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.initial_lr)
        
        # 添加学习率调度器
        self.num_warmup_steps = 0  # 会在train方法中设置
        self.num_training_steps = 0  # 会在train方法中设置
        
        # 确保模型保存目录存在（使用相对路径）
        self.sort_ai_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.sort_ai_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        
    def prepare_training_data(self, data_dir='training_data'):
        """准备训练数据"""
        logger.info("开始准备训练数据...")
        
        # 获取sort_ai目录的路径
        sort_ai_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(sort_ai_dir, data_dir)
        
        # 检查数据目录是否存在
        if not os.path.exists(data_dir):
            logger.error(f"训练数据目录 {data_dir} 不存在！")
            return [], []
        
        try:
            # 读取所有数据文件
            sessions_df = pd.read_csv(os.path.join(data_dir, 'sessions.csv'))
            documents_df = pd.read_csv(os.path.join(data_dir, 'documents.csv'))
            behaviors_df = pd.read_csv(os.path.join(data_dir, 'behaviors.csv'))
            results_df = pd.read_csv(os.path.join(data_dir, 'search_results.csv'))
            
            logger.info("已加载训练数据文件")
            
            # 准备训练样本
            sessions = []
            
            # 对每个会话处理
            for _, session_row in sessions_df.iterrows():
                session_id = session_row['session_id']
                
                # 获取该会话的用户行为
                session_behaviors = behaviors_df[behaviors_df['session_id'] == session_id]
                
                if len(session_behaviors) == 0:
                    continue
                
                # 获取点击的文档（正样本）
                clicked_docs = []
                for _, behavior in session_behaviors.iterrows():
                    if behavior['dwell_time'] > 30:  # 停留时间超过30秒视为正样本
                        doc = documents_df[documents_df['work_id'] == behavior['document_id']]
                        if not doc.empty:
                            clicked_docs.append((doc.iloc[0], behavior['document_id']))
                
                if not clicked_docs:
                    continue
                
                # 获取未点击的文档（负样本）
                session_results = results_df[results_df['session_id'] == session_id]
                unclicked_docs = []
                
                for _, result in session_results.iterrows():
                    if result['entity_id'] not in [doc_id for _, doc_id in clicked_docs]:
                        doc = documents_df[documents_df['work_id'] == result['entity_id']]
                        if not doc.empty:
                            unclicked_docs.append((doc.iloc[0], result['entity_id']))
                
                if not unclicked_docs:
                    continue
                
                def process_doc_text(doc):
                    """处理文档文本，确保类型安全"""
                    title = str(doc['title']) if pd.notna(doc['title']) else ''
                    abstract = str(doc['abstract_inverted_index']) if pd.notna(doc['abstract_inverted_index']) else ''
                    # 移除可能的特殊字符，只保留基本的标点和字母数字
                    title = ''.join(c for c in title if c.isalnum() or c.isspace() or c in '.,!?-')
                    abstract = ''.join(c for c in abstract if c.isalnum() or c.isspace() or c in '.,!?-')
                    return f"{title} {abstract}".strip()
                
                # 将所有正样本和负样本添加到会话中
                sessions.append({
                    'query': str(session_row['query_text']).strip(),
                    'pos_docs': [process_doc_text(doc) for doc, _ in clicked_docs],
                    'neg_docs': [process_doc_text(doc) for doc, _ in unclicked_docs],
                    'session_id': session_id,
                    'pos_doc_ids': [doc_id for _, doc_id in clicked_docs],
                    'neg_doc_ids': [doc_id for _, doc_id in unclicked_docs]
                })
                
                # 减少日志输出频率
                if len(sessions) % 10 == 0:
                    logger.info(f"已处理 {len(sessions)} 个会话")
            
            logger.info(f"数据准备完成，共生成 {len(sessions)} 个训练样本")
            
            if not sessions:
                logger.error("没有找到任何有效的训练样本！")
                return [], []
            
            # 使用固定的随机种子划分数据
            train_sessions, val_sessions = train_test_split(
                sessions, test_size=0.2, random_state=42
            )
            
            logger.info(f"准备完成，共有 {len(train_sessions)} 个训练样本，{len(val_sessions)} 个验证样本")
            
            return train_sessions, val_sessions
            
        except Exception as e:
            logger.error(f"准备训练数据时出错: {str(e)}")
            return [], []
    
    def collate_fn(self, batch):
        """自定义的批处理函数，处理不同长度的样本"""
        # 将所有样本的查询和会话ID收集到列表中
        queries = [item['query'] for item in batch]
        session_ids = [item['session_id'] for item in batch]
        
        # 收集所有正样本和负样本
        pos_docs_list = [item['pos_docs'] for item in batch]
        neg_docs_list = [item['neg_docs'] for item in batch]
        pos_doc_ids_list = [item['pos_doc_ids'] for item in batch]
        neg_doc_ids_list = [item['neg_doc_ids'] for item in batch]
        
        return {
            'query': queries,
            'pos_docs': pos_docs_list,
            'neg_docs': neg_docs_list,
            'session_id': session_ids,
            'pos_doc_ids': pos_doc_ids_list,
            'neg_doc_ids': neg_doc_ids_list
        }

    def get_scheduler(self, num_warmup_steps, num_training_steps):
        """获取学习率调度器"""
        from transformers import get_scheduler
        
        scheduler = get_scheduler(
            name="linear",  # 线性预热后线性衰减
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        return scheduler

    def train(self, train_sessions, val_sessions, epochs=30, batch_size=32):
        """训练模型"""
        logger.info("开始训练模型...")
        logger.info(f"训练集大小: {len(train_sessions)}, 验证集大小: {len(val_sessions)}")
        logger.info(f"批次大小: {batch_size}, 训练轮数: {epochs}")
        logger.info(f"初始学习率: {self.initial_lr}")
        
        train_dataset = SearchSessionDataset(train_sessions)
        generator = torch.Generator(device=self.device)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=4 if self.device.type == 'cuda' else 0,
            pin_memory=True if self.device.type == 'cuda' else False,
            generator=generator
        )
        
        # 设置学习率调度器参数
        num_update_steps_per_epoch = len(train_loader)
        self.num_training_steps = num_update_steps_per_epoch * epochs
        self.num_warmup_steps = self.num_training_steps // 10  # 预热步数为总步数的10%
        
        # 创建学习率调度器
        scheduler = self.get_scheduler(
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps
        )
        
        logger.info(f"总训练步数: {self.num_training_steps}")
        logger.info(f"预热步数: {self.num_warmup_steps}")
        
        total_steps = len(train_loader)
        best_val_metrics = None
        best_model_state = None
        best_epoch = -1
        
        # 用于记录训练历史
        history = {
            'train_loss': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            logger.info("-" * 50)
            
            # 训练阶段
            self.model.train()
            total_loss = 0
            batch_count = 0
            
            # 使用tqdm创建进度条
            pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', 
                       ncols=100, unit='batch')
            
            for batch_idx, batch in enumerate(pbar):
                self.optimizer.zero_grad()
                
                batch_loss = 0
                # 对每个查询处理
                for i in range(len(batch['query'])):
                    query = batch['query'][i]
                    pos_docs = batch['pos_docs'][i]
                    neg_docs = batch['neg_docs'][i]
                    
                    # 计算所有正样本的得分
                    pos_scores = []
                    for pos_doc in pos_docs:
                        score = self.model(query, pos_doc)
                        pos_scores.append(score.view(-1).to(self.device))
                    
                    # 计算所有负样本的得分
                    neg_scores = []
                    for neg_doc in neg_docs:
                        score = self.model(query, neg_doc)
                        neg_scores.append(score.view(-1).to(self.device))
                    
                    if not pos_scores or not neg_scores:
                        continue
                    
                    # 计算每个正样本与所有负样本之间的损失
                    pair_losses = []
                    for pos_score in pos_scores:
                        for neg_score in neg_scores:
                            target = torch.ones_like(pos_score).to(self.device)
                            loss = self.criterion(pos_score, neg_score, target)
                            pair_losses.append(loss)
                    
                    # 计算该查询的平均损失
                    if pair_losses:
                        query_loss = sum(pair_losses) / len(pair_losses)
                        batch_loss += query_loss
                
                # 平均批次损失
                if len(batch['query']) > 0:
                    batch_loss = batch_loss / len(batch['query'])
                    batch_loss.backward()
                    
                    # 梯度裁剪，防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    scheduler.step()  # 更新学习率
                    
                    current_lr = scheduler.get_last_lr()[0]
                    total_loss += batch_loss.item()
                    batch_count += 1
                    
                    # 更新进度条描述
                    avg_loss = total_loss / batch_count
                    pbar.set_postfix({
                        'loss': f'{batch_loss.item():.4f}',
                        'avg_loss': f'{avg_loss:.4f}',
                        'lr': f'{current_lr:.2e}'
                    })
                    
                    # 记录历史
                    history['learning_rates'].append(current_lr)
            
            # 关闭进度条
            pbar.close()
            
            avg_train_loss = total_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            logger.info(f"\nEpoch {epoch + 1} 训练完成")
            logger.info(f"平均损失: {avg_train_loss:.4f}")
            logger.info(f"当前学习率: {scheduler.get_last_lr()[0]:.2e}")
            
            # 保存当前epoch的模型
            model_path = os.path.join(self.models_dir, f'model_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': avg_train_loss,
            }, model_path)
            logger.info(f"√ 已保存Epoch {epoch + 1}的模型到 {model_path}")
            
            # 在验证集上评估
            logger.info("\n在验证集上评估...")
            try:
                # 评估原始排序和重排序效果
                original_accuracy = self.evaluate_original_ranking(val_sessions)
                val_metrics = self.evaluate(val_sessions)
                
                # 显示评估结果
                logger.info("\n验证集评估结果:")
                logger.info("原始检索排序:")
                logger.info(f"- 准确率: {original_accuracy:.2f}%")
                
                logger.info("\n模型重排序:")
                logger.info(f"- 准确率 (排名第一): {val_metrics['accuracy']:.2f}%")
                logger.info(f"- 平均排名: {val_metrics['mean_rank']:.2f}")
                logger.info(f"- MRR: {val_metrics['mrr']:.4f}")
                for k, recall in val_metrics['recall_at_k'].items():
                    logger.info(f"- 召回率@{k}: {recall:.2f}%")
                
                # 计算相对变化
                accuracy_change = val_metrics['accuracy'] - original_accuracy
                logger.info("\n效果变化:")
                logger.info(f"准确率变化: {accuracy_change:+.2f}% ({accuracy_change/original_accuracy*100:+.2f}% 相对变化)")
                
                # 更新最佳模型（基于MRR指标）
                if best_val_metrics is None or val_metrics['mrr'] > best_val_metrics['mrr']:
                    best_val_metrics = val_metrics
                    best_model_state = self.model.state_dict()
                    best_epoch = epoch + 1
                    
                    # 保存最佳模型
                    best_model_path = os.path.join(self.models_dir, 'best_model.pth')
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'metrics': val_metrics,
                        'original_accuracy': original_accuracy
                    }, best_model_path)
                    logger.info(f"\n√ 更新最佳模型 (Epoch {epoch + 1})")
                    logger.info(f"- 当前最佳MRR: {val_metrics['mrr']:.4f}")
                    logger.info(f"- 当前最佳准确率: {val_metrics['accuracy']:.2f}%")
            
            except Exception as e:
                logger.error(f"验证过程出错: {str(e)}")
                logger.info("继续训练...")
                continue
        
        logger.info("\n训练完成！")
        if best_val_metrics:
            logger.info(f"\n最佳模型来自Epoch {best_epoch}:")
            logger.info(f"- MRR: {best_val_metrics['mrr']:.4f}")
            logger.info(f"- 准确率: {best_val_metrics['accuracy']:.2f}%")
            logger.info(f"- 平均排名: {best_val_metrics['mean_rank']:.2f}")
            for k, recall in best_val_metrics['recall_at_k'].items():
                logger.info(f"- 召回率@{k}: {recall:.2f}%")
            
            # 恢复最佳模型的状态
            self.model.load_state_dict(best_model_state)
            logger.info("\n√ 已将模型状态恢复为最佳验证效果的状态")

    def evaluate_original_ranking(self, test_sessions):
        """评估原始检索排序的准确率"""
        logger.info("开始评估原始检索排序...")
        
        # 加载搜索结果数据
        sort_ai_dir = os.path.dirname(os.path.abspath(__file__))
        results_df = pd.read_csv(os.path.join(sort_ai_dir, 'training_data', 'search_results.csv'))
        
        correct = 0
        total = 0
        
        for session in test_sessions:
            query_text = session['query']
            pos_doc_id = session['pos_doc_ids'][0]  # 取第一个正样本
            
            # 获取该会话的搜索结果
            session_results = results_df[results_df['session_id'] == session['session_id']]
            
            if len(session_results) == 0:
                continue
            
            # 获取正样本和负样本的排名
            pos_results = session_results[session_results['entity_id'] == pos_doc_id]
            neg_results = session_results[session_results['entity_id'] == session['neg_doc_ids'][0]]
            
            if len(pos_results) == 0 or len(neg_results) == 0:
                continue
            
            pos_rank = pos_results.iloc[0]['rank']
            neg_rank = neg_results.iloc[0]['rank']
            
            # 如果正样本排在负样本前面，则判定为正确
            if pos_rank < neg_rank:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        logger.info(f"原始检索排序准确率: {accuracy:.4f}")
        return accuracy

    def evaluate(self, test_sessions, is_validation=False):
        """评估模型重排序效果"""
        logger.info("\n开始评估模型重排序效果...")
        
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_ranks = []
        reciprocal_ranks = []
        recall_at_k = {1: 0, 3: 0, 5: 0, 10: 0}  # 计算不同K值的召回率
        
        with torch.no_grad():
            for session in test_sessions:
                # 计算所有正样本的得分
                pos_scores = []
                for pos_doc in session['pos_docs']:
                    score = self.model(session['query'], pos_doc)
                    pos_scores.append(score.view(-1).to(self.device))  # 移动到GPU
                
                # 计算所有负样本的得分
                neg_scores = []
                for neg_doc in session['neg_docs']:
                    score = self.model(session['query'], neg_doc)
                    neg_scores.append(score.view(-1).to(self.device))  # 移动到GPU
                
                if not pos_scores or not neg_scores:
                    continue
                
                # 将所有得分合并并转换为张量
                pos_scores_tensor = torch.cat(pos_scores)  # 使用cat而不是stack
                neg_scores_tensor = torch.cat(neg_scores)  # 使用cat而不是stack
                all_scores = torch.cat([pos_scores_tensor, neg_scores_tensor])
                
                # 计算排名
                sorted_indices = torch.argsort(all_scores, descending=True)
                num_pos = len(pos_scores)
                
                # 找到所有正样本的排名
                pos_ranks = []
                for i in range(num_pos):
                    # 找到当前正样本在排序后的位置
                    rank_indices = torch.where(sorted_indices == i)[0]
                    if len(rank_indices) == 0:
                        print(f"警告：无法找到正样本 {i} 的排名")
                        continue
                    rank = rank_indices[0].item() + 1
                    pos_ranks.append(rank)
                    all_ranks.append(rank)
                    reciprocal_ranks.append(1.0 / rank)
                
                if not pos_ranks:
                    print("警告：该会话没有有效的排名")
                    continue
                
                # 计算最佳排名（用于准确率计算）
                best_rank = min(pos_ranks)
                if best_rank == 1:
                    total_correct += 1
                
                # 计算不同K值的召回率
                for k in recall_at_k.keys():
                    top_k_indices = set(sorted_indices[:k].tolist())
                    pos_in_top_k = sum(1 for i in range(num_pos) if i in top_k_indices)
                    recall_at_k[k] += pos_in_top_k / num_pos
                
                # 如果是验证阶段，计算损失
                if is_validation:
                    # 计算所有正负样本对的平均损失
                    validation_loss = 0
                    num_pairs = 0
                    for pos_score in pos_scores:
                        for neg_score in neg_scores:
                            target = torch.ones_like(pos_score).to(self.device)  # 移动到GPU
                            loss = self.criterion(pos_score, neg_score, target)
                            validation_loss += loss.item()
                            num_pairs += 1
                    
                    if num_pairs > 0:
                        validation_loss /= num_pairs
                        total_loss += validation_loss
                
                total_samples += 1
        
        if total_samples == 0:
            logger.warning("没有有效的测试样本！")
            return {
                'loss': float('inf') if is_validation else 0,
                'accuracy': 0,
                'mean_rank': 0,
                'mrr': 0,
                'recall_at_k': {k: 0 for k in recall_at_k.keys()}
            }
        
        # 计算评估指标
        accuracy = (total_correct / total_samples) * 100
        mean_rank = sum(all_ranks) / len(all_ranks) if all_ranks else 0
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
        
        # 计算平均召回率
        for k in recall_at_k.keys():
            recall_at_k[k] = (recall_at_k[k] / total_samples) * 100
        
        # 输出评估结果
        logger.info(f"评估结果:")
        logger.info(f"- 准确率 (排名第一): {accuracy:.2f}%")
        logger.info(f"- 平均排名: {mean_rank:.2f}")
        logger.info(f"- MRR (Mean Reciprocal Rank): {mrr:.4f}")
        for k, recall in recall_at_k.items():
            logger.info(f"- 召回率@{k}: {recall:.2f}%")
        
        if is_validation:
            avg_loss = total_loss / total_samples
            return {
                'loss': avg_loss,
                'accuracy': accuracy,
                'mean_rank': mean_rank,
                'mrr': mrr,
                'recall_at_k': recall_at_k
            }
        else:
            return {
                'accuracy': accuracy,
                'mean_rank': mean_rank,
                'mrr': mrr,
                'recall_at_k': recall_at_k
            }

    def load_model(self, model_path):
        """加载已训练的模型"""
        logger.info(f"从 {model_path} 加载模型...")
        try:
            # 尝试加载模型
            state_dict = torch.load(model_path)
            
            # 检查是否是新格式（包含model_state_dict）
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                # 新格式：包含模型和优化器状态
                self.model.load_state_dict(state_dict['model_state_dict'])
                self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            else:
                # 旧格式：直接是模型状态字典
                self.model.load_state_dict(state_dict)
            
            logger.info("模型加载成功")
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            return False

    def save_model(self, model_path):
        """保存当前模型"""
        logger.info(f"保存模型到 {model_path}...")
        try:
            # 保存为新格式
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, model_path)
            logger.info("模型保存成功")
            return True
        except Exception as e:
            logger.error(f"保存模型失败: {str(e)}")
            return False

def main():
    """主函数"""
    try:
        # 设置随机种子
        set_seed()
        
        # 创建训练器（使用默认的相对路径）
        trainer = ModelTrainer()
        
        # 让用户选择操作模式
        print("\n请选择操作模式：")
        print("1. 训练模型")
        print("2. 评估模型")
        print("3. 训练并评估模型")
        choice = input("请输入选项（1/2/3）: ").strip()
        
        # 准备数据（所有模式都需要）
        train_sessions, val_sessions = trainer.prepare_training_data()
        if not train_sessions or not val_sessions:
            logger.error("准备数据失败")
            return
        
        if choice == '1':
            # 仅训练模式
            logger.info("\n开始训练模式...")
            trainer.train(train_sessions, val_sessions)
            logger.info("训练完成！")
            
        elif choice == '2':
            # 仅评估模式
            logger.info("\n开始评估模式...")
            
            # 让用户选择要评估的模型
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
            if not os.path.exists(models_dir):
                logger.error("没有找到已训练的模型！")
                return
                
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
            if not model_files:
                logger.error("没有找到已训练的模型文件！")
                return
                
            print("\n可用的模型文件：")
            for i, model_file in enumerate(model_files, 1):
                print(f"{i}. {model_file}")
            
            model_choice = input("\n请选择要评估的模型编号: ").strip()
            try:
                model_idx = int(model_choice) - 1
                model_path = os.path.join(models_dir, model_files[model_idx])
            except (ValueError, IndexError):
                logger.error("无效的选择！")
                return
            
            # 加载选定的模型
            if not trainer.load_model(model_path):
                logger.error("无法加载模型，请确保模型文件完整")
                return
            
            # 评估原始排序和重排序效果
            original_accuracy = trainer.evaluate_original_ranking(val_sessions)
            rerank_metrics = trainer.evaluate(val_sessions)
            
            # 显示评估结果
            logger.info("\n评估结果对比:")
            logger.info("原始检索排序:")
            logger.info(f"- 准确率: {original_accuracy:.2f}%")
            
            logger.info("\n模型重排序:")
            logger.info(f"- 准确率 (排名第一): {rerank_metrics['accuracy']:.2f}%")
            logger.info(f"- 平均排名: {rerank_metrics['mean_rank']:.2f}")
            logger.info(f"- MRR: {rerank_metrics['mrr']:.4f}")
            for k, recall in rerank_metrics['recall_at_k'].items():
                logger.info(f"- 召回率@{k}: {recall:.2f}%")
            
            # 计算相对变化
            accuracy_change = rerank_metrics['accuracy'] - original_accuracy
            logger.info("\n效果变化:")
            logger.info(f"准确率变化: {accuracy_change:+.2f}% ({accuracy_change/original_accuracy*100:+.2f}% 相对变化)")
            
        elif choice == '3':
            # 训练并评估模式
            logger.info("\n开始训练并评估模式...")
            
            # 先完成训练
            trainer.train(train_sessions, val_sessions)
            logger.info("训练完成！")
            
            # 评估最后一个epoch的模型
            latest_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', f'model_epoch_3.pth')
            if not trainer.load_model(latest_model):
                logger.error("无法加载最新训练的模型")
                return
            
            # 评估原始排序和重排序效果
            original_accuracy = trainer.evaluate_original_ranking(val_sessions)
            rerank_metrics = trainer.evaluate(val_sessions)
            
            # 显示评估结果
            logger.info("\n评估结果对比:")
            logger.info("原始检索排序:")
            logger.info(f"- 准确率: {original_accuracy:.2f}%")
            
            logger.info("\n模型重排序:")
            logger.info(f"- 准确率 (排名第一): {rerank_metrics['accuracy']:.2f}%")
            logger.info(f"- 平均排名: {rerank_metrics['mean_rank']:.2f}")
            logger.info(f"- MRR: {rerank_metrics['mrr']:.4f}")
            for k, recall in rerank_metrics['recall_at_k'].items():
                logger.info(f"- 召回率@{k}: {recall:.2f}%")
            
            # 计算相对变化
            accuracy_change = rerank_metrics['accuracy'] - original_accuracy
            logger.info("\n效果变化:")
            logger.info(f"准确率变化: {accuracy_change:+.2f}% ({accuracy_change/original_accuracy*100:+.2f}% 相对变化)")
        
        else:
            logger.error("无效的选项！请输入1、2或3")
            return
        
    except Exception as e:
        logger.error(f"执行过程出错: {str(e)}")
        raise

if __name__ == '__main__':
    main() 