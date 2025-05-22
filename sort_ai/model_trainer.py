import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from datetime import datetime
import logging
from typing import List, Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

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
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SearchSessionDataset(Dataset):
    """搜索会话数据集"""
    
    def __init__(self, sessions: List[Dict]):
        """
        初始化数据集
        
        Args:
            sessions: 会话数据列表，每个会话包含查询和点击信息
        """
        if not sessions:
            raise ValueError("Sessions list cannot be empty")
        self.sessions = sessions
        
        # 验证数据格式
        required_fields = {'query', 'pos_docs', 'neg_docs', 'session_id', 'pos_doc_ids', 'neg_doc_ids'}
        for session in sessions:
            missing_fields = required_fields - set(session.keys())
            if missing_fields:
                raise ValueError(f"Session missing required fields: {missing_fields}")
            
            # 确保文档列表非空
            if not session['pos_docs'] or not session['neg_docs']:
                raise ValueError(f"Session {session['session_id']} has empty document lists")
            
            # 确保ID列表与文档列表长度匹配
            if len(session['pos_docs']) != len(session['pos_doc_ids']):
                raise ValueError(f"Mismatch between pos_docs and pos_doc_ids lengths in session {session['session_id']}")
            if len(session['neg_docs']) != len(session['neg_doc_ids']):
                raise ValueError(f"Mismatch between neg_docs and neg_doc_ids lengths in session {session['session_id']}")
        
    def __len__(self):
        return len(self.sessions)
        
    def __getitem__(self, idx):
        try:
            session = self.sessions[idx]
            
            # 确保查询是字符串类型
            query = str(session['query'])
            
            # 确保文档列表是字符串列表
            pos_docs = [str(doc) for doc in session['pos_docs']]
            neg_docs = [str(doc) for doc in session['neg_docs']]
            
            # 确保ID列表是字符串列表
            pos_doc_ids = [str(id) for id in session['pos_doc_ids']]
            neg_doc_ids = [str(id) for id in session['neg_doc_ids']]
            
            return {
                'query': query,
                'pos_docs': pos_docs,
                'neg_docs': neg_docs,
                'session_id': str(session['session_id']),
                'pos_doc_ids': pos_doc_ids,
                'neg_doc_ids': neg_doc_ids
            }
        except Exception as e:
            logger.error(f"Error processing session at index {idx}: {str(e)}")
            raise

class ModelTrainer:
    """模型训练器类"""
    
    def __init__(self, model_path='bert/bert-base-uncased'):
        """初始化模型训练器
        
        Args:
            model_path: 本地bert模型路径（相对于sort_ai目录）
        """
        # 获取当前文件所在目录的路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            logger.info(f"从本地路径加载模型: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=2  # 二分类：相关或不相关
            )
            self.model = self.model.to(self.device)
            logger.info(f"成功加载模型")
        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")
            raise
        
        logger.info(f"使用设备: {self.device}")
        logger.info(f"使用模型路径: {self.model_path}")
    
    def save_model(self, output_dir='models'):
        """保存模型到指定目录"""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"模型已保存到: {output_dir}")
    
    def load_model(self, model_dir='models'):
        """从指定目录加载模型"""
        if not os.path.exists(model_dir):
            logger.error(f"模型目录不存在: {model_dir}")
            return False
            
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = self.model.to(self.device)
            logger.info(f"模型已从 {model_dir} 加载")
            return True
        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")
            return False

    def prepare_training_data(self):
        """准备训练数据"""
        logger.info("开始准备训练数据...")
        
        try:
            # 获取数据文件的路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(current_dir, 'training_data')
            
            # 读取数据文件
            rerank_sessions = pd.read_csv(os.path.join(data_dir, 'rerank_sessions.csv'))
            search_results = pd.read_csv(os.path.join(data_dir, 'search_results.csv'))
            documents = pd.read_csv(os.path.join(data_dir, 'documents.csv'))
            behaviors = pd.read_csv(os.path.join(data_dir, 'behaviors.csv'))
            
            logger.info(f"读取到 {len(rerank_sessions)} 条重排序会话")
            logger.info(f"读取到 {len(search_results)} 条搜索结果")
            logger.info(f"读取到 {len(documents)} 条文档")
            logger.info(f"读取到 {len(behaviors)} 条行为记录")
            
            # 将搜索结果与文档信息合并
            search_results = search_results.merge(
                documents[['id', 'title', 'abstract_inverted_index']],
                left_on='entity_id',
                right_on='id',
                how='inner'
            )
            
            # 将搜索结果与重排序会话合并，使用正确的列名
            data = search_results.merge(
                rerank_sessions[['search_session_id', 'rerank_query']],
                left_on='session_id',
                right_on='search_session_id',
                how='inner'
            )
            
            # 创建训练数据
            training_data = []
            
            # 按会话分组处理
            for session_id, group in data.groupby('session_id'):
                query = str(group['rerank_query'].iloc[0])
                
                # 获取该会话中在behaviors中有记录的文档ID及其停留时间
                session_behaviors = behaviors[behaviors['session_id'] == session_id]
                # 创建文档ID到停留时间的映射，确保停留时间是数值类型
                doc_dwell_times = {
                    doc_id: float(dwell_time) if pd.notnull(dwell_time) else 0.0 
                    for doc_id, dwell_time in zip(session_behaviors['document_id'], session_behaviors['dwell_time'])
                }
                positive_doc_ids = set(doc_dwell_times.keys())
                
                # 计算该会话中正样本的平均停留时间，用于归一化
                dwell_times = [t for t in doc_dwell_times.values() if t > 0]
                if dwell_times:
                    mean_dwell_time = np.mean(dwell_times)
                    std_dwell_time = np.std(dwell_times) if len(dwell_times) > 1 else 1.0
                else:
                    mean_dwell_time = 0.0
                    std_dwell_time = 1.0
                
                # 获取该会话的所有文档
                session_docs = group[['entity_id', 'title', 'abstract_inverted_index']].drop_duplicates()
                
                # 为每个文档创建训练数据
                for _, doc in session_docs.iterrows():
                    is_positive = doc['entity_id'] in positive_doc_ids
                    
                    # 获取文档的停留时间，如果没有则为0
                    dwell_time = doc_dwell_times.get(doc['entity_id'], 0.0)
                    
                    # 计算归一化的停留时间分数（如果是正样本）
                    if is_positive and mean_dwell_time > 0:
                        # 使用z-score归一化，并将结果映射到[0.5, 1]范围
                        z_score = (dwell_time - mean_dwell_time) / (std_dwell_time + 1e-6)
                        normalized_score = 0.5 + 0.5 * (1 / (1 + np.exp(-z_score)))
                    else:
                        normalized_score = 0.0
                    
                    # 确保文本字段是字符串类型
                    title = str(doc['title']) if pd.notnull(doc['title']) else ""
                    abstract = str(doc['abstract_inverted_index']) if pd.notnull(doc['abstract_inverted_index']) else ""
                    
                    training_data.append({
                        'query': query,
                        'doc_text': title + " [SEP] " + abstract,
                        'label': normalized_score
                    })
            
            # 转换为DataFrame
            train_df = pd.DataFrame(training_data)
            
            if len(train_df) == 0:
                logger.error("没有生成任何训练数据")
                return pd.DataFrame(), pd.DataFrame()
            
            # 打乱数据
            train_df = train_df.sample(frac=1, random_state=42)
            
            # 分割训练集和验证集
            train_size = int(0.8 * len(train_df))
            train_data = train_df[:train_size]
            val_data = train_df[train_size:]
            
            logger.info(f"准备完成！训练集大小: {len(train_data)}, 验证集大小: {len(val_data)}")
            logger.info(f"正样本比例: {(train_df['label'] > 0).mean():.2%}")
            logger.info(f"平均标签值: {train_df['label'].mean():.4f}")
            
            return train_data, val_data
            
        except Exception as e:
            logger.error(f"准备训练数据时出错: {str(e)}")
            logger.exception("详细错误信息：")  # 添加详细的错误堆栈信息
            return pd.DataFrame(), pd.DataFrame()

    def _process_doc_text(self, doc):
        """处理文档文本，确保类型安全并保留更多语义信息"""
        # 处理标题
        title = str(doc['title']) if 'title' in doc else ''
        title = title.strip()
        
        # 处理摘要
        abstract = str(doc['abstract']) if 'abstract' in doc else ''
        abstract = abstract.strip()
        
        # 保留更多标点符号和特殊字符，只移除可能影响模型的字符
        def clean_text(text):
            # 保留所有字母、数字、基本标点和常用符号
            return ''.join(c for c in text if c.isalnum() or c.isspace() or c in '.,!?-()[]{}:;"\'')
        
        title = clean_text(title)
        abstract = clean_text(abstract)
        
        # 使用特殊标记分隔标题和摘要，帮助模型区分不同部分
        return f"[TITLE] {title} [ABSTRACT] {abstract}".strip()

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

    def _check_early_stopping(self, current_mrr):
        """检查是否需要早停
        
        Args:
            current_mrr: 当前epoch的MRR值
            
        Returns:
            bool: 是否应该停止训练
        """
        if current_mrr > self.early_stopping['best_mrr'] + self.early_stopping['min_delta']:
            self.early_stopping['best_mrr'] = current_mrr
            self.early_stopping['counter'] = 0
            return False
        else:
            self.early_stopping['counter'] += 1
            if self.early_stopping['counter'] >= self.early_stopping['patience']:
                return True
            return False

    def train(self, train_sessions, val_sessions, num_epochs=10, batch_size=32):
        """Train the model using the prepared training data."""
        logger.info("开始训练模型...")
        
        # 准备训练数据
        train_texts = [session['query'] + " [SEP] " + session['doc_text'] for session in train_sessions]
        train_labels = torch.tensor([session['label'] for session in train_sessions])
        
        # 准备验证数据
        val_texts = [session['query'] + " [SEP] " + session['doc_text'] for session in val_sessions]
        val_labels = torch.tensor([session['label'] for session in val_sessions])
        
        # 将文本转换为模型输入格式
        train_encodings = self.tokenizer(train_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        val_encodings = self.tokenizer(val_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        
        # 创建数据集
        train_dataset = TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            train_labels
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(
            val_encodings['input_ids'],
            val_encodings['attention_mask'],
            val_labels
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # 设置优化器
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        
        # 训练循环
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            # 训练阶段
            self.model.train()
            total_train_loss = 0
            
            # 使用tqdm显示进度条
            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in train_iterator:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                
                # 清零梯度
                optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels.float()  # 确保标签是浮点数类型
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 更新进度条
                train_iterator.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # 验证阶段
            self.model.eval()
            total_val_loss = 0
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels.float()
                    )
                    
                    loss = outputs.loss
                    total_val_loss += loss.item()
                    
                    logits = outputs.logits
                    preds = torch.sigmoid(logits).squeeze()  # 使用sigmoid获取概率值
                    val_preds.extend(preds.cpu().numpy())
                    val_true.extend(labels.cpu().numpy())
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            # 计算MSE（因为我们在处理回归问题）
            val_mse = np.mean((np.array(val_true) - np.array(val_preds)) ** 2)
            
            logger.info(f'Epoch {epoch+1}:')
            logger.info(f'Average training loss: {avg_train_loss:.4f}')
            logger.info(f'Average validation loss: {avg_val_loss:.4f}')
            logger.info(f'Validation MSE: {val_mse:.4f}')
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model()
                logger.info("保存了新的最佳模型")
            
            logger.info("-" * 60)

    def _save_checkpoint(self, epoch, state):
        """保存检查点
        
        只保存最佳模型的检查点
        """
        checkpoint_path = os.path.join(self.models_dir, 'best_model.pth')
        try:
            torch.save(state, checkpoint_path)
            logger.info(f"模型已保存到 {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"保存模型失败: {str(e)}")
            return False

    def _load_best_model(self):
        """加载最佳模型"""
        best_model_path = os.path.join(self.models_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_history = checkpoint.get('training_history', self.training_history)
            self.best_metrics = checkpoint.get('best_metrics', None)
            self.early_stopping = checkpoint.get('early_stopping', self.early_stopping)
            logger.info("已加载最佳模型状态")
            return True
        return False

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
        device = next(self.model.parameters()).device  # 获取模型所在设备
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_ranks = []
        reciprocal_ranks = []
        recall_at_k = {1: 0, 3: 0, 5: 0, 10: 0}  # 计算不同K值的召回率
        
        try:
            with torch.no_grad():
                for session in test_sessions:
                    try:
                        # 将查询移动到正确的设备
                        query = session['query']
                        
                        # 计算所有正样本的得分
                        pos_scores = []
                        for pos_doc in session['pos_docs']:
                            try:
                                score = self.model(query, pos_doc)
                                pos_scores.append(score.view(-1).to(device))
                            except Exception as e:
                                logger.warning(f"处理正样本时出错: {str(e)}")
                                continue
                        
                        # 计算所有负样本的得分
                        neg_scores = []
                        for neg_doc in session['neg_docs']:
                            try:
                                score = self.model(query, neg_doc)
                                neg_scores.append(score.view(-1).to(device))
                            except Exception as e:
                                logger.warning(f"处理负样本时出错: {str(e)}")
                                continue
                        
                        if not pos_scores or not neg_scores:
                            logger.warning(f"会话 {session['session_id']} 没有有效的样本对")
                            continue
                        
                        # 将所有得分合并并转换为张量
                        try:
                            pos_scores_tensor = torch.cat(pos_scores)
                            neg_scores_tensor = torch.cat(neg_scores)
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
                                    logger.warning(f"无法找到正样本 {i} 的排名")
                                    continue
                                rank = rank_indices[0].item() + 1
                                pos_ranks.append(rank)
                                all_ranks.append(rank)
                                reciprocal_ranks.append(1.0 / rank)
                            
                            if not pos_ranks:
                                logger.warning(f"会话 {session['session_id']} 没有有效的排名")
                                continue
                            
                            # 计算最佳排名（用于准确率计算）
                            best_rank = min(pos_ranks)
                            if best_rank == 1:
                                total_correct += 1
                            
                            # 计算不同K值的召回率
                            for k in recall_at_k.keys():
                                if k <= len(sorted_indices):
                                    top_k_indices = set(sorted_indices[:k].tolist())
                                    pos_in_top_k = sum(1 for i in range(num_pos) if i in top_k_indices)
                                    recall_at_k[k] += pos_in_top_k / num_pos
                            
                            # 如果是验证阶段，计算损失
                            if is_validation:
                                validation_loss = 0
                                num_pairs = 0
                                for pos_score in pos_scores:
                                    for neg_score in neg_scores:
                                        target = torch.ones_like(pos_score).to(device)
                                        loss = self.criterion(pos_score, neg_score, target)
                                        if torch.isfinite(loss):
                                            validation_loss += loss.item()
                                            num_pairs += 1
                                
                                if num_pairs > 0:
                                    validation_loss /= num_pairs
                                    total_loss += validation_loss
                            
                            total_samples += 1
                            
                        except Exception as e:
                            logger.error(f"计算评估指标时出错: {str(e)}")
                            continue
                            
                    except Exception as e:
                        logger.error(f"处理会话时出错: {str(e)}")
                        continue
            
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
            logger.info(f"评估结果 (总样本数: {total_samples}):")
            logger.info(f"- 准确率 (排名第一): {accuracy:.2f}%")
            logger.info(f"- 平均排名: {mean_rank:.2f}")
            logger.info(f"- MRR (Mean Reciprocal Rank): {mrr:.4f}")
            for k, recall in recall_at_k.items():
                logger.info(f"- 召回率@{k}: {recall:.2f}%")
            
            if is_validation:
                avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
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
                
        except Exception as e:
            logger.error(f"评估过程出错: {str(e)}")
            raise

def main():
    """主函数"""
    try:
        logger.info("开始运行模型训练程序...")
        
        # 创建模型训练器，使用本地bert模型路径
        trainer = ModelTrainer(model_path='bert/bert-base-uncased')
        
        # 准备训练数据
        logger.info("准备训练数据...")
        train_df, val_df = trainer.prepare_training_data()
        
        if train_df.empty or val_df.empty:
            logger.error("训练数据为空，请检查数据准备过程")
            return
            
        logger.info(f"训练集大小: {len(train_df)}, 验证集大小: {len(val_df)}")
        
        # 将 DataFrame 转换为字典列表
        train_sessions = train_df.to_dict('records')
        val_sessions = val_df.to_dict('records')
        
        # 开始训练
        trainer.train(
            train_sessions=train_sessions,
            val_sessions=val_sessions,
            num_epochs=10,
            batch_size=32
        )
        
        logger.info("训练完成！")
        
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")
        raise

if __name__ == '__main__':
    main() 