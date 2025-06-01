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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from sort_ai.model import NLRankingModel

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
        
        # 设置模型保存目录
        self.models_dir = os.path.join(current_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        try:
            logger.info(f"初始化NLRankingModel，使用BERT路径: {self.model_path}")
            # 使用NLRankingModel替代AutoModelForSequenceClassification
            self.model = NLRankingModel(bert_path=self.model_path)
            self.model = self.model.to(self.device)
            # 获取模型内部的tokenizer
            self.tokenizer = self.model.tokenizer
            
            # 打印模型参数统计
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"模型总参数量: {total_params:,}")
            logger.info(f"可训练参数量: {trainable_params:,}")
            logger.info(f"参数冻结比例: {(total_params-trainable_params)/total_params:.2%}")
            
        except Exception as e:
            logger.error(f"初始化模型时出错: {str(e)}")
            raise
        
        logger.info(f"使用设备: {self.device}")
        logger.info(f"使用模型路径: {self.model_path}")
        logger.info(f"模型保存目录: {self.models_dir}")
    
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

    def calculate_relevance_score(self, dwell_time, is_clicked, session_mean_dwell, session_std_dwell):
        """计算文档的相关度分数
        
        Args:
            dwell_time: 停留时间
            is_clicked: 是否被点击
            session_mean_dwell: 会话平均停留时间
            session_std_dwell: 会话停留时间标准差
            
        Returns:
            float: 相关度分数 [0, 1]
        """
        if not is_clicked:
            return 0.0
            
        # 基础分数：根据是否点击
        base_score = 0.3 if is_clicked else 0.0
        
        # 停留时间分数
        if dwell_time > 0 and session_mean_dwell > 0:
            # 使用z-score归一化
            z_score = (dwell_time - session_mean_dwell) / (session_std_dwell + 1e-6)
            # 将z-score映射到[0, 0.7]范围，加上基础分数
            dwell_score = 0.7 * (1 / (1 + np.exp(-z_score)))
            return base_score + dwell_score
        
        return base_score

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
            
            logger.info("开始处理训练数据...")
            training_data = []
            
            # 对每个重排序会话进行处理
            for _, rerank_session in rerank_sessions.iterrows():
                try:
                    search_session_id = rerank_session['search_session_id']
                    query = rerank_session['rerank_query']
                    
                    # 获取该会话的所有检索结果
                    session_results = search_results[
                        search_results['session_id'] == search_session_id
                    ].copy()
                    
                    if len(session_results) == 0:
                        continue
                    
                    # 获取该会话的行为数据
                    session_behaviors = behaviors[
                        behaviors['session_id'] == search_session_id
                    ]
                    
                    # 创建文档ID到停留时间的映射
                    doc_dwell_times = {}
                    for _, behavior in session_behaviors.iterrows():
                        doc_id = behavior['document_id']
                        dwell_time = behavior['dwell_time']
                        if pd.notnull(dwell_time):
                            doc_dwell_times[doc_id] = float(dwell_time)
                    
                    # 计算停留时间的统计信息，用于归一化
                    if doc_dwell_times:
                        dwell_times = list(doc_dwell_times.values())
                        max_dwell_time = max(dwell_times)
                        mean_dwell_time = np.mean(dwell_times)
                        std_dwell_time = np.std(dwell_times) if len(dwell_times) > 1 else 1.0
                    else:
                        continue  # 如果没有任何停留时间数据，跳过该会话
                    
                    # 为每个检索结果计算相关性得分
                    session_docs = []
                    for _, result in session_results.iterrows():
                        doc_id = result['entity_id']
                        
                        # 获取文档信息
                        doc_info = documents[documents['id'] == doc_id]
                        if len(doc_info) == 0:
                            continue
                        
                        doc_info = doc_info.iloc[0]
                        
                        # 计算归一化的相关性得分
                        dwell_time = doc_dwell_times.get(doc_id, 0.0)
                        if dwell_time > 0:
                            # 使用z-score归一化，然后映射到[0,1]区间
                            z_score = (dwell_time - mean_dwell_time) / (std_dwell_time + 1e-6)
                            relevance_score = 1 / (1 + np.exp(-z_score))  # sigmoid函数
                        else:
                            relevance_score = 0.0
                        
                        session_docs.append({
                            'id': str(doc_id),
                            'title': str(doc_info['title']),
                            'abstract': str(doc_info['abstract_inverted_index']),
                            'relevance_score': relevance_score,
                            'dwell_time': dwell_time,
                            'rank_position': result['rank_position']
                        })
                    
                    # 如果没有有效的文档，跳过该会话
                    if not session_docs:
                        continue
                    
                    # 按相关性得分排序文档
                    session_docs.sort(key=lambda x: x['relevance_score'], reverse=True)
                    
                    # 创建训练样本
                    training_sample = {
                        'session_id': str(search_session_id),
                        'query': query,
                        'documents': session_docs,
                        'max_dwell_time': max_dwell_time,
                        'mean_dwell_time': mean_dwell_time,
                        'std_dwell_time': std_dwell_time
                    }
                    
                    training_data.append(training_sample)
                    
                except Exception as e:
                    logger.warning(f"处理会话 {search_session_id} 时出错: {str(e)}")
                    continue
            
            # 打印一些统计信息
            logger.info(f"总共处理了 {len(training_data)} 个有效的重排序会话")
            
            # 计算一些统计信息
            total_docs = sum(len(session['documents']) for session in training_data)
            avg_docs_per_session = total_docs / len(training_data) if training_data else 0
            
            logger.info(f"平均每个会话包含 {avg_docs_per_session:.2f} 个文档")
            
            # 分割训练集和验证集
            train_size = int(0.8 * len(training_data))
            train_data = training_data[:train_size]
            val_data = training_data[train_size:]
            
            logger.info(f"训练集大小: {len(train_data)}, 验证集大小: {len(val_data)}")
            
            return train_data, val_data
            
        except Exception as e:
            logger.error(f"准备训练数据时出错: {str(e)}")
            logger.exception("详细错误信息：")
            return [], []

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
        queries = []
        session_ids = []
        doc_titles = []
        doc_abstracts = []
        relevance_scores = []
        
        # 处理每个会话的数据
        for session in batch:
            query = session['query']
            session_id = session['session_id']
            docs = session['documents']
            
            # 对于会话中的每个文档
            for doc in docs:
                queries.append(query)  # 重复查询，每个文档一次
                session_ids.append(session_id)
                doc_titles.append(doc['title'])
                doc_abstracts.append(doc['abstract'])
                relevance_scores.append(doc['relevance_score'])
        
        # 将列表转换为张量
        relevance_scores = torch.tensor(relevance_scores, dtype=torch.float)
        
        return {
            'query': queries,
            'session_id': session_ids,
            'title': doc_titles,
            'abstract': doc_abstracts,
            'relevance_score': relevance_scores
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

    def calculate_precision_at_k(self, predictions, labels, k):
        """计算P@K指标
        
        Args:
            predictions: 预测分数列表
            labels: 真实标签列表
            k: 取前k个结果计算准确率
            
        Returns:
            float: P@K值
        """
        if len(predictions) < k:
            return 0.0
            
        # 获取预测分数最高的k个位置的索引
        top_k_indices = np.argsort(predictions)[-k:]
        # 获取这k个位置的真实标签
        top_k_relevant = labels[top_k_indices]
        # 计算准确率（相关文档数量/k）
        return np.sum(top_k_relevant) / k

    def calculate_ndcg_at_k(self, predictions, labels, k):
        """计算NDCG@K
        
        Args:
            predictions: 预测分数列表
            labels: 真实标签列表（相关度分数）
            k: 取前k个结果计算NDCG
            
        Returns:
            float: NDCG@K值
        """
        if len(predictions) < k:
            return 0.0
            
        # 获取预测分数最高的k个位置的索引
        top_k_indices = np.argsort(predictions)[-k:][::-1]  # 降序排列
        
        # 计算DCG
        dcg = 0
        for i, idx in enumerate(top_k_indices):
            rel = labels[idx]
            dcg += (2 ** rel - 1) / np.log2(i + 2)  # i+2 是因为log2从1开始
            
        # 计算IDCG（理想DCG）
        ideal_labels = np.sort(labels)[::-1]  # 降序排列
        idcg = 0
        for i in range(min(k, len(ideal_labels))):
            rel = ideal_labels[i]
            idcg += (2 ** rel - 1) / np.log2(i + 2)
            
        # 避免除零
        if idcg == 0:
            return 0.0
            
        return dcg / idcg

    def train(self, train_sessions, val_sessions, num_epochs=10, batch_size=32, patience=3, min_delta=1e-4):
        """Train the model using the prepared training data."""
        logger.info("开始训练模型...")
        
        # 设置优化器
        optimizer = AdamW(
            self.model.parameters(),
            lr=2e-5,
            weight_decay=0.01
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_sessions,
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2,
            collate_fn=self.collate_fn
        )
        
        # 初始化早停相关变量
        best_val_loss = float('inf')
        best_metrics = None
        best_model_state = None
        patience_counter = 0
        
        logger.info("训练过程说明：")
        logger.info("1. 每个epoch在前一个epoch的基础上继续训练")
        logger.info("2. 使用MSE损失计算文档相关性得分")
        logger.info("3. 每个epoch结束后在验证集上评估")
        logger.info(f"4. 使用早停机制（耐心值={patience}, 最小改善={min_delta}）")
        logger.info("5. 保存验证损失最小的模型\n")
        
        for epoch in range(num_epochs):
            self.model.train()
            total_train_loss = 0
            
            # 使用tqdm显示进度条
            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in train_iterator:
                # 获取批次数据
                queries = batch['query']
                titles = batch['title']
                abstracts = batch['abstract']
                relevance_scores = batch['relevance_score'].to(self.device)
                
                # 清零梯度
                optimizer.zero_grad()
                
                batch_loss = 0
                num_docs = len(queries)
                
                # 对每个查询-文档对计算得分
                predicted_scores = []
                for query, title, abstract in zip(queries, titles, abstracts):
                    # 使用模型计算预测得分
                    score = self.model(query, title, abstract)
                    predicted_scores.append(score)
                
                # 将预测得分转换为张量
                predicted_scores = torch.stack(predicted_scores).squeeze()
                
                # 计算MSE损失
                loss = nn.MSELoss()(predicted_scores, relevance_scores)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # 更新参数
                optimizer.step()
                
                # 累计损失
                total_train_loss += loss.item()
                
                # 更新进度条
                train_iterator.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # 计算平均训练损失
            avg_train_loss = total_train_loss / len(train_loader)
            logger.info(f"\nEpoch {epoch+1}/{num_epochs} - 平均训练损失: {avg_train_loss:.4f}")
            
            # 在验证集上评估
            logger.info("在验证集上评估...")
            val_metrics = self.evaluate(val_sessions, is_validation=True)
            val_loss = val_metrics['loss']
            
            # 输出验证集指标
            logger.info(f"验证集损失: {val_loss:.4f}")
            logger.info(f"验证集NDCG@5: {val_metrics.get('ndcg@5', 0.0):.4f}")
            logger.info(f"验证集NDCG@10: {val_metrics.get('ndcg@10', 0.0):.4f}")
            
            # 检查是否需要保存最佳模型
            if val_loss < best_val_loss - min_delta:
                logger.info("发现更好的模型！")
                best_val_loss = val_loss
                best_metrics = val_metrics
                
                # 保存最佳模型状态
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'metrics': val_metrics
                }
                
                # 保存检查点
                self._save_checkpoint(epoch, best_model_state)
                
                # 重置耐心计数器
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"模型未改善，耐心计数：{patience_counter}/{patience}")
            
            # 检查是否需要早停
            if patience_counter >= patience:
                logger.info(f"\n触发早停！连续 {patience} 个epoch验证损失未改善")
                break
            
            logger.info("-" * 60)
        
        # 训练结束，恢复最佳模型
        if best_model_state is not None:
            logger.info("\n训练结束，恢复最佳模型状态...")
            self.model.load_state_dict(best_model_state['model_state_dict'])
            optimizer.load_state_dict(best_model_state['optimizer_state_dict'])
            
            # 输出最佳结果
            logger.info(f"最佳模型来自第 {best_model_state['epoch']+1} 个epoch")
            logger.info(f"最佳验证损失: {best_val_loss:.4f}")
        
        return best_metrics

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

    def evaluate(self, test_sessions, is_validation=False):
            """评估模型重排序效果"""
            logger.info("\n开始评估模型重排序效果...")
            
            self.model.eval()
            total_loss = 0
            total_correct = 0
            total_samples = 0
            all_ranks = []
            reciprocal_ranks = []
            recall_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
            
            try:
                with torch.no_grad():
                    for session in test_sessions:
                        try:
                            query = session['query']
                            documents = session['documents']
                            
                            if not documents:  # 如果没有文档，跳过这个会话
                                continue
                            
                            # 计算所有文档的得分
                            doc_scores = []
                            true_scores = []
                            for doc in documents:
                                # 使用模型计算预测得分
                                score = self.model(
                                    query,
                                    doc['title'],
                                    doc['abstract']
                                )
                                # 确保score是标量
                                if isinstance(score, torch.Tensor):
                                    score = score.detach().cpu()
                                    if score.dim() > 0:
                                        score = score.mean()  # 如果是多维张量，取平均值
                                    score = score.item()  # 转换为Python标量
                                doc_scores.append(score)
                                true_scores.append(float(doc['relevance_score']))
                            
                            if not doc_scores:  # 如果没有有效的得分，跳过
                                continue
                            
                            # 转换为numpy数组进行排序
                            doc_scores = np.array(doc_scores)
                            true_scores = np.array(true_scores)
                            
                            # 如果是验证阶段，计算MSE损失
                            if is_validation:
                                loss = np.mean((doc_scores - true_scores) ** 2)
                                total_loss += loss
                            
                            # 计算排序
                            predicted_ranks = np.argsort(-doc_scores)  # 降序排序
                            true_ranks = np.argsort(-true_scores)  # 降序排序
                            
                            # 计算各种指标
                            # 1. 计算最高分文档是否正确预测
                            if predicted_ranks[0] == true_ranks[0]:
                                total_correct += 1
                            
                            # 2. 计算MRR
                            for i, true_top in enumerate(true_ranks[:1]):  # 只考虑真实最相关的文档
                                pred_rank = np.where(predicted_ranks == true_top)[0][0] + 1
                                all_ranks.append(pred_rank)
                                reciprocal_ranks.append(1.0 / pred_rank)
                            
                            # 3. 计算不同K值的召回率
                            for k in recall_at_k.keys():
                                if k <= len(predicted_ranks):
                                    pred_top_k = set(predicted_ranks[:k])
                                    true_top_k = set(true_ranks[:k])
                                    recall_at_k[k] += len(pred_top_k & true_top_k) / len(true_top_k)
                            
                            total_samples += 1
                            
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
                        'recall_at_k': {k: 0 for k in recall_at_k.keys()},
                        'ndcg@5': 0,
                        'ndcg@10': 0
                    }
                
                # 计算最终指标
                accuracy = (total_correct / total_samples) * 100
                mean_rank = sum(all_ranks) / len(all_ranks) if all_ranks else 0
                mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
                
                # 计算平均召回率
                for k in recall_at_k.keys():
                    recall_at_k[k] = (recall_at_k[k] / total_samples) * 100
                
                # 构建返回的指标字典
                metrics = {
                    'accuracy': accuracy,
                    'mean_rank': mean_rank,
                    'mrr': mrr,
                    'recall_at_k': recall_at_k,
                    'ndcg@5': recall_at_k[5],  # 简化版NDCG
                    'ndcg@10': recall_at_k[10]  # 简化版NDCG
                }
                
                if is_validation:
                    metrics['loss'] = total_loss / total_samples if total_samples > 0 else float('inf')
                
                # 输出评估结果
                logger.info(f"评估结果 (总样本数: {total_samples}):")
                logger.info(f"- 准确率 (排名第一): {accuracy:.2f}%")
                logger.info(f"- 平均排名: {mean_rank:.2f}")
                logger.info(f"- MRR (Mean Reciprocal Rank): {mrr:.4f}")
                logger.info(f"- NDCG@5: {metrics['ndcg@5']:.2f}%")
                logger.info(f"- NDCG@10: {metrics['ndcg@10']:.2f}%")
                
                return metrics
            
            except Exception as e:
                logger.error(f"评估过程出错: {str(e)}")
                raise
    def test_evaluate_logic(self, test_sessions):
            """测试评估逻辑的函数"""
            logger.info("\n开始测试评估逻辑...")
            
            total_samples = 0
            total_correct = 0  # 添加这个初始化
            all_ranks = []
            reciprocal_ranks = []
            recall_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
            
            try:
                for session in test_sessions:
                    try:
                        query = session['query']
                        documents = session['documents']
                        
                        if not documents:
                            continue
                        
                        # 模拟模型预测，直接使用真实的相关性得分作为预测得分
                        doc_scores = []
                        true_scores = []
                        for doc in documents:
                            # 使用真实的相关性得分
                            score = doc['relevance_score']
                            doc_scores.append(score)
                            true_scores.append(score)
                        
                        if not doc_scores:
                            continue
                        
                        # 转换为numpy数组进行排序
                        doc_scores = np.array(doc_scores)
                        true_scores = np.array(true_scores)
                        
                        # 计算排序
                        predicted_ranks = np.argsort(-doc_scores)  # 降序排序
                        true_ranks = np.argsort(-true_scores)  # 降序排序
                        
                        # 计算各种指标
                        # 1. 计算最高分文档是否正确预测
                        if predicted_ranks[0] == true_ranks[0]:
                            total_correct += 1
                        
                        # 2. 计算MRR
                        for i, true_top in enumerate(true_ranks[:1]):
                            pred_rank = np.where(predicted_ranks == true_top)[0][0] + 1
                            all_ranks.append(pred_rank)
                            reciprocal_ranks.append(1.0 / pred_rank)
                        
                        # 3. 计算不同K值的召回率
                        for k in recall_at_k.keys():
                            if k <= len(predicted_ranks):
                                pred_top_k = set(predicted_ranks[:k])
                                true_top_k = set(true_ranks[:k])
                                recall_at_k[k] += len(pred_top_k & true_top_k) / len(true_top_k)
                        
                        total_samples += 1
                        
                        # 打印每个会话的详细信息
                        logger.info(f"\n会话ID: {session['session_id']}")
                        logger.info(f"查询: {query}")
                        logger.info(f"文档数量: {len(documents)}")
                        logger.info("文档得分:")
                        for i, (score, true_score) in enumerate(zip(doc_scores, true_scores)):
                            logger.info(f"  文档{i}: 预测={score:.4f}, 真实={true_score:.4f}")
                        logger.info(f"预测排序: {predicted_ranks}")
                        logger.info(f"真实排序: {true_ranks}")
                        
                    except Exception as e:
                        logger.error(f"处理会话时出错: {str(e)}")
                        continue
                
                if total_samples == 0:
                    logger.warning("没有有效的测试样本！")
                    return
                
                # 计算最终指标
                accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
                mean_rank = sum(all_ranks) / len(all_ranks) if all_ranks else 0
                mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
                
                # 计算平均召回率
                for k in recall_at_k.keys():
                    recall_at_k[k] = (recall_at_k[k] / total_samples) * 100 if total_samples > 0 else 0
                
                # 输出评估结果
                logger.info(f"\n评估结果 (总样本数: {total_samples}):")
                logger.info(f"- 准确率 (排名第一): {accuracy:.2f}%")
                logger.info(f"- 平均排名: {mean_rank:.2f}")
                logger.info(f"- MRR (Mean Reciprocal Rank): {mrr:.4f}")
                for k, recall in recall_at_k.items():
                    logger.info(f"- 召回率@{k}: {recall:.2f}%")
            
                return {
                    'accuracy': accuracy,
                    'mean_rank': mean_rank,
                    'mrr': mrr,
                    'recall_at_k': recall_at_k
                }
                
            except Exception as e:
                logger.error(f"测试评估逻辑时出错: {str(e)}")
                raise
    def predict(self, query, doc_text):
        """使用模型预测文档相关性得分
        
        Args:
            query: 查询文本
            doc_text: 文档文本
            
        Returns:
            float: 相关性得分
        """
        try:
            # 确保模型处于评估模式
            self.model.eval()
            
            # 准备输入
            text = f"{query} [SEP] {doc_text}"
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # 将输入移到正确的设备上
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 使用模型进行预测
            with torch.no_grad():
                outputs = self.model(**inputs)
                score = outputs.logits.squeeze().item()
            
            return score
            
        except Exception as e:
            logger.error(f"预测出错: {str(e)}")
            return 0.0  # 出错时返回默认得分
            
    def predict_batch(self, queries, doc_texts):
        """批量预测文档相关性得分
        
        Args:
            queries: 查询文本列表
            doc_texts: 文档文本列表
            
        Returns:
            list: 相关性得分列表
        """
        try:
            # 确保模型处于评估模式
            self.model.eval()
            
            # 准备输入
            texts = [f"{q} [SEP] {d}" for q, d in zip(queries, doc_texts)]
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # 将输入移到正确的设备上
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 使用模型进行预测
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = outputs.logits.squeeze().cpu().tolist()
                
            # 确保返回的是列表
            if not isinstance(scores, list):
                scores = [scores]
                
            return scores
            
        except Exception as e:
            logger.error(f"批量预测出错: {str(e)}")
            return [0.0] * len(queries)  # 出错时返回默认得分列表

def main():
    """主函数"""
    try:
        logger.info("开始运行模型训练程序...")
        
        # 创建模型训练器，使用本地bert模型路径
        trainer = ModelTrainer(model_path='bert/bert-base-uncased')
        
        # 准备训练数据
        logger.info("准备训练数据...")
        train_data, val_data = trainer.prepare_training_data()
        
        # 检查训练数据是否为空
        if not train_data or not val_data:
            logger.error("训练数据为空，请检查数据准备过程")
            return
            
        logger.info(f"训练集大小: {len(train_data)}, 验证集大小: {len(val_data)}")
        # 在main函数中添加
        logger.info("测试评估逻辑...")
        trainer.test_evaluate_logic(val_data[0:3])
        # 开始训练
        trainer.train(
            train_sessions=train_data,
            val_sessions=val_data,
            num_epochs=10,
            batch_size=32
        )
        
        logger.info("训练完成！")
        
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")
        raise

if __name__ == '__main__':
    main() 