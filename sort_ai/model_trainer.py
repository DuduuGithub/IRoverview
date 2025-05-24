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
                num_labels=1,  # 回归问题，使用单个输出
                problem_type="regression",
                hidden_dropout_prob=0.2,  # 添加dropout
                attention_probs_dropout_prob=0.1  # 添加attention dropout
            )
            
            # 冻结BERT底层参数
            for param in self.model.bert.embeddings.parameters():
                param.requires_grad = False
            for layer in self.model.bert.encoder.layer[:8]:  # 冻结前8层
                for param in layer.parameters():
                    param.requires_grad = False
                    
            self.model = self.model.to(self.device)
            logger.info(f"成功加载模型")
            
            # 打印模型参数统计
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"模型总参数量: {total_params:,}")
            logger.info(f"可训练参数量: {trainable_params:,}")
            logger.info(f"参数冻结比例: {(total_params-trainable_params)/total_params:.2%}")
            
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
            
            # 将搜索结果与重排序会话合并
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
                
                # 获取该会话中的行为数据
                session_behaviors = behaviors[behaviors['session_id'] == session_id]
                
                # 创建文档ID到行为数据的映射
                doc_behaviors = {
                    doc_id: {
                        'dwell_time': float(dwell_time) if pd.notnull(dwell_time) else 0.0,
                        'is_clicked': True
                    }
                    for doc_id, dwell_time in zip(session_behaviors['document_id'], session_behaviors['dwell_time'])
                }
                
                # 计算会话级别的统计信息
                dwell_times = [b['dwell_time'] for b in doc_behaviors.values() if b['dwell_time'] > 0]
                session_mean_dwell = np.mean(dwell_times) if dwell_times else 0.0
                session_std_dwell = np.std(dwell_times) if len(dwell_times) > 1 else 1.0
                
                # 获取该会话的所有文档
                session_docs = group[['entity_id', 'title', 'abstract_inverted_index']].drop_duplicates()
                
                # 为每个文档创建训练数据
                for _, doc in session_docs.iterrows():
                    # 获取文档的行为数据
                    behavior = doc_behaviors.get(doc['entity_id'], {'dwell_time': 0.0, 'is_clicked': False})
                    
                    # 计算相关度分数
                    relevance_score = self.calculate_relevance_score(
                        dwell_time=behavior['dwell_time'],
                        is_clicked=behavior['is_clicked'],
                        session_mean_dwell=session_mean_dwell,
                        session_std_dwell=session_std_dwell
                    )
                    
                    # 确保文本字段是字符串类型
                    title = str(doc['title']) if pd.notnull(doc['title']) else ""
                    abstract = str(doc['abstract_inverted_index']) if pd.notnull(doc['abstract_inverted_index']) else ""
                    
                    training_data.append({
                        'query': query,
                        'doc_text': title + " [SEP] " + abstract,
                        'label': relevance_score
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
            
            # 输出相关度分数的分布信息
            relevance_stats = train_df['label'].describe()
            logger.info("\n相关度分数统计:")
            logger.info(f"总样本数: {len(train_df)}")
            logger.info(f"平均分数: {relevance_stats['mean']:.4f}")
            logger.info(f"标准差: {relevance_stats['std']:.4f}")
            logger.info(f"最小值: {relevance_stats['min']:.4f}")
            logger.info(f"25%分位: {relevance_stats['25%']:.4f}")
            logger.info(f"中位数: {relevance_stats['50%']:.4f}")
            logger.info(f"75%分位: {relevance_stats['75%']:.4f}")
            logger.info(f"最大值: {relevance_stats['max']:.4f}")
            
            logger.info(f"准备完成！训练集大小: {len(train_data)}, 验证集大小: {len(val_data)}")
            
            return train_data, val_data
            
        except Exception as e:
            logger.error(f"准备训练数据时出错: {str(e)}")
            logger.exception("详细错误信息：")
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
        
        # 准备训练数据
        train_texts = [session['query'] + " [SEP] " + session['doc_text'] for session in train_sessions]
        train_labels = torch.tensor([session['label'] for session in train_sessions], dtype=torch.float)
        
        # 准备验证数据
        val_texts = [session['query'] + " [SEP] " + session['doc_text'] for session in val_sessions]
        val_labels = torch.tensor([session['label'] for session in val_sessions], dtype=torch.float)
        
        # 将文本转换为模型输入格式
        train_encodings = self.tokenizer(
            train_texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        val_encodings = self.tokenizer(
            val_texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        
        # 创建数据集
        train_dataset = TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            train_labels
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2  # 使用多进程加载数据
        )
        
        val_dataset = TensorDataset(
            val_encodings['input_ids'],
            val_encodings['attention_mask'],
            val_labels
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            num_workers=2
        )
        
        # 设置优化器和学习率调度器
        optimizer = AdamW(
            [
                {"params": self.model.bert.encoder.layer[8:].parameters(), "lr": 2e-5},
                {"params": self.model.bert.pooler.parameters(), "lr": 3e-5},
                {"params": self.model.classifier.parameters(), "lr": 5e-5}
            ],
            weight_decay=0.01  # 添加权重衰减
        )
        
        # 创建学习率调度器
        num_training_steps = len(train_loader) * num_epochs
        num_warmup_steps = num_training_steps // 10  # 10%的步数用于预热
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        criterion = nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler()  # 使用混合精度训练
        
        # 训练循环
        best_val_loss = float('inf')
        best_metrics = None
        best_model_state = None
        
        # 早停相关变量
        patience_counter = 0
        best_val_loss_for_stopping = float('inf')
        
        logger.info("训练过程说明：")
        logger.info("1. 每个epoch在前一个epoch的基础上继续训练")
        logger.info("2. 使用MSE损失函数计算预测值与真实值的差异")
        logger.info("3. 计算P@K指标（K=1,3,5,10）评估排序质量")
        logger.info(f"4. 使用早停机制（耐心值={patience}, 最小改善={min_delta}）")
        logger.info("5. 保存验证损失最小的模型\n")
        
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
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    logits = outputs.logits.squeeze()
                    loss = criterion(logits, labels)
                
                # 反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # 更新进度条
                train_iterator.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # 验证阶段
            self.model.eval()
            total_val_loss = 0
            all_val_preds = []
            all_val_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    logits = outputs.logits.squeeze()
                    loss = criterion(logits, labels)
                    total_val_loss += loss.item()
                    
                    # 收集所有预测和标签用于计算P@K
                    all_val_preds.extend(logits.cpu().numpy())
                    all_val_labels.extend(labels.cpu().numpy())
            
            # 转换为numpy数组以便计算指标
            all_val_preds = np.array(all_val_preds)
            all_val_labels = np.array(all_val_labels)
            
            # 计算各项指标
            avg_val_loss = total_val_loss / len(val_loader)
            val_mse = np.mean((all_val_labels - all_val_preds) ** 2)
            
            # 计算P@K指标
            p_at_1 = self.calculate_precision_at_k(all_val_preds, all_val_labels, k=1)
            p_at_3 = self.calculate_precision_at_k(all_val_preds, all_val_labels, k=3)
            p_at_5 = self.calculate_precision_at_k(all_val_preds, all_val_labels, k=5)
            p_at_10 = self.calculate_precision_at_k(all_val_preds, all_val_labels, k=10)
            
            # 计算NDCG@K指标
            ndcg_at_1 = self.calculate_ndcg_at_k(all_val_preds, all_val_labels, k=1)
            ndcg_at_3 = self.calculate_ndcg_at_k(all_val_preds, all_val_labels, k=3)
            ndcg_at_5 = self.calculate_ndcg_at_k(all_val_preds, all_val_labels, k=5)
            ndcg_at_10 = self.calculate_ndcg_at_k(all_val_preds, all_val_labels, k=10)
            
            # 输出详细的训练信息
            logger.info(f'Epoch {epoch+1}/{num_epochs}:')
            logger.info(f'训练集 - 损失: {avg_train_loss:.4f}')
            logger.info(f'验证集 - 损失: {avg_val_loss:.4f}, MSE: {val_mse:.4f}')
            logger.info(f'排序指标:')
            logger.info(f'  - P@1:  {p_at_1:.4f}  NDCG@1:  {ndcg_at_1:.4f}')
            logger.info(f'  - P@3:  {p_at_3:.4f}  NDCG@3:  {ndcg_at_3:.4f}')
            logger.info(f'  - P@5:  {p_at_5:.4f}  NDCG@5:  {ndcg_at_5:.4f}')
            logger.info(f'  - P@10: {p_at_10:.4f}  NDCG@10: {ndcg_at_10:.4f}')
            
            # 检查是否需要保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_metrics = {
                    'loss': avg_val_loss,
                    'mse': val_mse,
                    'p@1': p_at_1,
                    'p@3': p_at_3,
                    'p@5': p_at_5,
                    'p@10': p_at_10,
                    'ndcg@1': ndcg_at_1,
                    'ndcg@3': ndcg_at_3,
                    'ndcg@5': ndcg_at_5,
                    'ndcg@10': ndcg_at_10
                }
                # 保存最佳模型状态
                best_model_state = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': avg_val_loss,
                    'metrics': best_metrics
                }
                self.save_model()
                logger.info("√ 保存了新的最佳模型")
                logger.info(f"  - 验证损失: {best_val_loss:.4f}")
                logger.info(f"  - P@1: {p_at_1:.4f}, P@3: {p_at_3:.4f}")
                logger.info(f"  - P@5: {p_at_5:.4f}, P@10: {p_at_10:.4f}")
            
            # 早停检查
            if avg_val_loss < best_val_loss_for_stopping - min_delta:
                # 如果验证损失有显著改善
                best_val_loss_for_stopping = avg_val_loss
                patience_counter = 0
                logger.info("! 验证损失显著改善，重置早停计数器")
            else:
                # 如果验证损失没有显著改善
                patience_counter += 1
                logger.info(f"! 验证损失未改善，早停计数：{patience_counter}/{patience}")
            
            if patience_counter >= patience:
                logger.info(f"\n早停触发！连续 {patience} 个epoch验证损失未改善")
                logger.info("正在恢复最佳模型状态...")
                # 恢复最佳模型状态
                self.model.load_state_dict(best_model_state['model_state_dict'])
                optimizer.load_state_dict(best_model_state['optimizer_state_dict'])
                logger.info(f"训练结束于第 {epoch+1} 个epoch")
                logger.info(f"最佳验证损失: {best_val_loss:.4f}")
                logger.info(f"最佳P@1: {best_metrics['p@1']:.4f}")
                logger.info(f"最佳P@3: {best_metrics['p@3']:.4f}")
                logger.info(f"最佳P@5: {best_metrics['p@5']:.4f}")
                logger.info(f"最佳P@10: {best_metrics['p@10']:.4f}")
                break
            
            logger.info("-" * 60)
        
        # 如果没有触发早停，也输出最终的最佳结果
        if epoch == num_epochs - 1:
            logger.info("\n达到最大训练轮数，训练结束")
            logger.info(f"最佳验证损失: {best_val_loss:.4f}")
            logger.info(f"最佳P@1: {best_metrics['p@1']:.4f}")
            logger.info(f"最佳P@3: {best_metrics['p@3']:.4f}")
            logger.info(f"最佳P@5: {best_metrics['p@5']:.4f}")
            logger.info(f"最佳P@10: {best_metrics['p@10']:.4f}")

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