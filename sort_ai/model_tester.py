import os
import sys
import torch
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import pandas as pd
from tqdm import tqdm
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sort_ai.model_backup import IRRankingModel, FusionRankingModel
from sort_ai.model_trainer_backup import SearchSessionDataset
from Database.model import Work, Author, WorkAuthorship

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTester:
    """模型测试器，用于评估和对比不同排序方法的效果"""
    
    def __init__(self, model_path=None, fusion_weights=None):
        """
        初始化测试器
        
        Args:
            model_path: 模型路径，如果为None则只测试基础排序方法
            fusion_weights: 融合模型权重列表，如果提供则测试多个融合比例
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 初始化TF-IDF向量化器
        self.vectorizer = None
        
        # 如果提供了模型路径，加载模型
        self.model = None
        self.fusion_models = {}
        
        if model_path and os.path.exists(model_path):
            try:
                # 加载原始模型
                self.model = IRRankingModel()
                checkpoint = torch.load(model_path)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                self.model = self.model.to(self.device)
                self.model.eval()
                logger.info(f"成功加载模型: {model_path}")
                
                # 如果提供了融合权重，创建融合模型
                if fusion_weights:
                    for weight in fusion_weights:
                        fusion_model = FusionRankingModel(base_weight=weight)
                        fusion_model.semantic_model = self.model
                        fusion_model = fusion_model.to(self.device)
                        fusion_model.eval()
                        self.fusion_models[weight] = fusion_model
                        logger.info(f"创建融合模型，基础权重: {weight}")
                else:
                    # 默认创建一个0.5权重的融合模型
                    fusion_model = FusionRankingModel(base_weight=0.5)
                    fusion_model.semantic_model = self.model
                    fusion_model = fusion_model.to(self.device)
                    fusion_model.eval()
                    self.fusion_models[0.5] = fusion_model
                    logger.info("创建默认融合模型，基础权重: 0.5")
                    
            except Exception as e:
                logger.error(f"加载模型失败: {str(e)}")
                self.model = None
    
    def basic_relevance_score(self, query: str, doc_text: str) -> float:
        """
        计算基础相关性得分（TF-IDF + 余弦相似度）
        
        Args:
            query: 查询文本
            doc_text: 文档文本
            
        Returns:
            float: 相关性得分
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # 创建TF-IDF向量化器
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            token_pattern=r'\b\w+\b'
        )
        
        # 构建文档集合
        texts = [query, doc_text]
        
        # 计算TF-IDF矩阵
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            # 计算余弦相似度
            similarity = cosine_similarity(
                tfidf_matrix[0:1], 
                tfidf_matrix[1:2]
            )[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"计算基础相关性得分出错: {str(e)}")
            return 0.0
    
    def model_relevance_score(self, query: str, doc_text: str) -> float:
        """
        使用模型计算相关性得分
        
        Args:
            query: 查询文本
            doc_text: 文档文本
            
        Returns:
            float: 相关性得分
        """
        if not self.model:
            return 0.0
            
        try:
            with torch.no_grad():
                score = self.model(query, doc_text)
                return float(score.cpu().numpy())
        except Exception as e:
            logger.error(f"计算模型相关性得分出错: {str(e)}")
            return 0.0
    
    def fusion_relevance_score(self, query: str, doc_text: str, weight: float) -> float:
        """
        使用融合模型计算相关性得分
        
        Args:
            query: 查询文本
            doc_text: 文档文本
            weight: 使用的融合权重
            
        Returns:
            float: 相关性得分
        """
        if weight not in self.fusion_models:
            return 0.0
            
        fusion_model = self.fusion_models[weight]
        try:
            with torch.no_grad():
                score = fusion_model(query, doc_text)
                return float(score.cpu().numpy())
        except Exception as e:
            logger.error(f"计算融合模型相关性得分出错: {str(e)}")
            return 0.0
    
    def evaluate_ranking(self, 
                        test_sessions: List[Dict],
                        method: str = 'basic',
                        fusion_weight: float = None) -> Dict:
        """
        评估排序方法的效果
        
        Args:
            test_sessions: 测试会话列表
            method: 排序方法，'basic'、'model' 或 'fusion'
            fusion_weight: 融合权重，仅当method为'fusion'时使用
            
        Returns:
            Dict: 评估指标
        """
        total_queries = len(test_sessions)
        if total_queries == 0:
            return {
                'mrr': 0.0,
                'accuracy': 0.0,
                'recall_at_k': {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0},
                'mean_rank': 0.0,
                'total_queries': 0
            }
            
        mrr_sum = 0.0
        accuracy_sum = 0.0
        recall_at_k = {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0}
        all_ranks = []
        
        # 使用tqdm显示进度
        method_desc = f"{method}"
        if method == 'fusion' and fusion_weight is not None:
            method_desc = f"{method}(w={fusion_weight})"
            
        for session in tqdm(test_sessions, desc=f"评估{method_desc}排序"):
            query = session['query']
            pos_docs = session['pos_docs']
            neg_docs = session['neg_docs']
            
            # 计算所有文档的得分
            doc_scores = []
            
            # 计算正样本得分
            for doc in pos_docs:
                if method == 'basic':
                    score = self.basic_relevance_score(query, doc)
                elif method == 'model':
                    score = self.model_relevance_score(query, doc)
                elif method == 'fusion':
                    score = self.fusion_relevance_score(query, doc, fusion_weight)
                doc_scores.append((score, 1))  # 1表示正样本
                
            # 计算负样本得分
            for doc in neg_docs:
                if method == 'basic':
                    score = self.basic_relevance_score(query, doc)
                elif method == 'model':
                    score = self.model_relevance_score(query, doc)
                elif method == 'fusion':
                    score = self.fusion_relevance_score(query, doc, fusion_weight)
                doc_scores.append((score, 0))  # 0表示负样本
            
            # 按得分降序排序
            doc_scores.sort(key=lambda x: x[0], reverse=True)
            
            # 找到第一个正样本的排名
            for rank, (_, is_positive) in enumerate(doc_scores, 1):
                if is_positive:
                    all_ranks.append(rank)
                    mrr_sum += 1.0 / rank
                    
                    # 计算准确率（是否排在第一位）
                    if rank == 1:
                        accuracy_sum += 1
                    
                    # 计算不同K值的召回率
                    for k in recall_at_k.keys():
                        if rank <= k:
                            recall_at_k[k] += 1
                    break
        
        # 计算平均指标
        mrr = mrr_sum / total_queries
        accuracy = (accuracy_sum / total_queries) * 100
        mean_rank = sum(all_ranks) / len(all_ranks)
        
        # 计算召回率百分比
        for k in recall_at_k:
            recall_at_k[k] = (recall_at_k[k] / total_queries) * 100
            
        return {
            'mrr': mrr,
            'accuracy': accuracy,
            'recall_at_k': recall_at_k,
            'mean_rank': mean_rank,
            'total_queries': total_queries
        }
    
    def compare_methods(self, test_sessions: List[Dict]) -> Dict:
        """
        对比不同排序方法的效果
        
        Args:
            test_sessions: 测试会话列表
            
        Returns:
            Dict: 各方法的评估指标
        """
        logger.info("开始对比排序方法...")
        
        # 评估基础排序方法
        logger.info("评估基础排序方法...")
        basic_metrics = self.evaluate_ranking(test_sessions, 'basic')
        
        results = {
            'basic': basic_metrics,
            'model': None,
            'fusion': {}
        }
        
        # 如果模型可用，评估模型排序方法
        if self.model:
            logger.info("评估模型排序方法...")
            model_metrics = self.evaluate_ranking(test_sessions, 'model')
            results['model'] = model_metrics
        
        # 如果融合模型可用，评估融合模型
        for weight in self.fusion_models:
            logger.info(f"评估融合模型(权重={weight})...")
            fusion_metrics = self.evaluate_ranking(test_sessions, 'fusion', weight)
            results['fusion'][str(weight)] = fusion_metrics
        
        # 输出对比结果
        self._print_comparison(results)
        
        return results
    
    def _print_comparison(self, results: Dict):
        """打印对比结果"""
        logger.info("\n排序方法对比结果:")
        logger.info("-" * 50)
        
        methods = ['basic']
        if results['model']:
            methods.append('model')
        
        # 添加融合模型结果
        fusion_weights = list(results['fusion'].keys())
        for weight in fusion_weights:
            methods.append(f"fusion_{weight}")
        
        # 打印每种方法的指标
        for method in methods:
            if method == 'basic':
                metrics = results['basic']
            elif method == 'model':
                metrics = results['model']
            elif method.startswith('fusion_'):
                weight = method.split('_')[1]
                metrics = results['fusion'][weight]
            else:
                continue
                
            logger.info(f"\n{method.upper()}排序方法:")
            logger.info(f"总查询数: {metrics['total_queries']}")
            logger.info(f"MRR: {metrics['mrr']:.4f}")
            logger.info(f"准确率: {metrics['accuracy']:.2f}%")
            logger.info(f"平均排名: {metrics['mean_rank']:.2f}")
            logger.info("召回率:")
            for k, recall in metrics['recall_at_k'].items():
                logger.info(f"  @{k}: {recall:.2f}%")
        
        # 计算相对于基础方法的提升
        basic = results['basic']
        logger.info("\n相对基础方法的提升:")
        
        if results['model']:
            model = results['model']
            logger.info("\n模型方法:")
            mrr_improve = ((model['mrr'] - basic['mrr']) / basic['mrr']) * 100
            acc_improve = model['accuracy'] - basic['accuracy']
            rank_improve = ((basic['mean_rank'] - model['mean_rank']) / basic['mean_rank']) * 100
            
            logger.info(f"MRR提升: {mrr_improve:+.2f}%")
            logger.info(f"准确率提升: {acc_improve:+.2f}%")
            logger.info(f"平均排名提升: {rank_improve:+.2f}%")
            
            logger.info("召回率提升:")
            for k in basic['recall_at_k'].keys():
                improve = model['recall_at_k'][k] - basic['recall_at_k'][k]
                logger.info(f"  @{k}: {improve:+.2f}%")
        
        # 打印融合模型的提升
        for weight in fusion_weights:
            fusion = results['fusion'][weight]
            logger.info(f"\n融合模型(权重={weight}):")
            mrr_improve = ((fusion['mrr'] - basic['mrr']) / basic['mrr']) * 100
            acc_improve = fusion['accuracy'] - basic['accuracy']
            rank_improve = ((basic['mean_rank'] - fusion['mean_rank']) / basic['mean_rank']) * 100
            
            logger.info(f"MRR提升: {mrr_improve:+.2f}%")
            logger.info(f"准确率提升: {acc_improve:+.2f}%")
            logger.info(f"平均排名提升: {rank_improve:+.2f}%")
            
            logger.info("召回率提升:")
            for k in basic['recall_at_k'].keys():
                improve = fusion['recall_at_k'][k] - basic['recall_at_k'][k]
                logger.info(f"  @{k}: {improve:+.2f}%")

    def test_tfidf_ranking(self, test_sessions: List[Dict]) -> Dict:
        """
        专门测试TF-IDF排序的效果
        
        Args:
            test_sessions: 测试会话列表
            
        Returns:
            Dict: 评估指标
        """
        logger.info("开始评估TF-IDF排序效果...")
        
        if not test_sessions:
            logger.warning("没有测试数据")
            return {
                'mrr': 0.0,
                'accuracy': 0.0,
                'recall_at_k': {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0},
                'mean_rank': 0.0,
                'total_queries': 0
            }
        
        # 初始化TF-IDF向量化器
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                token_pattern=r'\b\w+\b'
            )
        
        total_queries = len(test_sessions)
        mrr_sum = 0.0
        accuracy_sum = 0.0
        recall_at_k = {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0}
        all_ranks = []
        
        # 使用tqdm显示进度
        for session in tqdm(test_sessions, desc="评估TF-IDF排序"):
            query = session['query']
            pos_docs = session['pos_docs']
            neg_docs = session['neg_docs']
            
            # 构建文档集合
            all_docs = pos_docs + neg_docs
            doc_labels = [1] * len(pos_docs) + [0] * len(neg_docs)
            
            if not all_docs:
                continue
                
            try:
                # 计算TF-IDF矩阵
                tfidf_matrix = self.vectorizer.fit_transform([query] + all_docs)
                
                # 计算查询与所有文档的相似度
                similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
                
                # 将相似度和标签打包并排序
                doc_scores = list(zip(similarities[0], doc_labels))
                doc_scores.sort(key=lambda x: x[0], reverse=True)
                
                # 找到第一个正样本的排名
                for rank, (_, is_positive) in enumerate(doc_scores, 1):
                    if is_positive:
                        all_ranks.append(rank)
                        mrr_sum += 1.0 / rank
                        
                        # 计算准确率（是否排在第一位）
                        if rank == 1:
                            accuracy_sum += 1
                        
                        # 计算不同K值的召回率
                        for k in recall_at_k.keys():
                            if rank <= k:
                                recall_at_k[k] += 1
                        break
                        
            except Exception as e:
                logger.error(f"处理会话时出错: {str(e)}")
                continue
        
        # 计算平均指标
        mrr = mrr_sum / total_queries
        accuracy = (accuracy_sum / total_queries) * 100
        mean_rank = sum(all_ranks) / len(all_ranks) if all_ranks else 0
        
        # 计算召回率百分比
        for k in recall_at_k:
            recall_at_k[k] = (recall_at_k[k] / total_queries) * 100
        
        results = {
            'mrr': mrr,
            'accuracy': accuracy,
            'recall_at_k': recall_at_k,
            'mean_rank': mean_rank,
            'total_queries': total_queries
        }
        
        # 输出结果
        logger.info("\nTF-IDF排序评估结果:")
        logger.info(f"总查询数: {total_queries}")
        logger.info(f"MRR: {mrr:.4f}")
        logger.info(f"准确率: {accuracy:.2f}%")
        logger.info(f"平均排名: {mean_rank:.2f}")
        logger.info("召回率:")
        for k, recall in recall_at_k.items():
            logger.info(f"  @{k}: {recall:.2f}%")
        
        return results

def main():
    """主函数"""
    try:
        # 设置模型路径
        sort_ai_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(sort_ai_dir, 'models', 'best_model.pth')
        
        # 让用户选择操作模式
        print("\n请选择操作模式：")
        print("1. 测试TF-IDF排序（仅测试集）")
        print("2. 测试TF-IDF排序（全部数据）")
        print("3. 测试深度学习模型（仅测试集）")
        print("4. 测试深度学习模型（仅训练集）")
        print("5. 测试深度学习模型（训练集和测试集）")
        print("6. 测试融合模型")
        print("7. 对比所有方法")
        choice = input("请输入选项（1-7）: ").strip()
        
        # 准备测试数据
        from sort_ai.model_trainer_backup import ModelTrainer
        trainer = ModelTrainer()
        train_sessions, test_sessions = trainer.prepare_training_data()
        
        if not test_sessions:
            logger.error("准备测试数据失败")
            return
        
        # 创建测试器
        fusion_weights = [0.3, 0.5, 0.7, 0.9] if choice in ['6', '7'] else None
        tester = ModelTester(model_path if choice not in ['1', '2'] else None, fusion_weights)
        
        results = {}
        
        if choice == '1':
            # 仅测试集上测试TF-IDF
            logger.info(f"在测试集上评估TF-IDF（{len(test_sessions)}个会话）...")
            results['tfidf_test'] = tester.test_tfidf_ranking(test_sessions)
        elif choice == '2':
            # 在全部数据上测试TF-IDF
            all_sessions = train_sessions + test_sessions
            logger.info(f"在全部数据上评估TF-IDF（{len(all_sessions)}个会话）...")
            results['tfidf_all'] = tester.test_tfidf_ranking(all_sessions)
            
            # 分别记录训练集和测试集的结果
            logger.info(f"\n单独评估训练集（{len(train_sessions)}个会话）...")
            results['tfidf_train'] = tester.test_tfidf_ranking(train_sessions)
            
            logger.info(f"\n单独评估测试集（{len(test_sessions)}个会话）...")
            results['tfidf_test'] = tester.test_tfidf_ranking(test_sessions)
        elif choice == '3':
            # 仅在测试集上测试深度学习模型
            logger.info(f"在测试集上评估深度学习模型（{len(test_sessions)}个会话）...")
            results['model_test'] = tester.evaluate_ranking(test_sessions, 'model')
        elif choice == '4':
            # 仅在训练集上测试深度学习模型
            logger.info(f"在训练集上评估深度学习模型（{len(train_sessions)}个会话）...")
            results['model_train'] = tester.evaluate_ranking(train_sessions, 'model')
            
            # 打印详细结果
            metrics = results['model_train']
            logger.info("\n训练集评估结果:")
            logger.info("-" * 50)
            logger.info(f"总查询数: {metrics['total_queries']}")
            logger.info(f"MRR: {metrics['mrr']:.4f}")
            logger.info(f"准确率: {metrics['accuracy']:.2f}%")
            logger.info(f"平均排名: {metrics['mean_rank']:.2f}")
            logger.info("召回率:")
            for k, recall in metrics['recall_at_k'].items():
                logger.info(f"  @{k}: {recall:.2f}%")
                
        elif choice == '5':
            # 在训练集和测试集上测试深度学习模型
            logger.info(f"在训练集上评估深度学习模型（{len(train_sessions)}个会话）...")
            results['model_train'] = tester.evaluate_ranking(train_sessions, 'model')
            
            logger.info(f"\n在测试集上评估深度学习模型（{len(test_sessions)}个会话）...")
            results['model_test'] = tester.evaluate_ranking(test_sessions, 'model')
            
            # 在全部数据上测试
            all_sessions = train_sessions + test_sessions
            logger.info(f"\n在全部数据上评估深度学习模型（{len(all_sessions)}个会话）...")
            results['model_all'] = tester.evaluate_ranking(all_sessions, 'model')
            
            # 打印对比结果
            logger.info("\n各数据集结果对比:")
            logger.info("-" * 50)
            for dataset, metrics in results.items():
                logger.info(f"\n{dataset.upper()}:")
                logger.info(f"总查询数: {metrics['total_queries']}")
                logger.info(f"MRR: {metrics['mrr']:.4f}")
                logger.info(f"准确率: {metrics['accuracy']:.2f}%")
                logger.info(f"平均排名: {metrics['mean_rank']:.2f}")
                logger.info("召回率:")
                for k, recall in metrics['recall_at_k'].items():
                    logger.info(f"  @{k}: {recall:.2f}%")
                    
            # 计算训练集和测试集的差异
            train_metrics = results['model_train']
            test_metrics = results['model_test']
            
            logger.info("\n训练集和测试集的差异:")
            logger.info("-" * 50)
            mrr_diff = ((train_metrics['mrr'] - test_metrics['mrr']) / test_metrics['mrr']) * 100
            acc_diff = train_metrics['accuracy'] - test_metrics['accuracy']
            rank_diff = ((test_metrics['mean_rank'] - train_metrics['mean_rank']) / test_metrics['mean_rank']) * 100
            
            logger.info(f"MRR差异: {mrr_diff:+.2f}%")
            logger.info(f"准确率差异: {acc_diff:+.2f}%")
            logger.info(f"平均排名差异: {rank_diff:+.2f}%")
            
            logger.info("召回率差异:")
            for k in train_metrics['recall_at_k'].keys():
                diff = train_metrics['recall_at_k'][k] - test_metrics['recall_at_k'][k]
                logger.info(f"  @{k}: {diff:+.2f}%")
                
        elif choice == '6':
            # 仅测试融合模型
            for weight in fusion_weights:
                results[f'fusion_{weight}'] = tester.evaluate_ranking(
                    test_sessions, 'fusion', weight)
        else:
            # 对比所有方法
            results = tester.compare_methods(test_sessions)
            # 添加在训练集上的结果
            logger.info("\n在训练集上评估各方法...")
            results['model_train'] = tester.evaluate_ranking(train_sessions, 'model')
            results['tfidf_train'] = tester.test_tfidf_ranking(train_sessions)
            for weight in fusion_weights:
                results[f'fusion_train_{weight}'] = tester.evaluate_ranking(
                    train_sessions, 'fusion', weight)
        
        # 保存结果
        output_dir = os.path.join(sort_ai_dir, 'test_results')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_dir, f'ranking_results_{timestamp}.json')
        
        # 将结果保存为JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            def convert_to_serializable(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(results, f, indent=4, ensure_ascii=False, default=convert_to_serializable)
            
        logger.info(f"\n结果已保存到: {output_file}")
        
    except Exception as e:
        logger.error(f"测试过程出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 