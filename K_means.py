#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project    : pythonProject
@FileName   : 11.py
@Author     : 梁瀚夫
@Email      : shujiansheng7777777@gmail.com
@Time       : 2025/12/9 02:04
@Version    : 1.0
@Description: K-means聚类算法实现
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import warnings
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("whitegrid")

"""
    K-means聚类处理器
"""
class K_Means_Processor:

    def __init__(self):
        self.df_original = None
        self.df_cleaned = None
        self.df_scaled = None
        self.file_path = None
        self.cluster_labels = None
        self.cluster_centers = None
        self.kmeans_model = None
        self.best_k = None

    """
        加载CSV数据文件
        参数:
        file_path: CSV文件路径
        返回:
        pandas DataFrame
    """
    def load_data(self, file_path):
        try:
            # 尝试使用不同编码方式读取文件
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file_path, encoding='latin-1')
                except:
                    df = pd.read_csv(file_path, encoding='ISO-8859-1')

            self.df_original = df
            self.file_path = file_path

            print(f"数据加载成功! 数据集形状: {df.shape}")
            return df

        except Exception as e:
            print(f"数据加载失败: {e}")
            return None

    """
        数据探索和基本统计分析
        参数:
        df: pandas DataFrame
        返回:
        包含统计信息的字典
    """
    def explore_data(self, df):

        stats = {}
        if df is None or df.empty:
            return stats
        # 基本统计信息
        stats['shape'] = df.shape
        stats['columns'] = list(df.columns)
        stats['dtypes'] = df.dtypes.to_dict()
        # 缺失值统计
        missing_values = df.isnull().sum()
        missing_percent = (df.isnull().sum() / len(df)) * 100
        missing_df = pd.DataFrame({
            '缺失值数量': missing_values,
            '缺失值比例(%)': missing_percent
        })
        stats['missing_values'] = missing_df[missing_df['缺失值数量'] > 0].to_dict()
        # 重复行统计
        stats['duplicates'] = df.duplicated().sum()
        # 数值列描述性统计
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats['numeric_stats'] = df[numeric_cols].describe().to_dict()
        return stats

    """
        数据清洗函数
        参数:
        df: 原始数据DataFrame
        dataset_type: 数据集类型
        返回:
        清洗后的DataFrame和统计信息
    """
    def clean_data(self, df, dataset_type='mall'):
        if df is None or df.empty:
            return df, {'duplicates_removed': 0, 'initial_rows': 0, 'final_rows': 0}
        df_clean = df.copy()
        # 处理缺失值
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    # 数值型列用中位数填充
                    median_val = df_clean[col].median()
                    df_clean[col].fillna(median_val, inplace=True)
                else:
                    # 分类型列用众数填充
                    mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                    df_clean[col].fillna(mode_val, inplace=True)
        # 删除重复行
        initial_rows = len(df_clean)
        df_clean.drop_duplicates(inplace=True, ignore_index=True)
        final_rows = len(df_clean)
        stats = {
            'duplicates_removed': initial_rows - final_rows,
            'initial_rows': initial_rows,
            'final_rows': final_rows
        }
        # 根据数据集类型进行特定的清洗
        if dataset_type == 'mall':
            df_clean = self._clean_mall_data(df_clean)
        elif dataset_type == 'retail':
            df_clean = self._clean_retail_data(df_clean)
        elif dataset_type == 'telecom':
            df_clean = self._clean_telecom_data(df_clean)
        self.df_cleaned = df_clean
        return df_clean, stats


    def _clean_mall_data(self, df):
        # 处理商场客户数据
        # 标准化列名
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
        # 重命名列
        column_mapping = {
            'annual_income_(k$)': 'annual_income',
            'spending_score_(1-100)': 'spending_score',
            'annual_income': 'annual_income',
            'spending_score': 'spending_score'
        }

        for old, new in column_mapping.items():
            if old in df.columns:
                df.rename(columns={old: new}, inplace=True)

        return df

    def _clean_retail_data(self, df):
        # 处理零售数据
        return df

    def _clean_telecom_data(self, df):
        # 处理电信数据
        return df
    """
        对分类特征进行编码
        参数:
        df: 清洗后的DataFrame
        返回:
        编码后的DataFrame和编码映射
    """
    def encode_categorical_features(self, df):
        if df is None or df.empty:
            return df, {}
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
        encoding_mappings = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            # 保存编码映射
            unique_values = df[col].unique()
            encoded_values = le.transform(unique_values)
            encoding_mappings[col] = dict(zip(unique_values, encoded_values))
        return df_encoded, encoding_mappings

    """
        对数值特征进行标准化
        参数:
        df: 编码后的DataFrame
        exclude_cols: 不需要标准化的列
        返回:
        标准化后的DataFrame和标准化信息
    """
    def scale_features(self, df, exclude_cols=None):
        if exclude_cols is None:
            exclude_cols = []
        df_scaled = df.copy()
        numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        scaling_info = {}
        if numerical_cols:
            scaler = StandardScaler()
            df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            scaling_info = {
                'mean': scaler.mean_.tolist(),
                'scale': scaler.scale_.tolist(),
                'features': numerical_cols
            }
        self.df_scaled = df_scaled
        return df_scaled, scaling_info

    """
        找到最优的聚类数量K
        参数:
        df: 标准化后的数据
        max_k: 最大K值
        method: 方法 ('elbow', 'silhouette', 'both')
        返回:
        最优K值和评估指标
    """
    def find_optimal_k(self, df, max_k=10, method='both'):
        if df is None or len(df) == 0:
            raise ValueError("数据为空")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            raise ValueError("需要至少2个数值特征进行聚类")
        # 准备数据
        X = df[numeric_cols].values
        # 计算不同K值的指标
        inertias = []
        silhouette_scores = []
        db_scores = []
        k_values = list(range(2, max_k + 1))
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            if len(X) > k:  # 需要足够的样本来计算轮廓系数
                try:
                    silhouette_avg = silhouette_score(X, kmeans.labels_)
                except:
                    silhouette_avg = 0
                silhouette_scores.append(silhouette_avg)
                try:
                    db_score = davies_bouldin_score(X, kmeans.labels_)
                except:
                    db_score = np.inf
                db_scores.append(db_score)
            else:
                silhouette_scores.append(0)
                db_scores.append(np.inf)
        # 根据方法选择最佳K
        if method == 'elbow':
            # 肘部法则：找到拐点
            if len(inertias) > 1:
                diffs = np.diff(inertias)
                if len(diffs) > 1:
                    diff_ratios = diffs[1:] / diffs[:-1]
                    optimal_k = np.argmin(diff_ratios) + 3  # 加3是因为我们从k=2开始
                else:
                    optimal_k = 2
            else:
                optimal_k = 2
        elif method == 'silhouette':
            # 轮廓系数：越大越好
            silhouette_scores_array = np.array(silhouette_scores)
            optimal_k = k_values[np.argmax(silhouette_scores_array)]
        else:  # 'both'
            # 结合肘部法则和轮廓系数
            if len(inertias) > 1 and len(silhouette_scores) > 1:
                # 标准化指标
                inertias_norm = (np.array(inertias) - np.min(inertias)) / (np.max(inertias) - np.min(inertias) + 1e-10)
                silhouette_norm = (np.array(silhouette_scores) - np.min(silhouette_scores)) / (np.max(silhouette_scores) - np.min(silhouette_scores) + 1e-10)
                # 组合得分（我们希望惯性小，轮廓系数大）
                combined_scores = silhouette_norm - inertias_norm
                optimal_k = k_values[np.argmax(combined_scores)]
            else:
                optimal_k = 2
        # 确保最优K在范围内
        optimal_k = max(2, min(optimal_k, max_k))
        results = {
            'optimal_k': optimal_k,
            'k_values': k_values,
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'db_scores': db_scores,
            'best_silhouette': silhouette_scores[optimal_k-2] if optimal_k >= 2 and len(silhouette_scores) > optimal_k-2 else 0
        }
        self.best_k = optimal_k
        return results

    """
        执行K-means聚类
        参数:
        df: 标准化后的数据
        n_clusters: 聚类数量，如果为None则使用最优K
        返回:
        聚类结果
    """
    def perform_clustering(self, df, n_clusters=None):
        if df is None or len(df) == 0:
            raise ValueError("数据为空")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            raise ValueError("需要至少2个数值特征进行聚类")
        X = df[numeric_cols].values
        if n_clusters is None:
            if self.best_k is None:
                # 如果没有指定K，自动寻找
                k_results = self.find_optimal_k(df, max_k=10)
                n_clusters = k_results['optimal_k']
            else:
                n_clusters = self.best_k
        # 执行K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        cluster_labels = kmeans.fit_predict(X)
        cluster_centers = kmeans.cluster_centers_
        # 计算评估指标
        silhouette_avg = silhouette_score(X, cluster_labels)
        db_score = davies_bouldin_score(X, cluster_labels)
        ch_score = calinski_harabasz_score(X, cluster_labels)
        # 将聚类标签添加到数据中
        df_clustered = df.copy()
        df_clustered['cluster'] = cluster_labels
        # 保存结果
        self.cluster_labels = cluster_labels
        self.cluster_centers = cluster_centers
        self.kmeans_model = kmeans
        # 分析每个聚类的特征
        cluster_stats = self._analyze_clusters(df_clustered, numeric_cols)
        results = {
            'n_clusters': n_clusters,
            'cluster_labels': cluster_labels,
            'cluster_centers': cluster_centers,
            'inertia': kmeans.inertia_,
            'silhouette_score': silhouette_avg,
            'davies_bouldin_score': db_score,
            'calinski_harabasz_score': ch_score,
            'df_clustered': df_clustered,
            'cluster_stats': cluster_stats
        }
        return results

    def _analyze_clusters(self, df_clustered, feature_cols):
        """分析每个聚类的特征"""
        cluster_stats = {}
        if df_clustered is None or 'cluster' not in df_clustered.columns:
            return cluster_stats
        for cluster_id in sorted(df_clustered['cluster'].unique()):
            cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
            stats = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df_clustered) * 100
            }
            # 计算每个特征的平均值
            for col in feature_cols:
                if col in cluster_data.columns and col != 'cluster':
                    stats[f'{col}_mean'] = cluster_data[col].mean()
                    stats[f'{col}_std'] = cluster_data[col].std()
            cluster_stats[cluster_id] = stats
        return cluster_stats

    """
        保存清洗后的数据
        参数:
        df: 清洗后的DataFrame
        original_path: 原始文件路径
        output_dir: 输出目录
        返回:
        保存的文件路径
    """
    def save_cleaned_data(self, df, original_path, output_dir='cleaned_data'):
        os.makedirs(output_dir, exist_ok=True)
        original_filename = os.path.basename(original_path)
        cleaned_filename = f"cleaned_{original_filename}"
        output_path = os.path.join(output_dir, cleaned_filename)
        df.to_csv(output_path, index=False)
        return output_path

    """
        保存聚类结果
        参数:
        df_clustered: 包含聚类标签的DataFrame
        original_path: 原始文件路径
        output_dir: 输出目录
        返回:
        保存的文件路径
    """
    def save_clustered_data(self, df_clustered, original_path, output_dir='clustered_data'):
        os.makedirs(output_dir, exist_ok=True)
        original_filename = os.path.basename(original_path)
        clustered_filename = f"clustered_{original_filename}"
        output_path = os.path.join(output_dir, clustered_filename)
        df_clustered.to_csv(output_path, index=False)
        return output_path

    """
        生成聚类分析报告
        参数:
        clustering_results: 聚类结果字典
        report_path: 报告文件路径
        include_explanation: 是否包含评估指标解释和建议部分
    """
    def generate_clustering_report(self, clustering_results, report_path='clustering_report.txt',
                                   include_explanation=True):
        try:
            with open(report_path, 'w', encoding='utf-8', errors='ignore') as f:
                # 报告头部
                f.write("=" * 60 + "\n")
                f.write("K-means聚类分析报告\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.flush()

                # 聚类参数
                f.write("1. 聚类参数\n")
                f.write("-" * 30 + "\n")
                f.write(f"   聚类数量 (K): {clustering_results.get('n_clusters', 0)}\n")
                f.write(f"   轮廓系数: {clustering_results.get('silhouette_score', 0):.4f}\n")
                f.write(f"   Davies-Bouldin指数: {clustering_results.get('davies_bouldin_score', 0):.4f}\n")
                f.write(f"   Calinski-Harabasz指数: {clustering_results.get('calinski_harabasz_score', 0):.4f}\n")
                f.write(f"   惯性 (Inertia): {clustering_results.get('inertia', 0):.4f}\n\n")
                f.flush()

                # 聚类分布
                f.write("2. 聚类分布\n")
                f.write("-" * 30 + "\n")
                cluster_stats = clustering_results.get('cluster_stats', {})

                for cluster_id, stats in cluster_stats.items():
                    f.write(f"   聚类 {cluster_id}:\n")
                    f.write(f"     样本数: {stats.get('size', 0)} ({stats.get('percentage', 0):.1f}%)\n")

                    # 显示特征的平均值（如果存在）
                    feature_means = [(k, v) for k, v in stats.items()
                                     if k.endswith('_mean') and k != 'size' and k != 'percentage']

                    for feature_name, mean_value in feature_means[:3]:  # 只显示前3个特征
                        feature = feature_name.replace('_mean', '')
                        std_value = stats.get(f'{feature}_std', 0)
                        f.write(f"     {feature}: {mean_value:.2f} (±{std_value:.2f})\n")

                    f.write("\n")
                    f.flush()

                # 聚类中心
                f.write("3. 聚类中心\n")
                f.write("-" * 30 + "\n")
                centers = clustering_results.get('cluster_centers', [])

                # 限制显示数量，避免文件过大
                max_centers_to_show = min(50, len(centers))
                if len(centers) > max_centers_to_show:
                    f.write(f"   注意: 聚类中心数量较多({len(centers)})，只显示前{max_centers_to_show}个\n\n")

                for i, center in enumerate(centers[:max_centers_to_show]):
                    # 安全处理聚类中心数据
                    try:
                        if hasattr(center, 'tolist'):
                            center_data = center.tolist()
                        else:
                            center_data = list(center)

                        # 格式化显示，限制小数位数
                        center_str = "["
                        for j, val in enumerate(center_data):
                            if j > 0:
                                center_str += ", "
                            center_str += f"{val:.6f}"
                        center_str += "]"

                        f.write(f"   聚类 {i} 中心: {center_str}\n")
                    except Exception as e:
                        f.write(f"   聚类 {i} 中心: [数据格式异常]\n")

                    # 定期刷新缓冲区
                    if i % 10 == 0:
                        f.flush()

                f.flush()

                # 可选的解释和建议部分
                if include_explanation:
                    f.write("\n4. 评估指标解释\n")
                    f.write("-" * 30 + "\n")
                    f.write("   轮廓系数: 值在[-1, 1]之间，越接近1表示聚类效果越好\n")
                    f.write("   Davies-Bouldin指数: 值越小表示聚类效果越好\n")
                    f.write("   Calinski-Harabasz指数: 值越大表示聚类效果越好\n")
                    f.write("   惯性: 表示样本到其最近聚类中心的平方距离之和，越小越好\n")
                    f.flush()

                    f.write("\n5. 建议\n")
                    f.write("-" * 30 + "\n")
                    silhouette = clustering_results.get('silhouette_score', 0)
                    if silhouette > 0.7:
                        f.write("   聚类效果优秀，可以考虑增加聚类数量进一步细分\n")
                    elif silhouette > 0.5:
                        f.write("   聚类效果良好，当前聚类数量合适\n")
                    elif silhouette > 0.3:
                        f.write("   聚类效果一般，可以尝试不同的聚类数量或特征组合\n")
                    else:
                        f.write("   聚类效果较差，建议检查数据质量或尝试其他聚类算法\n")
                    f.flush()

                return True

        except IOError as e:
            print(f"文件写入错误: {e}")
            raise
        except Exception as e:
            print(f"生成报告时发生错误: {e}")
            raise
    """
        创建聚类可视化图表
        返回:
        matplotlib图形对象
    """
    def create_visualization(self, df_clustered, feature1=None, feature2=None):
        if df_clustered is None or 'cluster' not in df_clustered.columns:
            raise ValueError("没有聚类结果或数据为空")

        numeric_cols = df_clustered.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'cluster']

        if len(numeric_cols) < 2:
            raise ValueError("需要至少2个数值特征进行可视化")

        fig, axes = plt.subplots(2, 2, figsize=(8, 6))

        # 1. 2D散点图
        if feature1 is not None and feature2 is not None and feature1 in df_clustered.columns and feature2 in df_clustered.columns:
            ax1 = axes[0, 0]
            scatter = ax1.scatter(df_clustered[feature1], df_clustered[feature2],
                                 c=df_clustered['cluster'], cmap='viridis', alpha=0.6)
            ax1.set_xlabel(feature1)
            ax1.set_ylabel(feature2)
            ax1.set_title(f'2D聚类散点图 ({feature1} vs {feature2})')
            plt.colorbar(scatter, ax=ax1)
        elif len(numeric_cols) >= 2:
            # 如果没有指定特征，使用前两个数值特征
            ax1 = axes[0, 0]
            scatter = ax1.scatter(df_clustered[numeric_cols[0]], df_clustered[numeric_cols[1]],
                                 c=df_clustered['cluster'], cmap='viridis', alpha=0.6)
            ax1.set_xlabel(numeric_cols[0])
            ax1.set_ylabel(numeric_cols[1])
            ax1.set_title(f'2D聚类散点图 ({numeric_cols[0]} vs {numeric_cols[1]})')
            plt.colorbar(scatter, ax=ax1)
        else:
            axes[0, 0].text(0.5, 0.5, '没有足够的数据进行可视化',
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('2D聚类散点图')

        # 2. PCA降维可视化
        try:
            ax2 = axes[0, 1]
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(df_clustered[numeric_cols])

            scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1],
                                  c=df_clustered['cluster'], cmap='viridis', alpha=0.6)
            ax2.set_xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax2.set_ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax2.set_title('PCA降维可视化')
            plt.colorbar(scatter2, ax=ax2)
        except Exception as e:
            axes[0, 1].text(0.5, 0.5, f'PCA降维失败:\n{str(e)[:50]}',
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('PCA降维可视化')

        # 3. 聚类大小分布
        ax3 = axes[1, 0]
        cluster_sizes = df_clustered['cluster'].value_counts().sort_index()
        bars = ax3.bar(cluster_sizes.index, cluster_sizes.values)
        ax3.set_xlabel('聚类编号')
        ax3.set_ylabel('样本数量')
        ax3.set_title('聚类大小分布')

        # 在柱状图上显示数值
        for i, (cluster_id, count) in enumerate(cluster_sizes.items()):
            ax3.text(cluster_id, count, str(count), ha='center', va='bottom')

        # 4. 聚类中心热力图
        if self.cluster_centers is not None and len(numeric_cols) > 0:
            ax4 = axes[1, 1]
            # 只显示前5个特征
            n_features = min(5, len(numeric_cols), self.cluster_centers.shape[1])
            centers_df = pd.DataFrame(self.cluster_centers[:, :n_features],
                                     columns=numeric_cols[:n_features])

            im = ax4.imshow(centers_df.T, cmap='YlOrRd', aspect='auto')
            ax4.set_xlabel('聚类编号')
            ax4.set_ylabel('特征')
            ax4.set_yticks(range(len(centers_df.columns)))
            ax4.set_yticklabels(centers_df.columns)
            ax4.set_title(f'聚类中心热力图 (前{n_features}个特征)')
            plt.colorbar(im, ax=ax4)

        plt.tight_layout()
        return fig