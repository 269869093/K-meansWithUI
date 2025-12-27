#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project    : pythonProject
@FileName   : 11.py
@Author     : 梁瀚夫
@Email      : shujiansheng7777777@gmail.com
@Time       : 2025/12/9 02:04
@Version    : 1.0
@Description: K-means聚类图形界面
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import warnings
import os
from datetime import datetime
import pandas as pd
import numpy as np

# 导入算法模块
from K_means import K_Means_Processor

warnings.filterwarnings('ignore')

"""
    K-means聚类图形界面
"""
class KMeansUI:

    def __init__(self, root):
        self.root = root
        self.root.title("K-means聚类工具 - 客户细分项目")
        self.root.geometry("1200x700")

        # 初始化算法处理器
        self.processor = K_Means_Processor()
        self.clustering_results = None

        # 创建界面
        self.create_widgets()

    """
        创建界面组件
    """
    def create_widgets(self):
        # 设置图形字体
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 创建左侧面板
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))

        # 创建右侧面板
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ==================== 左侧面板：控制区域 ====================
        control_frame = ttk.LabelFrame(left_panel, text="数据处理控制", padding=10)
        control_frame.pack(fill=tk.BOTH, pady=(0, 10))

        # 文件操作部分
        ttk.Label(control_frame, text="数据文件:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))

        self.file_path_var = tk.StringVar(value="未选择文件")
        file_path_label = ttk.Label(control_frame, textvariable=self.file_path_var,
                                   relief=tk.SUNKEN, padding=5, background='white')
        file_path_label.pack(fill=tk.X, pady=(0, 10))

        # 文件操作按钮
        file_button_frame = ttk.Frame(control_frame)
        file_button_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(file_button_frame, text="导入CSV文件",
                  command=self.load_csv, width=15).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_button_frame, text="查看数据",
                  command=self.show_data_preview, width=10).pack(side=tk.LEFT)

        # 处理按钮
        ttk.Button(control_frame, text="运行预处理",
                  command=self.run_preprocessing, width=20).pack(fill=tk.X, pady=(5, 5))
        ttk.Button(control_frame, text="运行聚类",
                  command=self.run_clustering, width=20).pack(fill=tk.X, pady=(5, 5))
        ttk.Button(control_frame, text="生成可视化",
                  command=self.generate_visualization, width=20).pack(fill=tk.X, pady=(5, 5))

        ttk.Button(control_frame, text="保存清洗后数据",
                  command=self.save_cleaned_data, width=20).pack(fill=tk.X, pady=(5, 5))
        ttk.Button(control_frame, text="保存聚类结果",
                  command=self.save_clustered_data, width=20).pack(fill=tk.X, pady=(5, 5))
        ttk.Button(control_frame, text="生成报告",
                  command=self.generate_report, width=20).pack(fill=tk.X, pady=(5, 5))

        # 状态指示器
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(control_frame, textvariable=self.status_var,
                                 font=('Arial', 9), foreground='green')
        status_label.pack(anchor=tk.W, pady=(10, 0))

        # 进度条
        self.progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var,
                                      mode='determinate', length=280)
        progress_bar.pack(fill=tk.X, pady=(5, 0))

        # ==================== 右侧面板：显示区域 ====================
        # 创建选项卡控件
        notebook = ttk.Notebook(right_panel)
        notebook.pack(fill=tk.BOTH, expand=True)

        # 数据预览选项卡
        data_tab = ttk.Frame(notebook)
        notebook.add(data_tab, text="数据预览")
        self.setup_data_tab(data_tab)

        # 日志选项卡
        log_tab = ttk.Frame(notebook)
        notebook.add(log_tab, text="处理日志")
        self.setup_log_tab(log_tab)

        # 可视化选项卡
        viz_tab = ttk.Frame(notebook)
        notebook.add(viz_tab, text="可视化")
        self.setup_viz_tab(viz_tab)

    def setup_data_tab(self, parent):
        """设置数据预览选项卡"""
        # 创建滚动文本框显示数据
        self.data_text = scrolledtext.ScrolledText(parent, wrap=tk.NONE,
                                                  width=80, height=30, font=('Consolas', 9))
        self.data_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def setup_log_tab(self, parent):
        """设置日志选项卡"""
        # 创建日志文本框
        self.log_text = scrolledtext.ScrolledText(parent, wrap=tk.WORD,
                                                 width=80, height=30, font=('Consolas', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def setup_viz_tab(self, parent):
        """设置可视化选项卡"""
        # 创建画布框架
        self.viz_frame = ttk.Frame(parent)
        self.viz_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=5)

    def load_csv(self):
        """加载CSV文件"""
        file_path = filedialog.askopenfilename(
            title="选择CSV文件",
            filetypes=[
                ("CSV文件", "*.csv"),
                ("文本文件", "*.txt"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            self.file_path = file_path
            self.file_path_var.set(os.path.basename(file_path))

            try:
                self.update_status("正在加载数据...", "info")
                self.progress_var.set(10)

                # 调用算法模块加载数据
                df = self.processor.load_data(file_path)

                if df is not None:
                    # 在日志中显示
                    self.log_message(f"已加载文件: {file_path}")
                    self.log_message(f"数据形状: {df.shape}")

                    # 显示前几行数据
                    self.data_text.delete(1.0, tk.END)
                    self.data_text.insert(tk.END, f"数据预览 - {os.path.basename(file_path)}\n")
                    self.data_text.insert(tk.END, "="*50 + "\n\n")
                    self.data_text.insert(tk.END, str(df.head(20)))
                    self.data_text.insert(tk.END, f"\n\n... 共 {len(df)} 行")

                    self.update_status("数据加载成功!", "success")
                    self.progress_var.set(100)
                else:
                    self.update_status("数据加载失败", "error")

            except Exception as e:
                self.update_status(f"加载失败: {str(e)}", "error")
                self.log_message(f"错误: {str(e)}")
                messagebox.showerror("错误", f"无法加载文件: {str(e)}")

    def run_preprocessing(self):
        """运行预处理流程"""
        if self.processor.df_original is None:
            messagebox.showwarning("警告", "请先导入数据文件")
            return

        self.update_status("开始数据预处理...", "info")
        self.progress_var.set(10)
        self.log_message("\n" + "="*50)
        self.log_message("开始数据预处理")
        self.log_message(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_message("="*50)

        try:
            # 步骤1: 数据清洗
            self.update_status("清洗数据...", "info")
            self.progress_var.set(30)

            df_cleaned, clean_stats = self.processor.clean_data(
                self.processor.df_original,
                'mall'  # 使用商场数据集类型
            )

            self.log_message("✓ 数据清洗完成")
            self.log_message(f"  删除了 {clean_stats.get('duplicates_removed', 0)} 个重复行")

            # 步骤2: 编码分类变量
            self.update_status("编码分类变量...", "info")
            self.progress_var.set(50)

            df_encoded, encoding_mappings = self.processor.encode_categorical_features(df_cleaned)
            self.log_message("✓ 分类变量编码完成")
            if encoding_mappings:
                for col, mapping in encoding_mappings.items():
                    self.log_message(f"  {col} 编码映射: {mapping}")

            # 步骤3: 标准化特征
            self.update_status("标准化特征...", "info")
            self.progress_var.set(70)

            df_scaled, scaling_info = self.processor.scale_features(df_encoded, [])
            self.log_message("✓ 特征标准化完成")
            if scaling_info:
                self.log_message(f"  标准化了 {len(scaling_info.get('features', []))} 个特征")

            # 更新UI
            self.update_status("数据预处理完成!", "success")
            self.progress_var.set(100)

            # 显示统计信息
            self.log_message(f"\n处理结果:")
            self.log_message(f"  原始数据形状: {self.processor.df_original.shape}")
            self.log_message(f"  清洗后数据形状: {df_scaled.shape}")
            self.log_message(f"  删除的行数: {clean_stats.get('duplicates_removed', 0)}")

            messagebox.showinfo("成功", "数据预处理完成!")

        except Exception as e:
            self.update_status(f"处理失败: {str(e)}", "error")
            self.log_message(f"错误: {str(e)}")
            messagebox.showerror("错误", f"预处理过程中出错: {str(e)}")

    def run_clustering(self):
        """运行K-means聚类"""
        if self.processor.df_scaled is None:
            messagebox.showwarning("警告", "请先运行数据预处理")
            return

        self.update_status("开始K-means聚类...", "info")
        self.progress_var.set(20)
        self.log_message("\n" + "="*50)
        self.log_message("开始K-means聚类")
        self.log_message(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_message("="*50)

        try:
            # 确定K值
            self.update_status("确定最优K值...", "info")
            self.progress_var.set(40)

            k_results = self.processor.find_optimal_k(
                self.processor.df_scaled,
                max_k=10,
                method='both'
            )

            optimal_k = k_results['optimal_k']
            self.log_message(f"自动选择K值: {optimal_k}")
            self.log_message(f"  轮廓系数: {k_results.get('best_silhouette', 0):.4f}")

            # 执行聚类
            self.update_status("执行K-means聚类...", "info")
            self.progress_var.set(70)

            clustering_results = self.processor.perform_clustering(
                self.processor.df_scaled,
                n_clusters=optimal_k
            )

            self.clustering_results = clustering_results

            # 更新UI
            self.update_status("K-means聚类完成!", "success")
            self.progress_var.set(100)

            # 显示聚类结果
            self.log_message(f"\n聚类结果:")
            self.log_message(f"  聚类数量: {clustering_results.get('n_clusters', 0)}")
            self.log_message(f"  轮廓系数: {clustering_results.get('silhouette_score', 0):.4f}")
            self.log_message(f"  Davies-Bouldin指数: {clustering_results.get('davies_bouldin_score', 0):.4f}")
            self.log_message(f"  Calinski-Harabasz指数: {clustering_results.get('calinski_harabasz_score', 0):.4f}")
            self.log_message(f"  惯性: {clustering_results.get('inertia', 0):.4f}")

            # 显示聚类分布
            cluster_stats = clustering_results.get('cluster_stats', {})
            for cluster_id, stats in cluster_stats.items():
                self.log_message(f"  聚类 {cluster_id}: {stats.get('size', 0)} 个样本 ({stats.get('percentage', 0):.1f}%)")

            messagebox.showinfo("成功", f"K-means聚类完成!\n最优K值: {optimal_k}\n轮廓系数: {clustering_results.get('silhouette_score', 0):.4f}")

        except Exception as e:
            self.update_status(f"聚类失败: {str(e)}", "error")
            self.log_message(f"错误: {str(e)}")
            messagebox.showerror("错误", f"聚类过程中出错: {str(e)}")

    def generate_visualization(self):
        """生成聚类可视化"""
        if not hasattr(self, 'clustering_results') or self.clustering_results is None:
            messagebox.showwarning("警告", "请先运行聚类分析")
            return

        try:
            self.update_status("生成可视化图表...", "info")

            # 清除之前的图表
            for widget in self.viz_frame.winfo_children():
                widget.destroy()

            # 创建图表
            fig = self.processor.create_visualization(
                self.clustering_results['df_clustered']
            )

            # 在界面中显示图表
            canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.update_status("可视化图表生成完成!", "success")
            self.log_message("✓ 可视化图表已生成")

        except Exception as e:
            self.update_status(f"生成图表失败: {str(e)}", "error")
            self.log_message(f"错误: {str(e)}")
            messagebox.showerror("错误", f"生成图表失败: {str(e)}")

    def save_cleaned_data(self):
        """保存清洗后的数据"""
        if self.processor.df_cleaned is None:
            messagebox.showwarning("警告", "没有可保存的数据，请先运行预处理")
            return

        file_path = filedialog.asksaveasfilename(
            title="保存清洗后的数据",
            initialfile="cleaned_data.csv",
            defaultextension=".csv",
            filetypes=[
                ("CSV文件", "*.csv"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            try:
                self.update_status("正在保存数据...", "info")
                self.processor.df_cleaned.to_csv(file_path, index=False)
                self.update_status("数据保存成功!", "success")
                self.log_message(f"清洗后数据已保存到: {file_path}")
                messagebox.showinfo("成功", f"数据已保存到:\n{file_path}")
            except Exception as e:
                self.update_status(f"保存失败: {str(e)}", "error")
                messagebox.showerror("错误", f"保存失败: {str(e)}")

    def save_clustered_data(self):
        """保存聚类后的数据"""
        if not hasattr(self, 'clustering_results') or self.clustering_results is None:
            messagebox.showwarning("警告", "没有可保存的聚类结果")
            return

        file_path = filedialog.asksaveasfilename(
            title="保存聚类结果",
            initialfile="clustered_data.csv",
            defaultextension=".csv",
            filetypes=[
                ("CSV文件", "*.csv"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            try:
                self.update_status("正在保存聚类结果...", "info")
                self.clustering_results['df_clustered'].to_csv(file_path, index=False)
                self.update_status("聚类结果保存成功!", "success")
                self.log_message(f"聚类结果已保存到: {file_path}")
                messagebox.showinfo("成功", f"聚类结果已保存到:\n{file_path}")
            except Exception as e:
                self.update_status(f"保存失败: {str(e)}", "error")
                messagebox.showerror("错误", f"保存失败: {str(e)}")

    def generate_report(self):
        """生成数据预处理报告"""
        if not hasattr(self, 'clustering_results') or self.clustering_results is None:
            messagebox.showwarning("警告", "没有聚类结果，无法生成完整报告")
            return

        file_path = filedialog.asksaveasfilename(
            title="保存完整报告",
            initialfile="kmeans_clustering_report.txt",
            defaultextension=".txt",
            filetypes=[
                ("文本文件", "*.txt"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            try:
                # 调用算法模块的统一报告生成函数
                success = self.processor.generate_clustering_report(
                    self.clustering_results,
                    file_path,
                    include_explanation=True  # UI中通常需要完整的解释和建议
                )

                if success:
                    self.log_message(f"报告已保存到: {file_path}")
                    messagebox.showinfo("成功", f"报告已保存到:\n{file_path}")
                else:
                    messagebox.showerror("错误", "报告生成失败")

            except Exception as e:
                messagebox.showerror("错误", f"生成报告失败: {str(e)}")

    def show_data_preview(self):
        """显示完整数据预览"""
        if self.processor.df_original is None:
            messagebox.showwarning("警告", "没有数据可预览")
            return

        # 创建新窗口显示完整数据
        preview_window = tk.Toplevel(self.root)
        preview_window.title("完整数据预览")
        preview_window.geometry("1000x600")

        # 创建滚动文本框
        text_widget = scrolledtext.ScrolledText(preview_window, wrap=tk.NONE,
                                               width=120, height=30, font=('Consolas', 9))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 显示完整数据
        text_widget.insert(tk.END, f"完整数据预览 - {os.path.basename(self.file_path) if self.file_path else '未命名'}\n")
        text_widget.insert(tk.END, "="*80 + "\n\n")
        text_widget.insert(tk.END, str(self.processor.df_original))

    def log_message(self, message):
        """在日志中添加消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)  # 自动滚动到底部

    def update_status(self, message, status_type="info"):
        """更新状态信息"""
        self.status_var.set(message)
        self.root.update_idletasks()

def main():
    """主函数"""
    root = tk.Tk()
    app = KMeansUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()