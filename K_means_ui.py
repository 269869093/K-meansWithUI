#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project    : pythonProject
@FileName   : K_means_ui.py
@Author     : 梁瀚夫
@Email      : shujiansheng7777777@gmail.com
@Time       : 2025/12/27 05:16
@Version    : 1.0
@Description: 
"""

import tkinter as tk
from tkinter import messagebox
import importlib.util
import sys
import os


class LanguageSelector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Language Selection / 语言选择")
        self.root.geometry("300x150")
        self.root.resizable(False, False)

        # 居中显示窗口
        self.center_window()

        self.selected_language = None

        self.create_widgets()

    def center_window(self):
        """将窗口居中显示"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def create_widgets(self):
        """创建界面控件"""
        # 主标题
        title_label = tk.Label(
            self.root,
            text="Please select language / 请选择语言",
            font=("Arial", 12, "bold"),
            pady=10
        )
        title_label.pack()

        # 按钮框架
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)

        # 英文按钮
        en_button = tk.Button(
            button_frame,
            text="English",
            font=("Arial", 10),
            width=10,
            height=2,
            bg="#4CAF50",
            fg="white",
            command=self.select_english
        )
        en_button.pack(side=tk.LEFT, padx=10)

        # 中文按钮
        cn_button = tk.Button(
            button_frame,
            text="中文",
            font=("Arial", 10),
            width=10,
            height=2,
            bg="#2196F3",
            fg="white",
            command=self.select_chinese
        )
        cn_button.pack(side=tk.LEFT, padx=10)

        # 版权信息
        copyright_label = tk.Label(
            self.root,
            text="© 2025 K-Means Cluster Tool",
            font=("Arial", 8),
            fg="gray"
        )
        copyright_label.pack(side=tk.BOTTOM, pady=5)

    def select_english(self):
        """选择英文版本"""
        self.selected_language = 'en'
        self.root.quit()
        self.root.destroy()

    def select_chinese(self):
        """选择中文版本"""
        self.selected_language = 'cn'
        self.root.quit()
        self.root.destroy()

    def get_language(self):
        """获取用户选择的语言"""
        self.root.mainloop()
        return self.selected_language


def load_module(module_name):
    """动态加载模块"""
    try:
        # 检查模块文件是否存在
        if not os.path.exists(f"{module_name}.py"):
            raise FileNotFoundError(f"Module file {module_name}.py not found")

        # 动态导入模块
        spec = importlib.util.spec_from_file_location(module_name, f"{module_name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load module: {str(e)}")
        return None


def main():
    """主函数"""
    # 创建语言选择器
    selector = LanguageSelector()
    language = selector.get_language()

    if language is None:
        messagebox.showwarning("Warning", "No language selected. Exiting...")
        return

    # 根据选择加载对应的模块
    if language == 'cn':
        module_name = "K_means_ui_cn"
        messagebox.showinfo("Info", "正在加载中文界面...")
    else:
        module_name = "K_means_ui_en"
        messagebox.showinfo("Info", "Loading English interface...")

    # 加载并运行对应模块
    module = load_module(module_name)
    if module:
        try:
            # 假设每个模块都有一个main函数
            if hasattr(module, 'main'):
                module.main()
            else:
                messagebox.showerror("Error", f"Module {module_name} does not have a main function")
        except Exception as e:
            messagebox.showerror("Error", f"Error running module: {str(e)}")
    else:
        messagebox.showerror("Error", f"Failed to load {module_name}.py")


if __name__ == "__main__":
    main()