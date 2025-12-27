#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project    : pythonProject
@FileName   : 11.py
@Author     : Liang Hanfu
@Email      : shujiansheng7777777@gmail.com
@Time       : 2025/12/9 02:04
@Version    : 1.0
@Description: K-means Clustering Graphical Interface
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

# Import algorithm module
from K_means import K_Means_Processor

warnings.filterwarnings('ignore')

"""
    K-means Clustering Graphical Interface
"""
class KMeansUIEN:

    def __init__(self, root):
        self.root = root
        self.root.title("K-means Clustering Tool - Customer Segmentation Project")
        self.root.geometry("1200x700")

        # Initialize algorithm processor
        self.processor = K_Means_Processor()
        self.clustering_results = None

        # Create interface
        self.create_widgets()

    """
        Create interface components
    """
    def create_widgets(self):
        # Set graph font
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create left panel
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))

        # Create right panel
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ==================== Left Panel: Control Area ====================
        control_frame = ttk.LabelFrame(left_panel, text="Data Processing Control", padding=10)
        control_frame.pack(fill=tk.BOTH, pady=(0, 10))

        # File operations section
        ttk.Label(control_frame, text="Data File:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))

        self.file_path_var = tk.StringVar(value="No file selected")
        file_path_label = ttk.Label(control_frame, textvariable=self.file_path_var,
                                   relief=tk.SUNKEN, padding=5, background='white')
        file_path_label.pack(fill=tk.X, pady=(0, 10))

        # File operation buttons
        file_button_frame = ttk.Frame(control_frame)
        file_button_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(file_button_frame, text="Import CSV File",
                  command=self.load_csv, width=15).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_button_frame, text="View Data",
                  command=self.show_data_preview, width=10).pack(side=tk.LEFT)

        # Processing buttons
        ttk.Button(control_frame, text="Run Preprocessing",
                  command=self.run_preprocessing, width=20).pack(fill=tk.X, pady=(5, 5))
        ttk.Button(control_frame, text="Run Clustering",
                  command=self.run_clustering, width=20).pack(fill=tk.X, pady=(5, 5))
        ttk.Button(control_frame, text="Generate Visualization",
                  command=self.generate_visualization, width=20).pack(fill=tk.X, pady=(5, 5))

        ttk.Button(control_frame, text="Save Cleaned Data",
                  command=self.save_cleaned_data, width=20).pack(fill=tk.X, pady=(5, 5))
        ttk.Button(control_frame, text="Save Clustered Data",
                  command=self.save_clustered_data, width=20).pack(fill=tk.X, pady=(5, 5))
        ttk.Button(control_frame, text="Generate Report",
                  command=self.generate_report, width=20).pack(fill=tk.X, pady=(5, 5))

        # Status indicator
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(control_frame, textvariable=self.status_var,
                                 font=('Arial', 9), foreground='green')
        status_label.pack(anchor=tk.W, pady=(10, 0))

        # Progress bar
        self.progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var,
                                      mode='determinate', length=280)
        progress_bar.pack(fill=tk.X, pady=(5, 0))

        # ==================== Right Panel: Display Area ====================
        # Create notebook control
        notebook = ttk.Notebook(right_panel)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Data preview tab
        data_tab = ttk.Frame(notebook)
        notebook.add(data_tab, text="Data Preview")
        self.setup_data_tab(data_tab)

        # Log tab
        log_tab = ttk.Frame(notebook)
        notebook.add(log_tab, text="Processing Log")
        self.setup_log_tab(log_tab)

        # Visualization tab
        viz_tab = ttk.Frame(notebook)
        notebook.add(viz_tab, text="Visualization")
        self.setup_viz_tab(viz_tab)

    def setup_data_tab(self, parent):
        """Set up data preview tab"""
        # Create scrollable text box to display data
        self.data_text = scrolledtext.ScrolledText(parent, wrap=tk.NONE,
                                                  width=80, height=30, font=('Consolas', 9))
        self.data_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def setup_log_tab(self, parent):
        """Set up log tab"""
        # Create log text box
        self.log_text = scrolledtext.ScrolledText(parent, wrap=tk.WORD,
                                                 width=80, height=30, font=('Consolas', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def setup_viz_tab(self, parent):
        """Set up visualization tab"""
        # Create canvas frame
        self.viz_frame = ttk.Frame(parent)
        self.viz_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=5)

    def load_csv(self):
        """Load CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[
                ("CSV Files", "*.csv"),
                ("Text Files", "*.txt"),
                ("All Files", "*.*")
            ]
        )

        if file_path:
            self.file_path = file_path
            self.file_path_var.set(os.path.basename(file_path))

            try:
                self.update_status("Loading data...", "info")
                self.progress_var.set(10)

                # Call algorithm module to load data
                df = self.processor.load_data(file_path)

                if df is not None:
                    # Display in log
                    self.log_message(f"File loaded: {file_path}")
                    self.log_message(f"Data shape: {df.shape}")

                    # Display first few rows of data
                    self.data_text.delete(1.0, tk.END)
                    self.data_text.insert(tk.END, f"Data Preview - {os.path.basename(file_path)}\n")
                    self.data_text.insert(tk.END, "="*50 + "\n\n")
                    self.data_text.insert(tk.END, str(df.head(20)))
                    self.data_text.insert(tk.END, f"\n\n... Total {len(df)} rows")

                    self.update_status("Data loaded successfully!", "success")
                    self.progress_var.set(100)
                else:
                    self.update_status("Data loading failed", "error")

            except Exception as e:
                self.update_status(f"Load failed: {str(e)}", "error")
                self.log_message(f"Error: {str(e)}")
                messagebox.showerror("Error", f"Unable to load file: {str(e)}")

    def run_preprocessing(self):
        """Run preprocessing flow"""
        if self.processor.df_original is None:
            messagebox.showwarning("Warning", "Please import a data file first")
            return

        self.update_status("Starting data preprocessing...", "info")
        self.progress_var.set(10)
        self.log_message("\n" + "="*50)
        self.log_message("Starting data preprocessing")
        self.log_message(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_message("="*50)

        try:
            # Step 1: Data cleaning
            self.update_status("Cleaning data...", "info")
            self.progress_var.set(30)

            df_cleaned, clean_stats = self.processor.clean_data(
                self.processor.df_original,
                'mall'  # Using mall dataset type
            )

            self.log_message("✓ Data cleaning completed")
            self.log_message(f"  Removed {clean_stats.get('duplicates_removed', 0)} duplicate rows")

            # Step 2: Encode categorical variables
            self.update_status("Encoding categorical variables...", "info")
            self.progress_var.set(50)

            df_encoded, encoding_mappings = self.processor.encode_categorical_features(df_cleaned)
            self.log_message("✓ Categorical variable encoding completed")
            if encoding_mappings:
                for col, mapping in encoding_mappings.items():
                    self.log_message(f"  {col} encoding mapping: {mapping}")

            # Step 3: Standardize features
            self.update_status("Standardizing features...", "info")
            self.progress_var.set(70)

            df_scaled, scaling_info = self.processor.scale_features(df_encoded, [])
            self.log_message("✓ Feature standardization completed")
            if scaling_info:
                self.log_message(f"  Standardized {len(scaling_info.get('features', []))} features")

            # Update UI
            self.update_status("Data preprocessing completed!", "success")
            self.progress_var.set(100)

            # Display statistics
            self.log_message(f"\nProcessing results:")
            self.log_message(f"  Original data shape: {self.processor.df_original.shape}")
            self.log_message(f"  Cleaned data shape: {df_scaled.shape}")
            self.log_message(f"  Rows removed: {clean_stats.get('duplicates_removed', 0)}")

            messagebox.showinfo("Success", "Data preprocessing completed!")

        except Exception as e:
            self.update_status(f"Processing failed: {str(e)}", "error")
            self.log_message(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Error during preprocessing: {str(e)}")

    def run_clustering(self):
        """Run K-means clustering"""
        if self.processor.df_scaled is None:
            messagebox.showwarning("Warning", "Please run data preprocessing first")
            return

        self.update_status("Starting K-means clustering...", "info")
        self.progress_var.set(20)
        self.log_message("\n" + "="*50)
        self.log_message("Starting K-means clustering")
        self.log_message(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_message("="*50)

        try:
            # Determine K value
            self.update_status("Determining optimal K value...", "info")
            self.progress_var.set(40)

            k_results = self.processor.find_optimal_k(
                self.processor.df_scaled,
                max_k=10,
                method='both'
            )

            optimal_k = k_results['optimal_k']
            self.log_message(f"Automatically selected K value: {optimal_k}")
            self.log_message(f"  Silhouette score: {k_results.get('best_silhouette', 0):.4f}")

            # Execute clustering
            self.update_status("Executing K-means clustering...", "info")
            self.progress_var.set(70)

            clustering_results = self.processor.perform_clustering(
                self.processor.df_scaled,
                n_clusters=optimal_k
            )

            self.clustering_results = clustering_results

            # Update UI
            self.update_status("K-means clustering completed!", "success")
            self.progress_var.set(100)

            # Display clustering results
            self.log_message(f"\nClustering results:")
            self.log_message(f"  Number of clusters: {clustering_results.get('n_clusters', 0)}")
            self.log_message(f"  Silhouette score: {clustering_results.get('silhouette_score', 0):.4f}")
            self.log_message(f"  Davies-Bouldin index: {clustering_results.get('davies_bouldin_score', 0):.4f}")
            self.log_message(f"  Calinski-Harabasz index: {clustering_results.get('calinski_harabasz_score', 0):.4f}")
            self.log_message(f"  Inertia: {clustering_results.get('inertia', 0):.4f}")

            # Display cluster distribution
            cluster_stats = clustering_results.get('cluster_stats', {})
            for cluster_id, stats in cluster_stats.items():
                self.log_message(f"  Cluster {cluster_id}: {stats.get('size', 0)} samples ({stats.get('percentage', 0):.1f}%)")

            messagebox.showinfo("Success", f"K-means clustering completed!\nOptimal K value: {optimal_k}\nSilhouette score: {clustering_results.get('silhouette_score', 0):.4f}")

        except Exception as e:
            self.update_status(f"Clustering failed: {str(e)}", "error")
            self.log_message(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Error during clustering: {str(e)}")

    def generate_visualization(self):
        """Generate clustering visualization"""
        if not hasattr(self, 'clustering_results') or self.clustering_results is None:
            messagebox.showwarning("Warning", "Please run clustering analysis first")
            return

        try:
            self.update_status("Generating visualization chart...", "info")

            # Clear previous charts
            for widget in self.viz_frame.winfo_children():
                widget.destroy()

            # Create chart
            fig = self.processor.create_visualization(
                self.clustering_results['df_clustered']
            )

            # Display chart in interface
            canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.update_status("Visualization chart generation completed!", "success")
            self.log_message("✓ Visualization chart generated")

        except Exception as e:
            self.update_status(f"Chart generation failed: {str(e)}", "error")
            self.log_message(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Chart generation failed: {str(e)}")

    def save_cleaned_data(self):
        """Save cleaned data"""
        if self.processor.df_cleaned is None:
            messagebox.showwarning("Warning", "No data to save, please run preprocessing first")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Cleaned Data",
            initialfile="cleaned_data.csv",
            defaultextension=".csv",
            filetypes=[
                ("CSV Files", "*.csv"),
                ("All Files", "*.*")
            ]
        )

        if file_path:
            try:
                self.update_status("Saving data...", "info")
                self.processor.df_cleaned.to_csv(file_path, index=False)
                self.update_status("Data saved successfully!", "success")
                self.log_message(f"Cleaned data saved to: {file_path}")
                messagebox.showinfo("Success", f"Data saved to:\n{file_path}")
            except Exception as e:
                self.update_status(f"Save failed: {str(e)}", "error")
                messagebox.showerror("Error", f"Save failed: {str(e)}")

    def save_clustered_data(self):
        """Save clustered data"""
        if not hasattr(self, 'clustering_results') or self.clustering_results is None:
            messagebox.showwarning("Warning", "No clustering results to save")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Clustering Results",
            initialfile="clustered_data.csv",
            defaultextension=".csv",
            filetypes=[
                ("CSV Files", "*.csv"),
                ("All Files", "*.*")
            ]
        )

        if file_path:
            try:
                self.update_status("Saving clustering results...", "info")
                self.clustering_results['df_clustered'].to_csv(file_path, index=False)
                self.update_status("Clustering results saved successfully!", "success")
                self.log_message(f"Clustering results saved to: {file_path}")
                messagebox.showinfo("Success", f"Clustering results saved to:\n{file_path}")
            except Exception as e:
                self.update_status(f"Save failed: {str(e)}", "error")
                messagebox.showerror("Error", f"Save failed: {str(e)}")

    def generate_report(self):
        """Generate data preprocessing report"""
        if not hasattr(self, 'clustering_results') or self.clustering_results is None:
            messagebox.showwarning("Warning", "No clustering results, cannot generate complete report")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Complete Report",
            initialfile="kmeans_clustering_report.txt",
            defaultextension=".txt",
            filetypes=[
                ("Text Files", "*.txt"),
                ("All Files", "*.*")
            ]
        )

        if file_path:
            try:
                # Call algorithm module's unified report generation function
                success = self.processor.generate_clustering_report(
                    self.clustering_results,
                    file_path,
                    include_explanation=True  # Usually need complete explanation and suggestions in UI
                )

                if success:
                    self.log_message(f"Report saved to: {file_path}")
                    messagebox.showinfo("Success", f"Report saved to:\n{file_path}")
                else:
                    messagebox.showerror("Error", "Report generation failed")

            except Exception as e:
                messagebox.showerror("Error", f"Report generation failed: {str(e)}")

    def show_data_preview(self):
        """Display complete data preview"""
        if self.processor.df_original is None:
            messagebox.showwarning("Warning", "No data to preview")
            return

        # Create new window to display complete data
        preview_window = tk.Toplevel(self.root)
        preview_window.title("Complete Data Preview")
        preview_window.geometry("1000x600")

        # Create scrollable text box
        text_widget = scrolledtext.ScrolledText(preview_window, wrap=tk.NONE,
                                               width=120, height=30, font=('Consolas', 9))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Display complete data
        text_widget.insert(tk.END, f"Complete Data Preview - {os.path.basename(self.file_path) if self.file_path else 'Unnamed'}\n")
        text_widget.insert(tk.END, "="*80 + "\n\n")
        text_widget.insert(tk.END, str(self.processor.df_original))

    def log_message(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)  # Auto-scroll to bottom

    def update_status(self, message, status_type="info"):
        """Update status information"""
        self.status_var.set(message)
        self.root.update_idletasks()

def main():
    """Main function"""
    root = tk.Tk()
    app = KMeansUIEN(root)
    root.mainloop()

if __name__ == "__main__":
    main()
