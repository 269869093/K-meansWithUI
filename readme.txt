K-means Clustering Analysis Processor
Project Overview

This is a Python-based K-means clustering algorithm processor that provides a complete workflow from data loading, cleaning, and preprocessing to clustering analysis, result evaluation, and visualization. The project encapsulates common clustering analysis tasks, supports automated search for the optimal number of clusters, and generates detailed analysis reports. It is particularly suitable for scenarios such as mall customer segmentation, retail data analysis, and telecom user grouping.

Features

Complete Data Processing Pipeline: Integrates data loading, exploratory analysis, missing value handling, categorical feature encoding, and numerical feature standardization.

Intelligent Clustering Analysis: Automatically determines the optimal number of clusters (K-value), supporting multiple evaluation methods like the Elbow Method and Silhouette Score.

Comprehensive Result Evaluation: Calculates various clustering evaluation metrics, including Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.

Rich Visualization Output: Provides 2D scatter plots, PCA dimensionality reduction plots, cluster distribution charts, and cluster center heatmaps.

Detailed Report Generation: Automatically generates text reports containing clustering parameters, cluster distribution statistics, and business suggestions.

Modular Design: Clear code structure, easy to extend and customize.

Installation Guide
Prerequisites

Python 3.7+

Required libraries are detailed in requirements.txt

Installation Steps

Clone the Project

bash
git clone https://your-project-repo.git
cd your-project-repo

Create a Virtual Environment (Optional but Recommended)

bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# Or
venv\Scripts\activate    # Windows

Install Dependencies

bash
pip install -r requirements.txt

For offline installation, place the dependency packages in the libs/directory and use the command: pip install --no-index --find-links=libs/ -r requirements.txt.

Core Dependencies

Key libraries include:

pandas

numpy

scikit-learn

matplotlib

seaborn

Quick Start

Here is a basic usage example to help you get started quickly:

python
from kmeans_processor import K_Means_Processor

# Initialize the processor
processor = K_Means_Processor()

# 1. Load data
df = processor.load_data('your_dataset.csv')

# 2. Explore data (Optional)
stats = processor.explore_data(df)

# 3. Data cleaning and preprocessing
df_cleaned, clean_stats = processor.clean_data(df, dataset_type='mall')
df_encoded, encoding_maps = processor.encode_categorical_features(df_cleaned)
df_scaled, scaling_info = processor.scale_features(df_encoded)

# 4. Find the optimal K value and perform clustering
k_results = processor.find_optimal_k(df_scaled, max_k=10, method='both')
clustering_results = processor.perform_clustering(df_scaled, n_clusters=k_results['optimal_k'])

# 5. Save results and generate report
clustered_data_path = processor.save_clustered_data(clustering_results['df_clustered'], 'your_dataset.csv')
report_path = processor.generate_clustering_report(clustering_results)

# 6. Result visualization
fig = processor.create_visualization(clustering_results['df_clustered'])
fig.savefig('clustering_visualization.png')
Usage Instructions
Data Format Requirements

Input data should be in CSV format. Numerical features will be used directly for clustering, while categorical features will be automatically encoded. Ensure the data contains at least two valid numerical features.

Determining the Optimal Number of Clusters (K)

The project provides two main methods to determine the optimal number of clusters (K-value) :

Elbow Method: Determine the K-value by observing the "elbow" point in the curve of the within-cluster sum of squares against different K values.

Silhouette Score Method: Choose the K-value that maximizes the Silhouette Score; a value closer to 1 indicates better clustering performance .

You can select the method by setting the methodparameter to 'elbow', 'silhouette', or 'both'.

Interpreting Results

Silhouette Score: Measures cluster separation; values closer to 1 are better .

Cluster Centers: Describe the typical characteristics of each cluster and are key to understanding the differences between clusters .

Visualization Charts: Intuitively display the distribution of data points among clusters and the positions of cluster centers .

Project Structure
pythonProject/
├── kmeans_processor.py  # Main program, KMeans processor class
├── requirements.txt     # Project dependency list
├── cleaned_data/        # Output directory for cleaned data
├── clustered_data/      # Output directory for clustering result data
└── README.md           # Project documentation
Contributing

We welcome all forms of contribution! Please follow these steps:

Fork the repository.

Create your feature branch (git checkout -b feature/AmazingFeature).

Commit your changes (git commit -m 'Add some AmazingFeature').

Push to the branch (git push origin feature/AmazingFeature).

Open a Pull Request.

License

(Please add your project's license information here, e.g., MIT License)

Contact

Author: Hanfu Liang

Email: shujiansheng7777777@gmail.com

If you have any questions or suggestions, please feel free to contact us via email.