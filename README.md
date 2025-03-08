# Intellihack_Astartes_2

# Customer Segmentation Analysis 📊

![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Overview

This project performs customer segmentation analysis on e-commerce customer behavior data. It identifies distinct customer segments (Bargain Hunters, High Spenders, and Window Shoppers) based on purchasing patterns and browsing behaviors using unsupervised machine learning techniques.

## 📝 Summary

This project applies machine learning techniques to segment e-commerce customers into distinct behavioral groups. Using K-means clustering on purchasing and browsing features, we identified three customer personas:

- **Bargain Hunters**: Customers who make frequent purchases of low-value items and actively use discount codes
- **High Spenders**: Premium customers who make fewer but high-value purchases with minimal discount usage
- **Window Shoppers**: Browsers who spend significant time exploring products but rarely convert to purchases

The segmentation enables targeted marketing strategies tailored to each group's behavior patterns. The analysis pipeline includes data preprocessing, exploratory data analysis, cluster modeling, and visualization techniques to effectively communicate the distinct customer segments.

This analysis provides actionable insights for marketing teams to optimize campaigns, improve user experience, and increase conversion rates through personalized approaches.

## 📑 Table of Contents

- [Project Overview](#overview)
- [Summary](#summary)
- [Dataset Description](#dataset-description)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Visualizations](#visualizations)
- [Contributing](#contributing)

## 📊 Dataset Description

The analysis uses `customer_behavior_analytcis.csv`, containing the following features:

| Feature | Description |
|---------|-------------|
| `customer_id` | Unique identifier for each customer |
| `total_purchases` | Number of purchases made |
| `avg_cart_value` | Average value of items in cart |
| `total_time_spent` | Time spent on the platform (minutes) |
| `product_click` | Number of products viewed |
| `discount_counts` | Number of times discount codes were used |

## 🔧 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/customer-segmentation.git
   cd customer-segmentation
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, install packages individually:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn plotly jupyterlab
   ```

## 🚀 Usage

1. Ensure the dataset `customer_behavior_analytcis.csv` is in the project directory

2. Open the Jupyter notebook:
   ```bash
   jupyter lab
   ```
   or
   ```bash
   jupyter notebook
   ```

3. Open `customer_segmentation.ipynb` and run all cells

4. The segmented customer data will be saved to `customer_segments.csv`

## 🔬 Methodology

The analysis follows these steps:

1. **Data Preprocessing**
   - Missing value imputation
   - Feature scaling using StandardScaler

2. **Exploratory Data Analysis**
   - Distribution analysis of features
   - Correlation analysis
   - Feature relationship exploration

3. **Clustering**
   - K-means algorithm with k=3
   - Validation using Elbow method and Silhouette score

4. **Visualization & Interpretation**
   - PCA-based visualizations
   - Radar charts and comparative plots
   - Mapping clusters to meaningful business segments

## 📈 Results

The analysis identifies three distinct customer segments:

### 1. Bargain Hunters
- **Characteristics**: Frequent purchases of low-value items, heavy discount usage
- **Behavior**: Deal-seekers who make frequent purchases of low-value items and heavily rely on discounts
- **Marketing Strategy**: Offer frequent small discounts, create loyalty programs

### 2. High Spenders
- **Characteristics**: Moderate purchases, high cart value, low discount usage
- **Behavior**: Premium buyers who focus on high-value purchases and are less influenced by discounts
- **Marketing Strategy**: Focus on premium product recommendations, provide exclusive services

### 3. Window Shoppers
- **Characteristics**: Low purchases, high browsing time, high product clicks
- **Behavior**: Browsers who spend significant time exploring but rarely convert
- **Marketing Strategy**: Implement targeted conversion strategies, create urgency with limited-time offers

## 📊 Visualizations

The notebook produces multiple visualizations:

- Feature distributions and correlations
- Cluster visualizations using PCA (2D and 3D)
- Radar charts showing segment profiles
- Comparative bar charts of segments across features

Here are some example visualizations:

![PCA Clusters](https://via.placeholder.com/600x400?text=PCA+Clusters+Visualization)
![Radar Chart](https://via.placeholder.com/600x400?text=Radar+Chart+of+Segments)
![Segment Distribution](https://via.placeholder.com/600x400?text=Segment+Distribution)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

