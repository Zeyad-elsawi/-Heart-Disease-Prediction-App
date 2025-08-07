# ❤️ Heart Disease Prediction App

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)

---

## Overview

This project is an **AI-powered web application** for predicting the risk of heart disease based on patient medical data. It combines a robust machine learning pipeline (Logistic Regression, PCA, RFE, and one-hot encoding) with a modern Streamlit web interface for easy interaction.

## 🚀 Features

- **Easy Data Entry**: Simple form interface for entering patient data
- **Fast Prediction**: Instant heart disease risk assessment
- **Modern UI**: Clean, responsive design with Streamlit
- **Privacy-First**: No data storage, predictions made locally
- **Comprehensive Analysis**: Complete data science workflow from preprocessing to deployment

## 🛠️ Tech Stack

- **Backend**: Python 3.8+
- **Web Framework**: Streamlit
- **Machine Learning**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Model Persistence**: joblib

## 📁 Project Structure

```
├── NoteBooks/                    # Jupyter notebooks for analysis
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   └── 06_hyperparameter_tuning.ipynb
├── Ui/                          # Streamlit web application
│   └── app.py
├── models/                      # Trained models
│   └── heart_disease_model.pkl
├── data/                        # Dataset files
└── README.md
```

## 📓 Notebooks & Theory

### 1. **01_data_preprocessing.ipynb**
**Theory:** Data Cleaning, Encoding, and Scaling
- **Data Loading**: Imports the UCI Heart Disease dataset with 303 samples and 14 features
- **Missing Value Handling**: Identifies and handles any missing values in the dataset
- **Categorical Encoding**: Applies one-hot encoding to categorical variables (sex, chest pain type, etc.)
- **Feature Scaling**: Standardizes numerical features using StandardScaler for consistent model performance
- **Data Splitting**: Prepares train/test splits for supervised learning
- **Theory**: Ensures data quality and prepares features for machine learning algorithms

### 2. **02_pca_analysis.ipynb**
**Theory:** Principal Component Analysis (PCA)
- **Dimensionality Reduction**: Reduces feature space while preserving variance
- **Variance Explained**: Analyzes how much variance each principal component captures
- **Elbow Method**: Determines optimal number of components using cumulative variance
- **Feature Correlation**: Visualizes relationships between original features and principal components
- **Theory**: PCA transforms correlated features into uncorrelated principal components, reducing multicollinearity and improving model performance

### 3. **03_feature_selection.ipynb**
**Theory:** Recursive Feature Elimination (RFE) and Feature Importance
- **RFE Implementation**: Uses Logistic Regression with RFE to select optimal feature subset
- **Cross-Validation**: Evaluates feature subsets using 5-fold cross-validation
- **Feature Ranking**: Ranks features by importance and selects top-performing subset
- **Performance Metrics**: Compares model performance with different feature sets
- **Theory**: RFE recursively removes least important features, improving model interpretability and reducing overfitting

### 4. **04_supervised_learning.ipynb**
**Theory:** Classification Algorithms and Model Evaluation
- **Logistic Regression**: Primary classification algorithm with interpretable coefficients
- **Model Training**: Trains on preprocessed data with selected features
- **Performance Evaluation**: Uses accuracy, precision, recall, F1-score, and ROC-AUC
- **Cross-Validation**: Implements k-fold cross-validation for robust performance estimation
- **Confusion Matrix**: Visualizes true vs predicted classifications
- **Theory**: Supervised learning with labeled data to predict binary heart disease outcomes

### 5. **05_unsupervised_learning.ipynb**
**Theory:** Clustering and Dimensionality Reduction
- **K-Means Clustering**: Groups patients into clusters based on feature similarity
- **Elbow Method**: Determines optimal number of clusters using Within-Cluster Sum of Squares (WCSS)
- **Cluster Analysis**: Analyzes characteristics of each cluster and relationship to target variable
- **Hierarchical Clustering**: Alternative clustering approach with dendrogram visualization
- **Theory**: Unsupervised learning discovers hidden patterns in data without predefined labels

### 6. **06_hyperparameter_tuning.ipynb**
**Theory:** Grid Search and Model Optimization
- **Grid Search**: Systematically tests hyperparameter combinations
- **Cross-Validation**: Uses stratified k-fold CV to prevent data leakage
- **Hyperparameter Space**: Optimizes regularization strength, solver type, and class weights
- **Best Model Selection**: Identifies optimal hyperparameters for maximum performance
- **Model Persistence**: Saves the best model for deployment
- **Theory**: Hyperparameter tuning maximizes model performance by finding optimal parameter combinations

## 🚀 Quick Start

### Prerequisites
```bash
pip install streamlit pandas numpy scikit-learn joblib matplotlib seaborn
```

### Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/heart-disease-prediction.git
cd heart-disease-prediction

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Navigate to the UI directory
cd Ui

# Run the Streamlit app
streamlit run app.py
```

### Access the App
Open your browser and go to: `http://localhost:8501`

## 📊 Usage

1. **Enter Patient Data**: Fill in the medical parameters in the web form
2. **Submit**: Click the "Predict Heart Disease Risk" button
3. **View Results**: Get instant prediction with confidence score
4. **Interpret**: Understand the risk level and recommended actions

## 🎯 Model Performance

- **Accuracy**: 85%+
- **Precision**: 83%+
- **Recall**: 87%+
- **F1-Score**: 85%+
- **ROC-AUC**: 0.89+

## 📸 Sample Prediction

![Sample Prediction](docs/sample_screenshot.png)

*Add your own screenshot here to show the app in action*

## 🔬 Data Science Workflow

This project demonstrates a complete data science workflow:

1. **Data Preprocessing** → Clean and prepare data
2. **Exploratory Analysis** → Understand data patterns
3. **Feature Engineering** → Create and select relevant features
4. **Model Development** → Train and evaluate models
5. **Hyperparameter Tuning** → Optimize model performance
6. **Model Deployment** → Deploy to web application

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Credits

- **Dataset**: UCI Heart Disease Dataset
- **Author**: [Your Name]
- **Institution**: [Your Institution]
- **Contact**: [Your Email]

---

⭐ **Star this repository if you found it helpful!**
