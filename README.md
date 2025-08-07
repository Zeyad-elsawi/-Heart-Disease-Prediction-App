# ❤️ Heart Disease Prediction App

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)

---

## Overview

This project is an **AI-powered web application** for predicting the risk of heart disease based on patient medical data. It combines a robust machine learning pipeline (Logistic Regression, PCA, RFE, and one-hot encoding) with a user-friendly Streamlit interface.

- **Instant risk prediction** for heart disease
- **Interactive UI** for entering patient data
- **Model pipeline**: Preprocessing, feature selection, and prediction
- **Confidence score** for each prediction

---

## Features

- 📝 **Easy Data Entry**: Enter age, sex, cholesterol, ECG, and more
- ⚡ **Fast Prediction**: Get results instantly with model confidence
- 📊 **Modern UI**: Clean, responsive Streamlit interface
- 🔒 **No data stored**: All predictions are local and private

---

## Tech Stack

- **Python 3.8+**
- **Streamlit** (UI)
- **scikit-learn** (ML pipeline: Logistic Regression, PCA, RFE)
- **pandas, numpy** (data handling)
- **joblib** (model serialization)

---

## Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
cd Ui
streamlit run app.py
```

### 4. Open in your browser
Go to [http://localhost:8501](http://localhost:8501) to use the app.

---

## File Structure
```
├── Data/                # Raw and processed datasets
├── NoteBooks/           # Jupyter notebooks (EDA, modeling, etc.)
│   └── heart_disease_model.pkl  # Trained ML pipeline
├── Ui/
│   └── app.py           # Streamlit web app
├── models/              # (Optional) Model files
├── .gitignore
├── README.md
└── requirements.txt     # (You should create this!)
```

---

## Example Prediction

![Sample UI Screenshot](docs/sample_screenshot.png)

*Above: Enter patient data and get instant risk prediction with confidence.*

---

## Credits

- **Dataset**: UCI Heart Disease Dataset
- **Author**: [Your Name](https://github.com/YOUR_USERNAME)
- **ML/AI**: scikit-learn, pandas, numpy
- **UI**: Streamlit

---

## License

This project is open-source and free to use under the MIT License.

---

> **Tip:** For best results, use the app locally. For deployment (Heroku, Streamlit Cloud, etc.), make sure to include all dependencies and model files.
