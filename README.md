# 🏥 Clinical Appointment No-Show Prediction

This project implements a Machine Learning-based healthcare operations system to predict the likelihood of patient appointment no-shows using historical scheduling data.

Developed as part of the Intro to GenAI Capstone (Milestone 1) at NST Sonipat.


## 🎯 Objective

Design and implement a supervised ML system that:

- Accepts historical appointment data
- Performs preprocessing and feature engineering
- Predicts appointment no-show probability
- Provides a working UI for real-time inference

No LLMs were used in model training (Traditional ML only).


## 📂 Dataset

Original Dataset: Kaggle – Medical Appointment No Shows  
Total Records: 110,000+ appointments  
Target Variable: No-show (Binary Classification)

### 🔄 Dataset Modifications

The original dataset contained only **2 null values**, which was not sufficient to properly demonstrate missing value handling techniques.

To simulate real-world healthcare data scenarios:

- The dataset was imported into **Google Sheets**
- Using **Google Apps Script**, additional null values were programmatically introduced into selected columns
- The modified dataset was exported and used for preprocessing and model training

This allowed proper demonstration of:
- Missing value detection
- Null handling strategies
- Data cleaning pipeline

Final dataset used in the project:
data/KaggleV2-May-2016-v2.csv


## 🔧 Technical Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- Streamlit


## 🧠 Models Evaluated

- Logistic Regression  
- Decision Tree  
- Random Forest  

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix


## 🏆 Model Selection Strategy

In healthcare systems, **false negatives are costly**.  
Failing to predict a no-show leads to:

- Wasted doctor time
- Unused appointment slots
- Revenue loss
- Operational inefficiency

Therefore, **Recall was selected as the primary evaluation metric.**

### 📈 Model Comparison (Tuned Models)

- Tuned Random Forest Recall: **87.5%**
- Tuned Decision Tree Recall: **86.9%**

Although Random Forest achieved slightly higher recall, the difference was marginal (~0.6%).

### ✅ Final Model Selected

The **Tuned Decision Tree** was selected as the final deployment model due to:

- High recall performance
- Better interpretability
- Transparent decision paths
- Suitability for healthcare environments

In clinical settings, model explainability is often more important than marginal performance improvements.


## 📊 Deployment Features

The deployed model uses:

- Gender
- Age
- Scholarship
- Hipertension
- Diabetes
- Alcoholism
- Handcap
- SMS_received
- waiting_days
- appointment_day_of_week


## 🏗️ Project Structure
```text
.
├── app.py
├── requirements.txt
├── data/
│   └── KaggleV2-May-2016-v2.csv
├── models/
│   ├── best_model.pkl
│   └── scaler.pkl
└── notebook/
    └── GenAI_MidSem_Project_4.ipynb
```


## ⚙️ Setup Instructions
Install dependencies:
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Run The Streamlit App
From the project root:
```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (typically `http://localhost:8501`)


## 📋 Predicted Output
- Model and scaler files are expected to exist before running the app.
- The app performs binary prediction where:
  - `1` => `No-Show`
  - `0` => `Will Attend`

---

# 🔗 Hosted Link
https://genai-clinical-no-show-prediction.streamlit.app/