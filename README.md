# 🏥 Clinical Appointment No-Show Prediction

This project implements a Machine Learning-based healthcare operations system to predict the likelihood of patient appointment no-shows using historical scheduling data.

It was developed as part of the Intro to GenAI Capstone (Milestone 1) at NST Sonipat.


## 🎯 Objective

Design and implement a supervised ML system that:

- Accepts historical appointment data
- Performs preprocessing and feature engineering
- Predicts appointment no-show probability
- Provides a working UI for real-time inference

No LLMs were used in model training (Traditional ML only).


## 📂 Dataset

Dataset: Kaggle - Medical Appointment No Shows  
Total Records: 110,000+ appointments  
Target Variable: No-show (Binary Classification)

The dataset includes demographic, medical condition, and scheduling features.


## 🔧 Technical Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- Streamlit


## 🧠 Models Used

- Logistic Regression  
- Decision Tree (Best Model Selected using F1 Score)

Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix


### 🏆 Model Selection

Due to class imbalance in the dataset, F1-score was used as the primary evaluation metric.

The Decision Tree model achieved the highest F1-score and was selected as the final deployment model.


## 📊 Input Features Used in Deployment

The deployed model uses the following features:

- Gender
- Age
- Scholarship
- Hypertension
- Diabetes
- Alcoholism
- Handcap
- SMS_received
- waiting_days (Lead Time)
- appointment_day_of_week


## 🏗️ Project Structure
```text
.
├── app.py
├── requirements.txt
├── data/
│   └── KaggleV2-May-2016.csv
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


## 📋 Notes
- Model and scaler files are expected to exist before running the app.
- The app performs binary prediction where:
  - `1` => `No-Show`
  - `0` => `Will Attend`

---

# 🔗 Hosted Link
https://genai-clinical-no-show-prediction.streamlit.app/