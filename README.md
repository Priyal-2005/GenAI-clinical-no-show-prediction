# Clinical Appointment No-Show Prediction

This project provides a Streamlit app to predict whether a patient is likely to miss a clinical appointment.

## What The App Does
- Loads a trained model from `models/best_model.pkl`
- Loads a scaler from `models/scaler.pkl`
- Takes patient details as input
- Predicts:
  - `No-Show` or `Will Attend`
  - Probability of no-show (in %)

## Input Features Used In The App
- `Gender` (`Female`/`Male`)
- `Age` (0 to 100)
- `Hypertension` (`No`/`Yes`)
- `Diabetes` (`No`/`Yes`)
- `Alcoholism` (`No`/`Yes`)
- `Handicap` (`No`/`Yes`)
- `SMS Received` (`No`/`Yes`)
- `Waiting Days (Lead Time)` (0 to 365)

## Project Structure
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

## Setup
```bash
pip install -r requirements.txt
```

## Run The Streamlit App
From the project root:
```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (typically `http://localhost:8501`).

## Notes
- Model and scaler files are expected to exist before running the app.
- The app performs binary prediction where:
  - `1` => `No-Show`
  - `0` => `Will Attend`
