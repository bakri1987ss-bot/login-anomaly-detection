# Login Anomaly Detection

**Login Anomaly Detection** is a machine learning project that detects abnormal login activities using user login features. The project demonstrates end-to-end ML workflow including feature engineering, model training, evaluation, and alert generation.

## Features

- Preprocessing of login data
- Feature engineering (failed login streaks, counts in different time windows, etc.)
- Machine learning model training with hyperparameter tuning
- Model evaluation: precision, recall, F1-score, ROC AUC, PR AUC
- Generation of alerts and feedback logging
- Production metrics recording

## Technologies Used

- Python 3.12
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib, Seaborn
- Jupyter Notebook

## File Structure

login-anomaly/
│
├── data/ # Raw and processed datasets
├── logs/ # Logs and alerts logs
├── models/ # Trained ML models
├── reports/ # Generated reports
├── scripts/ # Python scripts (feature engineering, model training, etc.)
├── tests/ # Unit tests
├── run_pipeline.py # Main script to run the pipeline
├── requirements.txt # Project dependencies
└── README.md # Project documentation

bash
Copy code

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/bakri1987ss-bot/login-anomaly-detection.git
cd login-anomaly-detection

2. Create a virtual environment:

python3 -m venv sec-env
source sec-env/bin/activate

3. Install dependencies:

pip install -r requirements.txt

4. Run the pipeline:

python3 run_pipeline.py

5. Check generated alerts:

cat data/alerts.csv

Testing
Run unit tests:
PYTHONPATH=$PWD pytest tests/

License
This project is licensed under the MIT License.
