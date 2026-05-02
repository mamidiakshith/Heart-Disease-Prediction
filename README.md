# ❤️ Heart Disease Detection using Machine Learning

## 📌 Overview

This project predicts the likelihood of heart disease using structured clinical data. It follows a complete machine learning pipeline including data cleaning, outlier handling, feature engineering, feature selection, model training, and evaluation.

The objective is to build a reliable classification model that can assist in early-stage medical risk assessment.

---

## 🚀 Key Highlights

* End-to-end ML pipeline implementation
* Advanced preprocessing (outlier capping instead of removal)
* Feature scaling using StandardScaler
* Feature selection using statistical tests (SelectKBest)
* Model training using Logistic Regression
* Detailed evaluation with multiple metrics

---

## 🧠 Tech Stack

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* Jupyter Notebook / Google Colab

---

## 📂 Project Structure

```
Heart-Disease-Detection/
│
├── Heart_disease_Detection.ipynb   # Main ML pipeline notebook
├── heart.csv                       # Dataset
├── README.md                       # Project documentation
└── requirements.txt               # Dependencies (optional)
```

---

## 📊 Dataset Description

The dataset consists of medical attributes used to predict heart disease:

* Age
* Sex
* Chest pain type (cp)
* Resting blood pressure (trestbps)
* Cholesterol (chol)
* Fasting blood sugar (fbs)
* Rest ECG (restecg)
* Maximum heart rate (thalach)
* Exercise-induced angina (exang)
* Oldpeak

### 🎯 Target Variable

* `0` → No heart disease
* `1` → Heart disease present

---

## ⚙️ ML Workflow

### 1. Data Cleaning

* Removed duplicate records

### 2. Outlier Handling

* Used percentile-based capping (1% & 99%)
* Preserved dataset size while reducing noise

### 3. Encoding

* Applied One-Hot Encoding for categorical features
* Avoided dummy variable trap (`drop_first=True`)

### 4. Feature Scaling

* Standardized numerical features using StandardScaler

### 5. Feature Selection

* Used SelectKBest with ANOVA (f_classif)
* Selected top 10 most important features

### 6. Model Training

* Logistic Regression classifier

### 7. Model Evaluation

* Accuracy Score
* Precision, Recall, F1-score
* Confusion Matrix

---

## 📈 Results

### 🔍 Model Performance

| Model                       | Accuracy |
| --------------------------- | -------- |
| Logistic Regression         | 83.61%   |
| Decision Tree Classifier    | 83.61%   |
| Support Vector Machine      | 80.33%   |
| Logistic Regression (Tuned) | 83.61%   |

---

### 📊 Best Model: Tuned Logistic Regression

* **Accuracy:** 83.61%
* **Cross-validation Accuracy:** 79.66%
* **Best Parameters:**

  * C = 1
  * Solver = liblinear

---

### 📄 Classification Report

```
              precision    recall  f1-score   support

           0       0.78      0.89      0.83        28
           1       0.90      0.79      0.84        33

    accuracy                           0.84        61
   macro avg       0.84      0.84      0.84        61
weighted avg       0.84      0.84      0.84        61
```

---

### 🧠 Key Insights

* Logistic Regression performed consistently well across metrics
* Feature selection helped improve accuracy and reduce overfitting
* Tuned model achieved stable performance with balanced precision and recall

---

## ▶️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/heart-disease-detection.git
cd heart-disease-detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run Jupyter Notebook:

```bash
jupyter notebook
```

4. Open:

```
Heart_disease_Detection.ipynb
```

---

## 🔮 Future Improvements

* Save trained model using joblib
* Build REST API using FastAPI
* Deploy model on cloud (Render / AWS / GCP)
* Create frontend using React for user interaction
* Hyperparameter tuning
* Try advanced models (XGBoost, Random Forest)

---

## 🌐 Deployment (Planned)

The model can be deployed using:

* FastAPI for backend
* Docker for containerization
* Cloud platforms like Render or AWS

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit a pull request.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

Akshith

---

## ⭐ Final Note

This project demonstrates a strong foundation in machine learning workflows and is designed to be extended into a full-stack AI application with deployment capabilities.
