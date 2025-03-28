# Logistic Regression for Diabetes Prediction

## 📌 Project Overview
This project implements a **Logistic Regression model** to predict whether a patient has diabetes based on various health indicators. The dataset used is the **Pima Indians Diabetes Dataset**, which includes features such as glucose level, blood pressure, and BMI.

## 📂 Dataset Information
The dataset consists of 768 samples and 9 features:
- **Pregnancies** → Number of times pregnant
- **Glucose** → Blood sugar level
- **BloodPressure** → Blood pressure
- **SkinThickness** → Skin fold thickness
- **Insulin** → Insulin level
- **BMI** → Body Mass Index
- **DiabetesPedigreeFunction** → Family diabetes history
- **Age** → Age in years
- **Outcome** → Target variable (1 = Diabetes, 0 = No Diabetes)

## 🚀 Model Implementation Steps
1. **Data Loading**: Load the dataset using Pandas.
2. **Data Preprocessing**: Standardize features for better model performance.
3. **Train-Test Split**: Split the dataset into 80% training and 20% testing.
4. **Model Training**: Train a Logistic Regression model using `scikit-learn`.
5. **Model Evaluation**: Calculate Accuracy, Confusion Matrix, and Classification Report.
6. **Feature Importance Analysis**: Identify key factors affecting diabetes prediction.
7. **Hyperparameter Tuning**: Optimize the model using `GridSearchCV`.
8. **Comparison with Other Models**: Train and evaluate **Random Forest** and **SVM** models for comparison.

## 🔬 Results & Model Comparison
| Model                | Accuracy |
|----------------------|----------|
| Logistic Regression | 75%      |
| Random Forest       | 73%      |
| SVM                 | 76%      |

## 📌 Key Findings
- **SVM achieved the highest accuracy (76%)**, slightly outperforming Logistic Regression (75%).
- **Feature importance analysis** revealed that **Glucose, BMI, and Age** are the most influential features.
- **Hyperparameter tuning** can further improve the model's performance.

## ⚡ Next Steps
- Fine-tune **SVM** and **Random Forest** hyperparameters for better accuracy.
- Perform **feature engineering** to create new meaningful features.
- Address class imbalance using **SMOTE** or other resampling techniques.

## 🛠 Requirements
- Python 3.x
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`

## ▶ How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/logistic-regression-diabetes.git
   cd logistic-regression-diabetes
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python logistic_regression.py
   ```

## 📜 License
This project is open-source and free to use under the **MIT License**.

---

### 📬 Contact
For questions or collaborations, feel free to reach out!

💡 **Happy Coding!** 🚀

