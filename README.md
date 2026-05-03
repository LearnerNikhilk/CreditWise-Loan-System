# CreditWise Loan System 🏦📊

An end-to-end Machine Learning project designed to predict loan approval status based on applicant demographics, financial background, and credit history. This project utilizes Python and Scikit-Learn to build, evaluate, and optimize classification models.

## 📝 Project Overview
Financial institutions receive thousands of loan applications daily. This project automates the loan eligibility process based on customer details provided while filling out an online application form. By analyzing various factors, the model helps identify eligible customers faster and with higher accuracy.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (Logistic Regression, KNN, Naive Bayes)

## 🗂️ Dataset Attributes
The model is trained on a variety of applicant features, including but not limited to:
* `Applicant_Income` & `Coapplicant_Income`
* `Credit_Score` & `Existing_Loans`
* `DTI_Ratio` (Debt-to-Income Ratio)
* `Savings` & `Collateral_Value`
* `Employment_Status`, `Education_Level`, & `Property_Area`
* **Target Variable:** `Loan_Approved` (Yes/No)

## 🚀 Workflow & Methodology
1.  **Data Preprocessing:** * Handled missing values using `SimpleImputer` (Mean for numerical, Most Frequent for categorical).
    * Encoded categorical variables using `LabelEncoder` and `OneHotEncoder`.
    * Scaled numerical features using `StandardScaler`.
2.  **Exploratory Data Analysis (EDA):** * Visualized data distributions and correlations using heatmaps, boxplots, and histograms.
3.  **Feature Engineering:** * Created polynomial features (e.g., `DTI_Ratio_sq`, `Credit_Score_sq`) to capture non-linear relationships and boost model performance.
4.  **Model Training & Evaluation:** * Trained multiple classifiers: Logistic Regression, K-Nearest Neighbors, and Gaussian Naive Bayes.
    * Evaluated models based on Accuracy, Precision, Recall, and F1-Score.

## 📈 Key Results
After applying feature engineering and scaling, the **Logistic Regression** model outperformed the others, yielding the best balance of precision and recall:
* **Accuracy:** 87.5%
* **Precision:** 79.0%
* **Recall:** 80.3%
* **F1-Score:** ~0.80

## 💻 How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/LearnerNikhilk/CreditWise-Loan-System.git](https://github.com/LearnerNikhilk/CreditWise-Loan-System.git)
