# Employee Salary Prediction

A Machine Learning project that predicts whether an individual's annual income is **<=50K or >50K** based on demographic, educational, and employment-related attributes.

The project includes data preprocessing, exploratory data analysis, model comparison, and an interactive Streamlit application for individual and batch predictions.

## 🎯 Problem Statement

The objective is to build a binary classification model that predicts an individual's income category using features such as:

- Age
- Workclass
- Education
- Marital Status
- Occupation
- Relationship
- Capital Gain/Loss
- Hours per Week
- Native Country
- Other demographic attributes

Target classes:

- `<=50K`
- `>50K`

## 📊 Dataset

The project uses the **Adult Census Income dataset**.

- Records: ~48,842
- Input features: 14
- Target: `income`

### Preprocessing

- Replaced `?` values with missing values
- Median imputation for numerical features
- Most-frequent imputation for categorical features
- StandardScaler for numerical features
- OneHotEncoder for categorical features
- 80/20 stratified train-test split

A Scikit-learn `ColumnTransformer` and `Pipeline` were used to keep preprocessing and model prediction consistent.

## 🤖 Model Comparison

Multiple classification algorithms were evaluated using Accuracy, Precision, Recall, and F1-score.

| Model | Accuracy | F1-Score (>50K) |
|---|---:|---:|
| Logistic Regression | 85.24% | 65.62% |
| Random Forest | 85.95% | 68.42% |
| KNN | 83.37% | 63.56% |
| SVM | 85.96% | 66.81% |
| **Gradient Boosting** | **86.76%** | **68.44%** |

### Final Model

**Gradient Boosting Classifier** was selected as the final model based on its overall performance.

- Accuracy: **86.76%**
- Precision (>50K): **79.70%**
- Recall (>50K): **59.97%**
- F1-score (>50K): **68.44%**

The complete preprocessing and model pipeline is saved using Joblib:

`model/salary_prediction_model.pkl`

## 🌐 Streamlit Application

The project includes an interactive Streamlit application with three main modules.

### 🔮 1. Individual Prediction

Users can enter all required employee attributes and receive:

- Predicted income category
- Probability of `<=50K`
- Probability of `>50K`

### 📊 2. Analytics Dashboard

The dashboard provides visual insights into the dataset, including:

- Income distribution
- Income by education
- Age vs income
- Average working hours by income
- Occupation vs income

### 📁 3. Batch Prediction

Users can upload a CSV file containing employee records.

The application:

- Validates required columns
- Predicts income for multiple records
- Calculates probability of `>50K`
- Displays the results
- Allows downloading the predictions as a CSV file

## 📁 Project Structure

    Employee-Salary-Prediction/
    │
    ├── data/
    │   ├── adult.csv
    │   └── test_batch.csv
    │
    ├── model/
    │   └── salary_prediction_model.pkl
    │
    ├── notebooks/
    │   └── Employee_Salary_Prediction.ipynb
    │
    ├── app.py
    ├── requirements.txt
    ├── README.md
    └── .gitignore

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Jupyter Notebook

## 🚀 How to Run

### 1. Clone the repository

    git clone <your-github-repository-url>

    cd Employee-Salary-Prediction

### 2. Install dependencies

    pip install -r requirements.txt

### 3. Run the Streamlit application

    streamlit run app.py

The application will open in your browser.

## 🔄 Project Workflow

    Dataset
       ↓
    Data Cleaning
       ↓
    Exploratory Data Analysis
       ↓
    Feature Preprocessing
       ↓
    Train-Test Split
       ↓
    Model Comparison
       ↓
    Gradient Boosting Model
       ↓
    Model Saving
       ↓
    Streamlit Deployment
       ↓
    Individual / Analytics / Batch Prediction

## 💡 Key Learning Outcomes

- Data cleaning and preprocessing
- Exploratory Data Analysis
- Classification algorithms
- Model evaluation and comparison
- Handling numerical and categorical features
- Scikit-learn Pipelines
- Model serialization using Joblib
- Data visualization
- Streamlit application development
- Batch prediction

## ⚠️ Limitations

- The model predicts income category, not exact salary.
- Performance depends on the training dataset and its distribution.
- The `>50K` class has lower recall than the `<=50K` class.
- The model should not be used as the sole basis for high-impact employment or compensation decisions.

## 🔮 Future Improvements

- Hyperparameter tuning
- Cross-validation
- Class imbalance handling
- Explainable AI using SHAP/LIME
- Model monitoring
- Cloud deployment
- Additional datasets for validation

## 👩‍💻 Author

**Deepika Katika**

B.Tech – Artificial Intelligence & Machine Learning