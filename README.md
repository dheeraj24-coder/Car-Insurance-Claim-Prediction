# 🚗 Car Insurance Claim Prediction
### 🤖 Machine Learning | Predictive Analytics | Classification

**Predicting whether a customer is likely to file a car insurance claim using Machine Learning.**
---

## 📌 Project Overview

Car insurance companies need to understand the likelihood of customers filing insurance claims to support risk assessment and business planning.

This project uses **Machine Learning classification algorithms** to predict whether a customer will file a car insurance claim based on customer, vehicle, and insurance-related information.

The target variable is `CLAIM_FLAG`:

- 🟢 `0` — No Claim
- 🔴 `1` — Claim

Three Machine Learning algorithms are trained and compared: **Logistic Regression, Decision Tree, and Random Forest**.

The models are evaluated using multiple performance metrics, including Accuracy, Precision, Recall, F1-Score, and ROC-AUC. The model with the highest ROC-AUC score is selected as the best-performing model in the comparison.

🎯 **Project Goal:** Build a complete Machine Learning pipeline that transforms raw insurance data into predictions, evaluates model performance, and saves the selected model for future use.

## 🎯 Problem Statement

The objective is to develop a binary classification model that can distinguish between customers who file insurance claims and those who do not.

The project addresses the following questions:

- Which Machine Learning algorithm performs best on the available dataset?
- How effectively can the models identify customers who file claims?
- How do Accuracy, Precision, Recall, F1-Score, and ROC-AUC compare across models?
- Which features contribute most to the Random Forest model's predictions?
- How can the trained model be saved and reused for future predictions?

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| 🐍 Python | Programming and model development |
| 🐼 Pandas | Data loading, cleaning, and manipulation |
| 🔢 NumPy | Numerical computations |
| 📊 Matplotlib | Data visualization |
| 🎨 Seaborn | Statistical charts and heatmaps |
| 🤖 Scikit-learn | Preprocessing, model training, and evaluation |
| 📦 Joblib | Saving and loading the trained model |
| 📓 Jupyter Notebook | Interactive development and analysis |

## 🔍 Project Workflow

The project follows a complete Machine Learning workflow:

**Data Loading → Data Cleaning → Exploratory Data Analysis → Feature Engineering → Data Splitting → Preprocessing → Model Training → Evaluation → Model Selection → Prediction → Model Saving**

### 1️⃣ Data Loading and Understanding

The dataset is loaded from a compressed CSV file using Pandas.

Initial analysis includes:

- Checking the number of rows and columns.
- Examining column names and data types.
- Reviewing sample records.
- Understanding numerical features using descriptive statistics.

This step helps identify the dataset's structure before preparing it for Machine Learning.

### 2️⃣ Data Cleaning and Preparation

The dataset is checked for duplicate records, and exact duplicate rows are removed.

Currency-related columns are converted into numerical values by removing dollar signs and commas.

Missing values are also examined to understand the quality of the dataset.

**Important data leakage prevention:** The `CLM_AMT` column is excluded from the model inputs because claim amount may only be known after a claim occurs. Using this information could allow the model to learn from information unavailable at prediction time.

The `ID` column is excluded as a predictive feature, while `BIRTH` is also removed from the model inputs.

### 3️⃣ Exploratory Data Analysis (EDA)

Exploratory Data Analysis is used to understand patterns and relationships in the data.

The project includes:

- 📊 **Claim Distribution:** Compares customers with and without claims.
- 📈 **Feature Distributions:** Examines variables such as `AGE`, `INCOME`, `CAR_AGE`, and `CLM_FREQ`.
- 🚘 **Claim Rate by Car Use:** Compares claim rates for private and commercial vehicle use.
- 🔥 **Correlation Heatmap:** Visualizes relationships between numerical variables.

EDA helps identify patterns and understand the dataset before model training. These visualizations show associations, not necessarily causal relationships.

### 4️⃣ Feature Selection and Data Splitting

The target variable is:

`y = CLAIM_FLAG`

The input features are:

`X = All eligible columns after removing ID, BIRTH, CLM_AMT, and CLAIM_FLAG`

The notebook uses **StratifiedGroupKFold** with five splits, shuffling, and a fixed random seed.

The first generated split is used to create the training and testing datasets.

This approach aims to preserve the target-class distribution while keeping records belonging to the same ID group from appearing in both training and testing data.

### 5️⃣ Data Preprocessing

The project uses Scikit-learn pipelines to process numerical and categorical features separately.

**🔢 Numerical Features**

- Missing values are filled using the median.
- StandardScaler standardizes the numerical features.

**🔤 Categorical Features**

- Missing values are filled using the most frequent category.
- OneHotEncoder converts categorical values into numerical representations.
- `handle_unknown='ignore'` allows the pipeline to handle previously unseen categories during transformation.

A ColumnTransformer combines both preprocessing pipelines.

Using a pipeline ensures that preprocessing is fitted on training data and then applied consistently to test data.

## 🤖 Machine Learning Algorithms

Three classification algorithms are trained and compared.

### 1. Logistic Regression 📈

Logistic Regression is a classification algorithm that estimates the probability of a customer filing an insurance claim.

It uses the logistic function to produce a probability between 0 and 1, which is converted into a class prediction.

**Why use it?**
- Provides a useful baseline classification model.
- Produces class probabilities.
- Offers a relatively simple model for comparison.

**Configuration:** `max_iter=2000`

### 2. Decision Tree 🌳

A Decision Tree predicts outcomes by applying a sequence of feature-based decision rules.

It divides the dataset into smaller groups until it reaches a prediction.

**Why use it?**
- Captures nonlinear relationships.
- Models interactions between features.
- Provides a visual representation of decision rules.

**Configuration:**
- `max_depth=8`
- `min_samples_split=20`
- `min_samples_leaf=10`
- `random_state=42`

### 3. Random Forest 🌲

Random Forest combines predictions from multiple decision trees to produce a final classification.

It uses an ensemble of trees, which can improve generalization compared with an individual decision tree.

**Why use it?**
- Captures complex patterns.
- Handles nonlinear relationships.
- Provides feature importance scores.
- Offers a strong comparison model for structured datasets.

**Configuration:**
- `n_estimators=200`
- `max_depth=12`
- `min_samples_split=10`
- `min_samples_leaf=5`
- `random_state=42`

- ## 📊 Model Performance Comparison

In this project, three Machine Learning classification algorithms were trained and evaluated on the test dataset:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The models were evaluated using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

### 1. Performance of All Three Algorithms

The following results are taken from the project's Jupyter Notebook:

| Evaluation Metric | Logistic Regression | Decision Tree | Random Forest |
|---|---:|---:|---:|
| **Accuracy** | **79.48%** | 76.47% | 78.85% |
| Precision | 68.10% | 57.73% | **70.55%** |
| Recall | 43.17% | **43.53%** | 35.34% |
| F1-Score | **52.84%** | 49.64% | 47.09% |
| ROC-AUC | **81.69%** | 76.17% | 81.59% |

### 2. Classification Report

The classification reports below show how well each algorithm predicts the two classes: `No Claim` and `Claim`.

#### Logistic Regression

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| No Claim | 82% | 93% | 87% | 1,512 |
| Claim | 68% | 43% | 53% | 549 |
| **Overall Accuracy** | | | **79%** | **2,061** |

**Explanation:** Logistic Regression achieved the highest accuracy and ROC-AUC among the three models. It performed well in identifying customers who did not make a claim.

#### Decision Tree

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| No Claim | 81% | 88% | 85% | 1,512 |
| Claim | 58% | 44% | 50% | 549 |
| **Overall Accuracy** | | | **76%** | **2,061** |

**Explanation:** Decision Tree achieved an accuracy of approximately 76%. It offers a tree-based approach that can help explain how the model makes predictions, but it had the lowest ROC-AUC among the three models.

#### Random Forest

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| No Claim | 80% | 95% | 87% | 1,512 |
| Claim | 71% | 35% | 47% | 549 |
| **Overall Accuracy** | | | **79%** | **2,061** |

**Explanation:** Random Forest achieved an accuracy of approximately 79% and the highest precision for the `Claim` class. However, its recall for claims was lower, meaning it missed a larger proportion of actual claim cases.

### 3. Which Algorithm Performed Best?

Based on the actual evaluation results:

- **Logistic Regression:** Best overall result, with 79.48% accuracy and 81.69% ROC-AUC.
- **Decision Tree:** Achieved 76.47% accuracy and 76.17% ROC-AUC.
- **Random Forest:** Achieved 78.85% accuracy and 81.59% ROC-AUC.

**Final Selection: Logistic Regression**

The notebook selects Logistic Regression because it achieved the highest ROC-AUC score of **0.8169 (81.69%)**.

ROC-AUC measures how well a model distinguishes between the two classes across different classification thresholds. Logistic Regression performed slightly better than Random Forest on this metric.

### 4. Important Observation

The dataset contains more `No Claim` records than `Claim` records:

- No Claim: 1,512 test samples
- Claim: 549 test samples

Although all three models achieved reasonably good overall accuracy, their recall scores for the `Claim` class were relatively low.

This means the models missed some customers who actually made insurance claims. In a real insurance business, missing genuine claims could be an important concern.

Therefore, model selection should consider not only accuracy but also ROC-AUC, recall, precision, and the business cost of incorrect predictions.

### 🏆 Model Comparison

All three models are trained using the same training and testing split and evaluated using the same metrics.

The results are sorted by ROC-AUC in descending order.

The model with the highest ROC-AUC is selected for the final prediction demonstration.

## 📊 Model Evaluation Metrics

The project uses five important evaluation metrics.

| Metric | What It Measures |
|---|---|
| 🎯 Accuracy | Percentage of all predictions that are correct |
| 🔍 Precision | Among predicted claims, the proportion that are actual claims |
| 🚨 Recall | Among actual claims, the proportion correctly identified |
| ⚖️ F1-Score | Harmonic mean of Precision and Recall |
| 📈 ROC-AUC | How well the model ranks claim cases above no-claim cases across classification thresholds |

### Why Are Multiple Metrics Important?

Accuracy alone may not tell the complete story, particularly when claim and no-claim records are imbalanced.

For an insurance claim prediction model, Recall helps measure how many actual claim cases are identified, while Precision helps measure how reliable positive claim predictions are.

F1-Score balances Precision and Recall.

ROC-AUC evaluates how well the model separates the two classes across different thresholds rather than relying on a single threshold.

The notebook also generates:

- 📋 Classification reports for each model.
- 🔲 Confusion matrices to compare actual and predicted classes.
- 📉 ROC curves to visualize model discrimination.
- 📊 A bar chart comparing Accuracy, F1-Score, and ROC-AUC.

These results provide a broader view of model performance than Accuracy alone.

## 🔬 Model Interpretation

The project includes two visual interpretation techniques.

**🌳 Decision Tree Visualization**

The top levels of the trained Decision Tree are displayed to illustrate how the model makes decisions using feature-based splits.

**📊 Random Forest Feature Importance**

The project calculates and visualizes the top 15 features according to the trained Random Forest's feature importance scores.

This helps identify which features contribute most to the forest's predictive decisions.

Feature importance does not establish causation and should not be interpreted as proof that a particular variable causes an insurance claim.

## 🔮 Prediction on a New Customer

After comparing all three models, the notebook selects the model with the highest ROC-AUC score.

It then predicts the claim status for one customer from the test dataset.

The prediction includes:

- **Predicted Class:** Claim or No Claim.
- **Estimated Claim Probability:** The model's predicted probability of the positive class.
- **Customer Features:** The input record used for prediction.

The prediction is then repeated after loading the saved model to verify that the saved pipeline can be reused.

The probability is a model estimate, not a guarantee that a claim will occur.

## 💾 Model Saving and Reusability

The selected model is saved using Joblib:

`car_insurance_claim_best_model.pkl`

The saved package contains:

- The selected preprocessing and classification pipeline.
- The selected model's name.
- The list of input feature columns.

Saving the entire pipeline preserves the preprocessing steps alongside the trained estimator, allowing the model to be loaded and used again without manually rebuilding the pipeline.

## 💡 Key Learning Outcomes

Through this project, I explored:

- End-to-end Machine Learning project development.
- Data cleaning and exploratory data analysis.
- Data leakage prevention.
- Numerical and categorical feature preprocessing.
- Stratified group-based train-test splitting.
- Binary classification using three algorithms.
- Model comparison and performance evaluation.
- Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- Confusion matrices and ROC curve visualization.
- Decision Tree interpretation and Random Forest feature importance.
- Model serialization and loading with Joblib.

## ⚠️ Limitations

- Model performance depends on the quality and representativeness of the dataset.
- The selected model is based on ROC-AUC, which may not be the best selection criterion for every insurance business objective.
- The notebook uses the test dataset as validation data during model training, so a separate validation set and untouched final test set would provide a more reliable evaluation.
- Predicted probabilities may require calibration before they can be interpreted as reliable real-world claim probabilities.
- The project is an educational Machine Learning implementation and should not be used as the sole basis for real insurance underwriting or pricing decisions.

## 👨‍💻 Connect With Me

**Dheeraj Chouhan**  
🎓 B.Tech Student | Aspiring AI/ML Engineer

- 🐙 **GitHub:** [dheeraj24-coder](https://github.com/dheeraj24-coder)
- 💼 **LinkedIn:** https://www.linkedin.com/in/dheeraj-chouhan-7a6996371/
- 📧 **Email:** dheerajchouhan4466@gmail.com

---

### 🚀 Turning Data into Intelligent Predictions

**Built with Python, Machine Learning, and Data Analytics. ❤️**
