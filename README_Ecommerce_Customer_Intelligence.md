# E-Commerce Customer Intelligence: Purchase Classification & Spending Prediction

A machine learning project exploring e-commerce customer behavior through two complementary predictive tasks:

- **Purchase Classification:** predict whether an online shopping session results in a purchase.
- **Spending Regression:** estimate a customer's total spending from shopping behavior.

The project compares multiple supervised learning algorithms, evaluates their results with task-appropriate metrics, and uses exploratory analysis and visualizations to interpret customer activity.

## Project Overview

| Notebook | Problem Type | Target | Purpose |
|---|---|---|---|
| `23CSE301_Classification(1).ipynb` | Binary classification | `Revenue` | Predict purchase vs. no purchase for an online session |
| `23CSE301_Regression(1).ipynb` | Regression | `Monetary` | Predict total customer spending |

## Repository Structure

```text
.
├── 23CSE301_Classification(1).ipynb
├── 23CSE301_Regression(1).ipynb
├── online_shoppers_intention.csv       # Required by classification notebook
├── ecommerce_user_segmentation.csv     # Required by regression notebook
└── README.md
```

Place each dataset in the repository root, or update the CSV paths in the notebooks to match your local data location.

## Machine Learning Workflow

1. Load the dataset and inspect its dimensions, columns, data types, missing values, duplicates, and descriptive statistics.
2. Perform exploratory data analysis (EDA) using distributions, count plots, correlation heatmaps, and feature relationship plots.
3. Clean and prepare the data, including duplicate handling, missing-value treatment, outlier handling, and categorical encoding where applicable.
4. Engineer behavior-related features.
5. Split data into training and test sets.
6. Scale features for model training.
7. Train and compare candidate models.
8. Evaluate performance and visualize model behavior.

## 1. Purchase Classification

**Objective:** Estimate whether a visitor completes a purchase during an online shopping session.

**Target:** `Revenue`  
**Classes:** Purchase and No Purchase

### Models

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Decision Tree
- Support Vector Machine (RBF kernel)

### Preparation and Evaluation

The notebook includes categorical feature encoding, engineered `Total_Duration` and `Total_Pages` features, a stratified 80/20 train-test split, and feature standardization. It evaluates models using:

- Accuracy
- Weighted F1-score
- Purchase-class F1-score
- Classification reports
- Confusion matrices

The target classes are imbalanced, so purchase-class performance is considered alongside overall accuracy.

## 2. Customer Spending Regression

**Objective:** Predict customer total spending from e-commerce behavior.

**Target:** `Monetary`

### Models

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Polynomial Regression (degree 2)
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor (RBF kernel)
- K-Nearest Neighbors Regressor

### Feature Engineering

The notebook creates an `Engagement_Score` by combining:

- `Session_Count`
- `Pages_Viewed`
- `Clicks`
- `Wishlist_Adds`

It also checks duplicates, handles outliers using the IQR approach, and excludes `Customer_ID` and `Segment_Label` when those columns are present.

### Evaluation and Model Selection

Regression models are compared using:

- **R²:** proportion of target variance explained by the model
- **RMSE:** error measure that penalizes larger prediction errors more strongly
- **MAE:** average absolute prediction error

The notebook further includes hyperparameter tuning with `GridSearchCV`, 5-fold cross-validation for selected models, actual-versus-predicted plots, residual analysis, and Random Forest feature importance.

## Tech Stack

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Add the datasets

Add the two CSV files referenced in the notebooks:

- `online_shoppers_intention.csv`
- `ecommerce_user_segmentation.csv`

Use the dataset versions permitted for your coursework and retain any required attribution.

### 4. Launch Jupyter

```bash
jupyter notebook
```

Open either notebook and run the cells from top to bottom.

## Reproducibility Notes

- The notebooks use `random_state=42` for the documented train-test splits and applicable model settings.
- The classification notebook uses a stratified split to preserve class proportions.
- Reported model results depend on the dataset version and execution environment. No performance figures are stated here because they should be taken from a fresh run of the notebooks.
- For a rigorous production workflow, fit preprocessing steps only on training data (preferably through a scikit-learn `Pipeline`) and keep the test set untouched until final evaluation.

## Project Scope

This is an educational machine learning project focused on comparing classification and regression approaches for e-commerce analytics. Its outputs are experimental predictions and should not be treated as guaranteed customer behavior or business outcomes.

## Author

**Shaik Moyeen**  
CSE-D | Machine Learning Project

---

*If you use or adapt this work, please retain appropriate dataset attribution and follow the dataset's license or terms of use.*
