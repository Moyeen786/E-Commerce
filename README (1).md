# E-Commerce Machine Learning: Classification & Regression

A supervised machine learning project that explores e-commerce data through two predictive tasks: purchase-intent classification and customer-spending regression. The notebooks cover data inspection, preprocessing, feature engineering, model training, evaluation, and visual analysis.

## Project Overview

| Notebook | ML Task | Target | Goal |
|---|---|---|---|
| `23CSE301_Classification(1).ipynb` | Binary Classification | `Revenue` | Predict whether an online shopping session results in a purchase |
| `23CSE301_Regression(1).ipynb` | Regression | `Monetary` | Estimate customer spending from behavioral features |

## Repository Structure

```text
.
├── 23CSE301_Classification(1).ipynb
├── 23CSE301_Regression(1).ipynb
├── online_shoppers_intention.csv
├── ecommerce_user_segmentation.csv
└── README.md
```

The CSV files are required inputs referenced by the notebooks. Place them in the repository root or update the notebook file paths if stored elsewhere.

## Classification: Purchase Prediction

### Objective

Build and compare supervised classification models to predict the `Revenue` outcome for an e-commerce browsing session.

### Workflow

- Inspect dataset shape, data types, summary statistics, missing values, and duplicates.
- Explore distributions and feature relationships through visualizations.
- Encode categorical variables.
- Create `Total_Duration` and `Total_Pages` features.
- Split data into training and testing sets using a stratified 80/20 split.
- Standardize features for model training.
- Train and compare classification algorithms.

### Algorithms

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Decision Tree
- Support Vector Machine (RBF kernel)

### Evaluation

- Accuracy
- Weighted F1-score
- Purchase-class F1-score
- Classification report
- Confusion matrix

Because purchase outcomes are imbalanced, class-specific metrics are useful alongside overall accuracy.

## Regression: Customer Spending Prediction

### Objective

Train regression models to estimate the `Monetary` value using customer activity and engagement features.

### Workflow

- Inspect the dataset and examine feature distributions.
- Check and handle duplicate records.
- Detect and handle outliers using the IQR approach.
- Create an `Engagement_Score` using `Session_Count`, `Pages_Viewed`, `Clicks`, and `Wishlist_Adds`.
- Exclude identifier or segment-label columns when present.
- Split the data into training and testing sets.
- Scale features for algorithms that are sensitive to feature magnitude.
- Train, tune, and compare regression models.

### Algorithms

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

### Evaluation

- **R²:** measures the proportion of target variance explained by the model.
- **RMSE:** measures prediction error while giving larger errors more weight.
- **MAE:** measures average absolute prediction error.

The notebook also includes GridSearchCV-based hyperparameter tuning, 5-fold cross-validation for selected models, actual-versus-predicted visualizations, residual analysis, and Random Forest feature-importance analysis.

## Technology Stack

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Setup and Execution

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

Ensure the required CSV files are available at the paths used in the notebooks:

- `online_shoppers_intention.csv`
- `ecommerce_user_segmentation.csv`

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

Open either notebook and execute the cells in order.

## Reproducibility and Interpretation

- The notebooks use fixed random seeds for the documented train-test splits and applicable model configurations.
- Classification uses stratified splitting to maintain class proportions.
- Model metrics depend on the dataset and execution environment; run the notebooks to obtain the current results.
- For a production-grade evaluation, fit preprocessing transformations only on training data, ideally within a scikit-learn `Pipeline`, and reserve the test set for final evaluation.

## Scope

This repository is an educational machine learning implementation demonstrating classification and regression workflows for e-commerce analytics. Predictions are experimental and should not be interpreted as guaranteed customer behavior or business outcomes.
