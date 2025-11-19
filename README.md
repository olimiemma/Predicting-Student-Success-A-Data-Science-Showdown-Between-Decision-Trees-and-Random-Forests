
# Predicting Student Success: A Comparison of Decision Trees and Random Forests

## Project Overview

This repository contains a comprehensive data science project that builds, evaluates, and compares machine learning models to predict high school academic outcomes. Using a dataset of over 73,000 observations from New York State school districts, this project aims to classify student subgroups into three performance tiers ("low," "medium," or "high") based on their Regents diploma attainment rate.

The core of this analysis is a head-to-head comparison between two powerful classification algorithms: the **Decision Tree** and the **Random Forest**. The project follows a complete data science workflow, from initial data exploration to final model selection and evaluation, making it a thorough case study in applied machine learning.

The final selected model, a **Random Forest Classifier**, achieved a **weighted F1-Score of 0.8499** on the held-out test set, demonstrating high predictive power and excellent generalization.

---

## Table of Contents

1.  [Project Goal](#project-goal)
2.  [Dataset](#dataset)
3.  [Methodology](#methodology)
4.  [Key Findings](#key-findings)
5.  [Final Model Performance](#final-model-performance)
6.  [Repository Structure](#repository-structure)
7.  [How to Run This Project](#how-to-run-this-project)
8.  [Libraries and Tools Used](#libraries-and-tools-used)

---

## Project Goal

The primary objective of this project, as part of the DAV 6150 (Module 11) assignment, is to answer a fundamental question in model selection: **Under what circumstances does the added complexity of an ensemble model (Random Forest) provide a significant performance advantage over a simpler, more interpretable model (Decision Tree)?**

To achieve this, the project systematically:
-   Performs a deep Exploratory Data Analysis (EDA) to understand data characteristics and relationships.
-   Engineers a categorical target variable (`reg_pct_level`) from a continuous source.
-   Implements and compares two distinct feature selection strategies (filter-based vs. embedded).
-   Builds, trains, and rigorously evaluates four candidate models (two Decision Trees and two Random Forests).
-   Selects a champion model based on a multi-criteria evaluation and validates its performance on a held-out test set.

---

## Dataset

The dataset used is `M11_Data.csv`, a subset of the New York State high school graduation data for the 2018-2019 school year.

-   **Observations:** 73,152 rows
-   **Features:** 19 initial attributes
-   **Description:** Each row represents a specific student subgroup (e.g., by race, gender, economic status) within a particular NYS school district. The attributes include enrollment counts, graduation counts, dropout rates, and district-level information.

A key challenge identified in the data was a **severe class imbalance** in the engineered target variable, with over 81% of valid observations belonging to the "medium" performance class.

---

## Methodology

The project follows a structured, end-to-end data science workflow:

1.  **Data Cleaning & EDA:**
    -   Initial data loading, cleaning of non-standard values (`-`, `%`), and data type correction.
    -   **Univariate Analysis:** Investigated the distribution of each variable to identify zero-variance predictors and key characteristics (e.g., the median of `reg_pct`).
    -   **Bivariate Analysis:** Explored relationships between variables, confirming the high predictive potential of features like `nrc_desc` (Needs/Resource Capacity) and `subgroup_name`.
    -   **Multivariate Analysis:** Used a correlation heatmap and pairplot to confirm multicollinearity among raw count variables and validate feature selection strategies.

2.  **Feature Engineering & Preparation:**
    -   Created the categorical target variable `reg_pct_level` ('low', 'medium', 'high') based on the median of the `reg_pct` column.
    -   Removed source columns (`reg_pct`, `reg_cnt`) to prevent data leakage.
    -   Dropped all rows with missing target values, resulting in a clean modeling dataset of 39,674 observations.

3.  **Feature Selection:**
    -   **Approach 1 (Filter Method):** Used **Mutual Information** to rank features based on their statistical dependency with the target, creating **Feature Set 1**.
    -   **Approach 2 (Embedded Method):** Used **Random Forest Feature Importance** to rank features based on their contribution during model training, creating **Feature Set 2**.

4.  **Model Training & Evaluation:**
    -   Split the data into a 70% training set and a 30% testing set using **stratification** to preserve the class imbalance.
    -   Trained four models with different feature sets and hyperparameters, using `class_weight='balanced'` to handle the imbalance.
    -   Performed **5-Fold Stratified Cross-Validation** to obtain robust performance estimates. The primary evaluation metric was the **weighted F1-Score**.

5.  **Model Selection:**
    -   Selected the champion model based on a comprehensive comparison of cross-validation scores, overfitting analysis, and model consistency.
    -   **Random Forest Model 2** was the clear winner, excelling in both performance and stability.

6.  **Final Testing:**
    -   Evaluated the champion model on the held-out test set to get a final, unbiased assessment of its generalization performance.
    -   Analyzed the **Classification Report** and **Confusion Matrix** to understand the model's behavior, particularly its high-recall, lower-precision performance on the minority classes.

---

## Key Findings

-   **Random Forest is Superior:** The Random Forest models significantly outperformed the Decision Tree models across all metrics, demonstrating the power of ensembling for this complex dataset.
-   **Performance vs. Overfitting:** The best-performing model (`Random Forest 2`) also had the largest "overfitting gap" (Train vs. CV accuracy). However, its superior performance and consistency made this a worthwhile trade-off.
-   **Key Predictive Features:** `grad_pct` (graduation percentage) was consistently the most important predictor. Other vital features included `dropout_pct`, `enroll_cnt` (subgroup size), `nrc_desc` (district type), and `subgroup_name`.

---

## Final Model Performance

The selected champion model, **Random Forest 2**, achieved the following performance on the unseen test set:

| Metric               | Score    |
| -------------------- | -------- |
| **F1-Score (Weighted)** | **0.8499** |
| Accuracy             | 0.8366   |
| Precision (Weighted) | 0.8811   |
| Recall (Weighted)    | 0.8366   |

The model demonstrated excellent generalization, as its test performance was consistent with the estimates from cross-validation.

---

## Repository Structure

.
├── M11_Assignment_Notebook.ipynb # The complete Jupyter Notebook with all code, analysis, and visualizations.
├── M11_Data.csv # The raw dataset used for the project.
└── README.md # This file.

code
Code
download
content_copy
expand_less
---

## How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone [your-repository-url]
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd [repository-name]
    ```
3.  **Install the required libraries:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
4.  **Run the Jupyter Notebook:**
    Open `M11_Assignment_Notebook.ipynb` in Jupyter Lab or Jupyter Notebook and run the cells sequentially. The notebook is self-contained and loads the data directly from its raw file.

---

## Libraries and Tools Used

-   **Data Manipulation:** Pandas, NumPy
-   **Data Visualization:** Matplotlib, Seaborn
-   **Machine Learning:** Scikit-learn
    -   `DecisionTreeClassifier`, `RandomForestClassifier`
    -   `train_test_split`, `cross_validate`, `StratifiedKFold`
    -   A full suite of metrics from `sklearn.metrics`
