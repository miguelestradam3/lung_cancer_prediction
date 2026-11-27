# Lung Cancer Level Prediction --- Logistic Regression

**Author:** Miguel Estrada

A machine-learning project that uses **Logistic Regression** to predict
the `Level` label (`Low`, `Medium`, or `High`) from the lung cancer
risk-factor dataset. The Jupyter Notebook demonstrates data preparation,
model evaluation, permutation importance, and interpretation of model
coefficients.

> **Important:** This is an educational machine-learning project. It is
> not a medical diagnostic tool and must not be used for clinical
> decisions.

## Project objectives

-   Inspect and validate the dataset.
-   Prepare predictors and the target variable.
-   Build a scikit-learn preprocessing and modeling pipeline.
-   Train a multiclass Logistic Regression model.
-   Evaluate predictions on a held-out test set.
-   Measure the importance of every original predictor with permutation
    importance.
-   Examine class-specific Logistic Regression coefficients.
-   Compare original categorical values with their encoded
    representations.

## Dataset

The notebook uses `cancer.csv` (the file can also be renamed to
`cancer.csv`).

-   **Rows:** 1,000
-   **Columns:** 26, including identifiers and the target
-   **Target:** `Level`
-   **Predictors:** 23 columns after excluding the two identifier
    columns

The `index` and `Patient Id` columns are excluded from training because
they identify records rather than represent risk factors.

### Target classes

  Original label     Encoded value
  ---------------- ---------------
  Low                            0
  Medium                         1
  High                           2

Class counts in the supplied dataset:

  Level      Records
  -------- ---------
  High           365
  Medium         332
  Low            303

## Features

The model uses the remaining columns as predictors, including:

`Age`, `Gender`, `Air Pollution`, `Alcohol use`, `Dust Allergy`,
`OccuPational Hazards`, `Genetic Risk`, `chronic Lung Disease`,
`Balanced Diet`, `Obesity`, `Smoking`, `Passive Smoker`, `Chest Pain`,
`Coughing of Blood`, `Fatigue`, `Weight Loss`, `Shortness of Breath`,
`Wheezing`, `Swallowing Difficulty`, `Clubbing of Finger Nails`,
`Frequent Cold`, `Dry Cough`, and `Snoring`.

The notebook treats the dataset's numeric values as provided. It does
not infer additional clinical meaning for coded values.

## Project workflow

1.  **Load and inspect data** --- review dimensions, columns, data
    types, missing values, and basic quality.
2.  **Prepare the target** --- map `Level` to numeric class labels.
3.  **Split the data** --- use an 80/20 train-test split with
    stratification and `random_state=42`.
4.  **Preprocess and train** --- use a scikit-learn pipeline to keep
    preprocessing within the training workflow and reduce data leakage.
5.  **Evaluate** --- review accuracy, balanced accuracy, classification
    metrics, and the confusion matrix.
6.  **Interpret the model** --- calculate permutation importance and
    inspect Logistic Regression coefficients.

## Model and preprocessing

The notebook uses:

-   Numeric missing-value imputation
-   `StandardScaler` for numeric features
-   Categorical imputation and One-Hot Encoding when categorical
    predictors are present
-   `LogisticRegression` with `max_iter=3000` and
    `class_weight="balanced"`

The train/test split is stratified to preserve the target-class
proportions in both sets.

## Feature interpretation

### Permutation importance

Permutation importance is calculated on the held-out test set using
**macro F1**. A predictor is shuffled and the change in the score is
measured.

The output includes every original predictor column. Larger positive
values indicate a greater decrease in macro F1 when that feature is
shuffled; values near zero indicate little measured impact; negative
values can occur due to sampling variability or model behavior.

Permutation importance describes predictive contribution for this model
and evaluation set. It does **not** establish causation.

### Logistic Regression coefficients

The notebook reports a coefficient for every transformed feature and
every target class.

-   Numeric predictors are standardized; their coefficients correspond
    to a one-standard-deviation change in the predictor.
-   Coefficient signs describe the direction of the class-specific model
    score, conditional on the other features.
-   If categorical features are present, One-Hot Encoding can create
    multiple transformed columns for one original predictor.

A separate summary maps the encoded columns back to the original
predictor columns. It includes every original predictor and summarizes
mean absolute coefficient magnitude across encoded levels and classes.
The detailed coefficient table should be used when signed,
class-specific values are needed.

**Permutation importance and coefficient magnitude are different
measures** and should not be interpreted as interchangeable.

## Encoding reference

The final notebook section documents:

-   The original `Level` categories and their numeric target codes.
-   Original categories and transformed columns for categorical
    predictors, when applicable.
-   Existing numeric codes in discrete columns (for example, `Gender`
    values) without assigning unsupported meanings to them.

## Getting started

### Requirements

-   Python 3
-   Jupyter Notebook or JupyterLab
-   pandas
-   NumPy
-   matplotlib
-   scikit-learn

Install the dependencies:

``` bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

### Run the notebook

1.  Download or clone this project.

2.  Place `cancer.csv` in the same directory as the notebook.
    Alternatively, rename the dataset to `cancer.csv`.

3.  Open the notebook:

    ``` bash
    jupyter notebook
    ```

4.  Open
    `main.ipynb`.

5.  Run the cells from top to bottom.

## Repository structure

``` text
.
├── cancer.csv
├── main.ipynb
└── README.md
```

## Results and limitations

The notebook contains the model's evaluation results, classification
report, confusion matrix, permutation-importance tables and plots, and
coefficient tables.

If the model produces unusually high or perfect scores, investigate the
dataset carefully. Such results may reflect simple or synthetic
relationships, target-related information in the predictors, or other
dataset-specific characteristics. A high score on this dataset does not
guarantee generalization to real-world patient data.

This project is intended for learning and portfolio demonstration. The
dataset and model are not validated for medical use, and predictions
should not be used to diagnose lung cancer or guide treatment.
