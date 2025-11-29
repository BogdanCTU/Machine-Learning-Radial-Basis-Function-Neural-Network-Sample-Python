# Machine-Learning-Radial-Basis-Function-Neural-Network-Sample-Python

# Food Nutrient Classifier

## Overview

This project utilizes a **Radial Basis Function (RBF) Neural Network** to classify food items as **Healthy (`True`)** or **Unhealthy (`False`)** based on their macronutrient profile.

The model is built using the [evorbf](https://github.com/thieu1995/EvoRBF) library from  and `scikit-learn`.
It is designed to handle non-linear decision boundaries, allowing it to distinguish complex nutritional cases (e.g., distinguishing high-carb healthy grains from high-carb sugary snacks).

Simulates a real-world scenario by training on historical data (first **16,000** records, reppresenting the 80% of dataset records) and testing on new/unseen data (remaining \**~4,000** records, the remaining 20%), of which only 3608 (18%) records are considered valid and tested.

The dataset is reppresented by USDA Branded Food Nutrient Dataset pubicly available at [USDA](https://fdc.nal.usda.gov/).

## Dependencies

To run this script, you need the following Python libraries:

```bash
pip install pandas scikit-learn evorbf joblib
```

## Dataset Structure

The script expects a CSV file named `INPUT.csv` with a semi-colon delimiter (`;`). The file must contain the following columns:

| Column Name | Description |
| :--- | :--- |
| `ENERGY` | Total energy (kcal) |
| `PROTEIN` | Protein content (g) |
| `CARBS` | Carbohydrate content (g) |
| `TOTAL_FAT` | Total fat content (g) |
| `SATURATED_FAT` | Saturated fat content (g) |
| `FIBER` | Dietary fiber (g) |
| `SUGARS` | Total sugars (g) |
| `CLASSIFICATION` | Target Label (`True` or `False`) |

## Methodology

### 1\. Data Split

Instead of a random shuffle, the data is split sequentially to test the model's ability to generalize to new entries added to the database.

  * **Training Set:** Rows 0 to 16,000 (80% of data records);
  * **Test Set:** Rows 16,001 to 20.000 (20% of data records).

### 2\. Preprocessing

  * **Label Encoding:** `False` $\rightarrow$ 0, `True` $\rightarrow$ 1;
  * **Scaling:** Standard Scikit-Learn `StandardScaler` is fitted **only** on the training data and applied to the test data to prevent data leakage.

### 3\. Model Configuration

The **EvoRBF** classifier is configured with:

  * **Hidden Neurons (`size_hidden=80`):** High count to capture specific nutritional "pockets" (e.g., High Carb + 0 Sat Fat);
  * **Sigma (`sigmas=0.5`):** Low spread to create sharper decision boundaries between similar food items.

## Performance Results

The model was evaluated on the final **3,608 records** of the dataset.

### Summary Metrics

| Metric | Value |
| :--- | :--- |
| **Accuracy** | **96.37%** |
| **Correct Predictions** | 3477 / 3608 |
| **Total Test Samples** | 3608 |

### Classification Report

```text
              precision    recall  f1-score   support

       False       0.97      0.99      0.98      3236
        True       0.87      0.76      0.81       372

    accuracy                           0.96      3608
   macro avg       0.92      0.88      0.90      3608
weighted avg       0.96      0.96      0.96      3608
```

### Analysis of Results

1.  **High Overall Accuracy (96%):** The model is extremely effective at identifying the majority class (Unhealthy foods).
2.  **Class Imbalance:** The test set contained 3,236 "False" items and only 372 "True" items.
3.  **Healthy Food Performance:**
      * **Precision (0.87):** When the model claims a food is Healthy, it is correct 87% of the time.
      * **Recall (0.76):** The model successfully found 76% of all Healthy foods. It is slightly conservative, meaning it occasionally mislabels a Healthy item as Unhealthy (likely complex edge cases), but it rarely mistakes junk food for healthy food.

## Usage

Run the script directly:

```bash
python classifier_script.py
```

  * The script will print the training progress.
  * It will output the accuracy on the last 4,000 rows.
  * It performs a sanity check on a hard-coded sample (High Carb/Low Fat) to ensure logical consistency.
