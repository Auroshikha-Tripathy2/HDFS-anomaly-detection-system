# HDFS Anomaly Detection in HDFS Logs

This project presents a complete machine learning pipeline for detecting anomalies in Hadoop Distributed File System (HDFS) logs with exceptionally high accuracy and precision. The model is designed to identify operational failures or potential security threats from log data, a critical task in managing large-scale data systems.

The final model achieves **99.98% accuracy** and a **0.9961 F1-Score** on a standard HDFS benchmark dataset, demonstrating its effectiveness for real-world applications where both high precision (minimizing false alarms) and high recall (finding all true threats) are crucial.

## Methodology

The success of this model is built on a robust, multi-stage methodology designed to handle the specific challenges of log data, such as class imbalance and complex data types. The entire workflow is implemented in the `2.ipynb` Jupyter Notebook.

### 1. Data Sourcing and Preparation
The model is trained and evaluated on the widely used HDFS log dataset from `honicky/hdfs-logs-encoded-blocks`, sourced from Hugging Face. The dataset is first cleaned of any missing values, and a standard stratified train-test split is performed to ensure the distribution of normal and anomalous samples is preserved in both sets.

### 2. Hybrid Feature Engineering
A key innovation of this project is the creation of a hybrid feature set that provides a rich, multi-faceted view of the data. This approach combines two techniques:

* **Numerical Feature Extraction:** We engineer features that capture the statistical properties of the `block_id` (e.g., `numeric_id`) and the sequence length of the log message template (`tokenized_block`). This provides the model with expert-driven signals about the data's structure.
* **Text Vectorization:** We treat the `tokenized_block` as a text document and use a Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer. This automatically identifies the most important words and sequences of words (n-grams) that are predictive of an anomaly, capturing the semantic context of the log message.

These two feature sets are then combined into a single feature matrix, giving the model both high-level statistical information and low-level textual patterns to learn from.

### 3. Handling Class Imbalance
Log data is naturally highly imbalanced, with anomalies being very rare. To address this, we employ a two-pronged strategy:
* **SMOTE (Synthetic Minority Over-sampling Technique):** We use SMOTE exclusively on the training data to create synthetic examples of the anomaly class. This gives the model a more balanced dataset to learn from without introducing data leakage into the test set.
* **Model-Level Weighting:** The XGBoost classifier is configured with the `scale_pos_weight` parameter, which internally gives more importance to the minority (anomaly) class during training.

### 4. Model Training and Hyperparameter Tuning
We use an **XGBoost Classifier**, a state-of-the-art gradient boosting algorithm known for its performance and efficiency. To extract the maximum performance from the model, we perform automated hyperparameter tuning using `RandomizedSearchCV`. This process systematically tests different combinations of model settings with cross-validation to find the optimal configuration for this specific dataset.

## Results

The final, tuned model demonstrates state-of-the-art performance on the hold-out test set. The key metrics confirm its effectiveness:

| Metric        | Score  |
| :------------ | :----- |
| **Accuracy** | 0.9998 |
| **F1 Score** | 0.9961 |
| **ROC AUC** | 1.0000 |
| **PR AUC** | 0.9997 |

The plots below provide a visual summary of the model's exceptional performance.

### Performance Curves
The ROC and Precision-Recall curves both show an Area Under the Curve (AUC) of 1.000, which signifies a near-perfect ability to distinguish between normal and anomalous logs.

![Performance Curves](image_7118ad.png)

### Confusion Matrix
The confusion matrix shows that the model is both highly precise and has excellent recall, with only 17 false positives and 4 false negatives on a test set of over 92,000 events.

![Confusion Matrix](image_7118c9.png)

### Feature Importance
The feature importance plot highlights the top 20 tokens and numerical features that were most predictive of an anomaly. This shows that the model learned to identify specific, meaningful patterns in the log data.

![Feature Importance](image_7118e8.png)

## How to Use This Project

The entire workflow is contained within the `2.ipynb` Jupyter Notebook.

### Prerequisites
To run this project, you will need Python 3 and the libraries listed in `requirements.txt`. You can install them using pip:
```bash
pip install -r requirements.txt
