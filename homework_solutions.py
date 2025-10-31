"""
ECS 171 Homework 2: Neural Network Classification on Yeast Dataset
Author: Solution File
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# PROBLEM 1: OUTLIER DETECTION [10pt]
# ============================================================================

def problem_1_outlier_detection():
    """
    Perform outlier detection using two methods and compare results
    """
    print("=" * 80)
    print("PROBLEM 1: OUTLIER DETECTION")
    print("=" * 80)
    
    # Load data
    df = pd.read_csv('data/yeast.csv', on_bad_lines='skip')
    feature_columns = ['mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc']
    
    X = df[feature_columns].values
    y = df['class_label'].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Method 1: Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    outliers_iso = iso_forest.fit_predict(X_scaled)
    
    # Method 2: Local Outlier Factor
    lof = LocalOutlierFactor(contamination=0.05)
    outliers_lof = lof.fit_predict(X_scaled)
    
    # Convert predictions to boolean (True = outlier)
    is_outlier_iso = outliers_iso == -1
    is_outlier_lof = outliers_lof == -1
    
    # Count outliers per class for each method
    print("\n--- Method 1: Isolation Forest ---")
    print(f"Total outliers detected: {np.sum(is_outlier_iso)}")
    print("\nOutliers per class:")
    for class_name in np.unique(y):
        class_mask = y == class_name
        n_outliers = np.sum(is_outlier_iso & class_mask)
        n_total = np.sum(class_mask)
        print(f"  {class_name}: {n_outliers}/{n_total} ({100*n_outliers/n_total:.1f}%)")
    
    print("\n--- Method 2: Local Outlier Factor ---")
    print(f"Total outliers detected: {np.sum(is_outlier_lof)}")
    print("\nOutliers per class:")
    for class_name in np.unique(y):
        class_mask = y == class_name
        n_outliers = np.sum(is_outlier_lof & class_mask)
        n_total = np.sum(class_mask)
        print(f"  {class_name}: {n_outliers}/{n_total} ({100*n_outliers/n_total:.1f}%)")
    
    # Calculate overlap
    overlap = np.sum(is_outlier_iso & is_outlier_lof)
    union = np.sum(is_outlier_iso | is_outlier_lof)
    
    print(f"\n--- Comparison ---")
    print(f"Overlap (detected by both): {overlap}")
    print(f"Union (detected by either): {union}")
    print(f"Agreement rate: {100*overlap/union:.1f}%")
    
    print("\n--- Assumptions ---")
    print("Isolation Forest: Assumes outliers are rare and different from normal points.")
    print("  Works well with high-dimensional data and doesn't assume data distribution.")
    print("\nLocal Outlier Factor: Assumes outliers have lower local density than neighbors.")
    print("  Sensitive to the choice of k neighbors and works well for local anomalies.")
    
    # Class distribution before/after removal
    print("\n--- Class Distribution ---")
    print("\nBefore outlier removal:")
    unique, counts = np.unique(y, return_counts=True)
    for class_name, count in zip(unique, counts):
        print(f"  {class_name}: {count}")
    
    # Use Isolation Forest for removal
    X_clean = X[~is_outlier_iso]
    y_clean = y[~is_outlier_iso]
    
    print("\nAfter outlier removal (Isolation Forest):")
    unique, counts = np.unique(y_clean, return_counts=True)
    for class_name, count in zip(unique, counts):
        print(f"  {class_name}: {count}")
    
    # Save cleaned data
    df_clean = df[~is_outlier_iso].copy()
    df_clean.to_csv('data/yeast_cleaned.csv', index=False)
    print("\nCleaned data saved to 'data/yeast_cleaned.csv'")
    
    return X_clean, y_clean


# [Rest of the code will be in next messages due to length]
# This is part 1 of the solution
