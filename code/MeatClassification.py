# -*- coding: utf-8 -*-
"""
Created on Sun May  3 15:09:56 2026

@author: sinch
"""
# =============================================================
# MEAT CLASSIFICATION PROJECT
# Using MIR Spectroscopy Data
# Dataset: MIRFreshMeats.csv
# Goal: Classify different meat types using SVM and KNN models

# IMPORTING ALL THE NECESSARY LIBRARY FOR THE CLASSIFICATION
#

from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import pandas as pd         # pandas = our data reader and organiser
import numpy as np          # numpy = our math engine
import matplotlib.pyplot as plt  # matplotlib = our graph drawer


# =============================================================
# SECTION 1: LOAD THE DATA


df = pd.read_csv(
    'C:/Users/sinch/OneDrive/Desktop/Project01/meat_classification/MIRFreshMeats.csv', index_col=0)

# WHAT IS df?
# df stands for "dataframe" — it's like a spreadsheet inside Python.
# index_col=0 means: treat the first column (Wavenumbers) as row labels,
# not as data. We don't want to analyse the wavenumber numbers themselves —
# they are just labels telling us WHICH frequency we measured.

# Show first 5 rows - the most common way to peek at data
print(df.head())
# %%

# Save the wavenumber labels separately
# We need them later when ploting graphs
# df.index contains the row labels (wavenumbers)
wavenumbers = df.index.astype(float)

print("How many wavenumbers?", len(wavenumbers))
print("First wavenumber:", wavenumbers[0])
print("Last wavenumber:", wavenumbers[-1])

# %%
# Flip the data - rows become columns, columns become rows
# Before: rows=wavenumbers, columns=samples
# After:  rows=samples, columns=wavenumbers, we need this because
# for machine learning we need row as sample
data_transposed = df.T

print("Shape before flip:", df.shape)
print("Shape after flip:", data_transposed.shape)
# %%
####### checking the label name#####
# print(data_transposed.index[0])
print("Row 0 name:", data_transposed.index[0])
print("Row 0 data:", data_transposed.iloc[0, :5].values)

# %%
# store the rows label in variable sample_names
sample_names = data_transposed.index


# Create a function to classify meat type from sample name
# WHY A FUNCTION?
#  because function is much flexible instead of writing the same if/elif code everywhere
# we wrap it in a function and just CALL it when needed

def classify_meat_types(name):
    # .lower() converts name to lowercase first
    # so "CHICKEN", "Chicken", "chicken" all work!
    name = name.lower()

    if "chicken" in name:
        return "Chicken"
    elif "turkey" in name:
        return "Turkey"
    elif "pork" in name:
        return "Pork"
    else:
        return "Unknown"  # safety net for unexpected names


# Apply the function to ALL sample names at once
# .apply() calls our function on every single sample name
# automatically - no need to loop manually!
labels = sample_names.to_series().apply(classify_meat_types)
labels = np.array(labels)  # convert to numpy array

# Check results
print("Chicken:", np.sum(labels == 'Chicken'))
print("Pork:", np.sum(labels == 'Pork'))
print("Turkey:", np.sum(labels == 'Turkey'))

# %%
# Import PCA tool from sklearn library

# PCA compressed the  spectral features in to components
# without loosing the important Features


pca = PCA()

# Apply PCA to our transposed data
# fit_transform does two things:
#   fit       = learns the patterns in the data
#   transform = converts 448 measurements to PC scores
pca_result = pca.fit_transform(data_transposed)

print("Shape before PCA:", data_transposed.shape)
print("Shape after PCA:", pca_result.shape)
# %%
# --- -----explained variance------


# Get the explained variance ratio for each PC
# This tells us how much information each PC captures
# means PC1=85%, PC2=10%, PC3=2% etc.
explained_variance = pca.explained_variance_ratio_

# Draw the scree plot
plt.figure(figsize=(8, 5))
plt.plot(explained_variance, 'bo-')

# Labels and title
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')
plt.title('Figure 1: Scree Plot — PCA Explained Variance (MIR Meat Classification)',
          fontsize=12, fontweight='bold')

# Show the plot
plt.tight_layout()
plt.show()


# Print how much variance first 2 PCs explain
print("PC1 explains:", round(explained_variance[0]*100, 2), "%")
print("PC2 explains:", round(explained_variance[1]*100, 2), "%")
print("PC1+PC2 total:",
      round((explained_variance[0]+explained_variance[1])*100, 2), "%")


# %%

# --------scree plot-----

# Calculate cumulative variance
# cumsum() adds up the variance step by step
# Loadings plot shows which wavenumbers are most important
# Peaks = spectral regions that best separate the meat types

cumulative_variance = explained_variance.cumsum()

# Find how many PCs needed to explain 95% variance
# This automatically finds the exact number for us
pcs_needed_95 = (cumulative_variance < 0.95).sum() + 1
print("Number of PCs needed for 95% variance:", pcs_needed_95)

# Get loadings for PC1 and PC2
# Loadings = how much each wavenumber contributes to each PC

pc1_loadings = pca.components_[0]  # weights of PC1
pc2_loadings = pca.components_[1]  # weights of PC2

# Draw the loadings plot
plt.figure(figsize=(10, 5))
plt.plot(wavenumbers, pc1_loadings,
         label='PC1 Loadings', color='black')
plt.plot(wavenumbers, pc2_loadings,
         label='PC2 Loadings', color='blue')
plt.title('Figure 2: PCA Loadings vs Wavenumbers — MIR Meat Classification',
          fontsize=12, fontweight='bold')
plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Loading Value')
plt.legend()
plt.grid(True)
plt.show()

# %%

# -------PCA Scatter plot----
# PCA Scatter Plot - PC1 vs PC2
# Each dot = one meat sample
# Color shows which meat type it is
plt.figure(figsize=(8, 6))

for meat, color, marker in [('Chicken', 'red', 'o'),
                            ('Pork', 'green', 's'),
                            ('Turkey', 'blue', '^')]:
    mask = labels == meat
    plt.scatter(pca_result[mask, 0],
                pca_result[mask, 1],
                color=color,
                marker=marker,
                label=meat)

plt.xlabel(f'PC1 ({explained_variance[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({explained_variance[1]*100:.1f}% variance)')
plt.title('Figure 3: PCA Scatter Plot — PC1 vs PC2 (MIR Meat Classification)',
          fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# %%---- Question 2----
# =============================================================
# QUESTION 2: kNN CLASSIFICATION


# STEP 1 - Binary labels
# Bird=1 (Chicken+Turkey), NotBird=0 (Pork)
binary_labels = []
for label in labels:
    if label == 'Pork':
        binary_labels.append(0)
    else:
        binary_labels.append(1)
binary_labels = np.array(binary_labels)

print("Bird samples:", np.sum(binary_labels == 1))
print("Not Bird samples:", np.sum(binary_labels == 0))

# STEP 2 - Store X and y
# # Only using first 2 PCs — they capture most variance and reduce noise
X = pca_result[:, :2]
y = binary_labels       # Bird or Not Bird

# STEP 3 - Split into 80% train and 20% test
# stratify=y ensures equal Bird/NotBird ratio in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# STEP 4 - Find best k using cross validation
k_values = range(1, 16)
cv_scores = []

stratified_k_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for k in k_values:
    fold_accuracies = []
    for train_index, val_index in stratified_k_cv.split(X_train, y_train):
        X_fold_train = X_train[train_index]
        X_fold_val = X_train[val_index]
        y_fold_train = y_train[train_index]
        y_fold_val = y_train[val_index]
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_fold_train, y_fold_train)
        y_pred = knn.predict(X_fold_val)
        acc = accuracy_score(y_fold_val, y_pred)
        fold_accuracies.append(acc)
    cv_scores.append(np.mean(fold_accuracies))

best_k = k_values[np.argmax(cv_scores)]
print("Best k value:", best_k)

# STEP 5 - Train final kNN with best k
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train, y_train)

# STEP 6 - Test the model
y_pred_test = knn_final.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test,
      target_names=['NotBird', 'Bird']))


# Plot k values vs accuracy
# This shows us WHY k=5 was the best choice
plt.figure(figsize=(8, 5))
plt.plot(k_values, cv_scores, 'bo-')
plt.xlabel('Number of Neighbours (k)')
plt.ylabel('Mean CV Accuracy')
plt.title('Figure 4: Cross Validated Accuracy vs k (kNN — Bird vs NotBird)',
          fontsize=12, fontweight='bold')
plt.grid(True)
plt.tight_layout()
plt.show()
# %%

# question3------
# =============================================================
# QUESTION 3: SVM CLASSIFICATION

# WHY SVM?
# kNN looks at nearest neighbours to classify
# SVM draws the BEST boundary between Bird and NotBird
# SVM is more powerful and reliable than kNN


# C = how strict the boundary is
# Small C = flexible boundary (allows some mistakes)
# Large C = strict boundary (tries to get everything right)
param_grid_linear = {'C': [0.1, 1, 10, 100]}

# Linear SVM
linear_svm = SVC(kernel='linear')

# StratifiedKFold same as original
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV automatically tests all C values
# and picks the one with best cross validation accuracy
grid_search_linear = GridSearchCV(
    estimator=linear_svm,
    param_grid=param_grid_linear,
    cv=cv)

grid_search_linear.fit(X_train, y_train)

print("Best C for Linear SVM:", grid_search_linear.best_params_['C'])
print("Linear SVM Test Accuracy:", grid_search_linear.score(X_test, y_test))
print(classification_report(y_test, grid_search_linear.predict(X_test),
      target_names=['NotBird', 'Bird']))


# RBF (Radial Basis Function) draws a CURVED boundary
# Better when classes are not linearly separable
# gamma controls how curved the boundary is

param_grid_rbf = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01]
}
rbf_svm = SVC(kernel='rbf')

grid_search_rbf = GridSearchCV(
    estimator=rbf_svm,
    param_grid=param_grid_rbf,
    cv=cv)

grid_search_rbf.fit(X_train, y_train)

print("Best C for RBF SVM:", grid_search_rbf.best_params_['C'])
print("Best Gamma for RBF SVM:", grid_search_rbf.best_params_['gamma'])
print("RBF SVM Test Accuracy:", grid_search_rbf.score(X_test, y_test))
print(classification_report(y_test, grid_search_rbf.predict(X_test),
      target_names=['NotBird', 'Bird']))

# %%
# =============================================================
# QUESTION 4: COMPARISON SUMMARY


print("kNN Test Accuracy:        ",
      accuracy_score(y_test, y_pred_test))
print("Linear SVM Test Accuracy: ",
      grid_search_linear.score(X_test, y_test))
print("RBF SVM Test Accuracy:    ",
      grid_search_rbf.score(X_test, y_test))

print("Best k for kNN:          ", best_k)
print("Best C for Linear SVM:   ",
      grid_search_linear.best_params_['C'])
print("Best C for RBF SVM:      ",
      grid_search_rbf.best_params_['C'])


# Get best parameters directly from the models
best_k_val = best_k
best_c_linear = grid_search_linear.best_params_['C']
best_c_rbf = grid_search_rbf.best_params_['C']
best_gamma_rbf = grid_search_rbf.best_params_['gamma']

# Model labels use actual values not hardcoded
models = [f'kNN\n(k={best_k_val})',
          f'Linear SVM\n(C={best_c_linear})',
          f'RBF SVM\n(C={best_c_rbf}, γ={best_gamma_rbf})']

accuracies = [accuracy_score(y_test, y_pred_test),
              grid_search_linear.score(X_test, y_test),
              grid_search_rbf.score(X_test, y_test)]
# Get best parameters directly from the models
best_k_val = best_k
best_c_linear = grid_search_linear.best_params_['C']
best_c_rbf = grid_search_rbf.best_params_['C']
best_gamma_rbf = grid_search_rbf.best_params_['gamma']

# Model labels and accuracy scores
models = [f'kNN\n(k={best_k_val})',
          f'Linear SVM\n(C={best_c_linear})',
          f'RBF SVM\n(C={best_c_rbf}, γ={best_gamma_rbf})']

accuracies = [accuracy_score(y_test, y_pred_test),
              grid_search_linear.score(X_test, y_test),
              grid_search_rbf.score(X_test, y_test)]

# ---- BAR PLOT ----
plt.figure(figsize=(8, 5))
bars = plt.bar(models,
               [acc * 100 for acc in accuracies],
               color=['blue', 'green', 'red'],
               width=0.4)

# Add percentage on top of each bar
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.5,
             f'{acc * 100:.1f}%',
             ha='center',
             fontsize=12,
             fontweight='bold')

plt.title('Figure 5: Model Accuracy Comparison — kNN vs SVM',
          fontsize=12, fontweight='bold')
plt.xlabel('Model', fontsize=12)
plt.ylabel('Test Accuracy (%)', fontsize=12)
plt.ylim(0, 110)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
