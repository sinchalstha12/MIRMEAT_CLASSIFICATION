### MIRMEAT_CLASSIFICATION
### MIR-based meat classification using machine learning
**Project Overview**
This project explores whether machine learning can be used to identify different types of fresh meat, specifically Chicken, Pork, and Turkey, using a technique called Mid-Infrared (MIR) spectroscopy.

What is MIR Spectroscopy?
When infrared light is shone onto a meat sample, different molecules in the meat absorb that light at different wavelengths, measured in wavenumbers. The result is a spectral fingerprint, a unique pattern of light absorption that reflects the chemical composition of that particular meat. Think of it like a barcode, but for molecular structure.

What Are We Trying to Do?
The central question this project aims to answer is whether a machine learning model can learn to tell Chicken, Pork, and Turkey apart based solely on the patterns in their infrared spectra. To answer this, spectral data from many meat samples was fed into classification models, trained to recognise the patterns associated with each meat type, and then tested on how accurately they identify unknown samples.

Why Does It Matter?
Accurate meat classification has real-world relevance, from food safety and quality control to detecting mislabelling in the supply chain. MIR spectroscopy offers a fast, non-destructive way to analyse meat, making it a practical tool when paired with machine learning.

**Dataset**
The dataset used in this project is stored in MIRFreshMeats.csv.
It contains MIR spectroscopy readings from 120 meat samples - 40 samples each of Chicken, Pork, and Turkey.
Each sample was measured across 448 wavenumber points, ranging from 1000 to 1800 cm⁻¹. In the CSV file, each row represents a wavenumber measurement, and each column represents one meat sample.

**Methodology**
**Step 1** : Data Loading and PreparationThe dataset was loaded using pandas and transposed so that each row represents one meat sample and each column represents a wavenumber. This step is necessary because scikit-learn expects data in the format where rows are samples and columns are features, which is the opposite of how the original spectroscopy data is stored.

**Step 2** : Dimensionality Reduction (PCA)
With 448 wavenumber features and only 120 samples, the dataset has far more features than samples. Using all 448 features directly risks overfitting, where a model learns the training data too closely and struggles to generalise to new samples.
Principal Component Analysis (PCA) was applied to reduce the 448 features down to a smaller set of principal components, while keeping the most important patterns from the spectral data. Each principal component captures a portion of the total variance in the data, ordered from most to least important.
To decide how many components to keep, a scree plot (Figure 1) was used. This plot shows how much variance each component accounts for, making it easier to identify the point where adding more components gives little to no additional benefit.

**Step 3** : Binary Label Creation
The original dataset contains three meat types: Chicken, Pork, and Turkey. For Questions 2 and 3, the problem was simplified into a binary classification task by grouping Chicken and Turkey together as Bird (1) and treating Pork as NotBird (0).
This grouping tests whether the models can separate poultry from non-poultry based on spectral data alone, without needing to distinguish between the individual bird species.

**Step 4** : kNN Classification
k-Nearest Neighbours (kNN) works by looking at the k closest samples in PCA space and assigning the most common class among them to the new sample.
To find the best value of k, Stratified 5-Fold Cross Validation was used. The dataset was split into 5 equal folds, with the model trained on 4 folds and tested on the remaining 1. This process was repeated 5 times so that every fold gets a turn as the test set.
Stratified folds were used to make sure each fold contains an equal proportion of Bird and NotBird samples. This is important because it gives a more reliable measure of accuracy, especially when the class sizes are balanced.

**Step 5** : SVM Classification
Support Vector Machine (SVM) works by finding the decision boundary that creates the largest possible gap between the two classes.
Two kernel types were tested:
Linear kernel : draws a straight boundary between classes, and works best when the two classes can be cleanly separated in a straight line.
RBF kernel : draws a curved boundary instead, which is more flexible and better suited for cases where the classes overlap or cannot be separated with a straight line.

The parameter C controls how strictly the model enforces the decision boundary. A small value of C allows some training samples to be misclassified in exchange for a smoother, more generalised boundary. A large value of C pushes the model to classify every training point correctly, which can lead to a tighter but less flexible boundary.
GridSearchCV was used to test C values of 0.1, 1, 10, and 100, with the best value selected based on cross validation performance.

**Step 6** : Model Comparison
The three models were compared using two measures: test accuracy and cross validation scores.
Test accuracy shows how well each model performed on unseen data. The cross validation scores show how consistent each model was across different folds during training. Looking at both together gives a fuller picture than test accuracy alone, since a model can score well on the test set but still perform inconsistently across folds.
The results were plotted using two bar charts, one for test accuracy and one for cross validation scores, making it easy to see which model performed best and which was the most consistent.

**Results**
All three models reached 100% classification accuracy. This shows that MIR spectroscopy data alone is enough to tell Chicken, Pork, and Turkey apart with no errors.
The PCA scatter plot supports this result, showing clear separation between the three meat clusters. This confirms that the spectral differences between meat types are distinct enough for any of the three models to classify them reliably.

**Figures**
Figure 1 : PCA Scree Plot
Shows how much variance each principal component captures, used to select the number of components to keep.
Figure 2 : PCA Scatter Plot
Shows the separation between Chicken, Pork, and Turkey clusters in PCA space.
Figure 3 : kNN Cross Validation Accuracy
Shows the cross validation accuracy for each value of k tested.
Figure 4 : SVM Cross Validation Accuracy
Shows the cross validation accuracy across different values of C for both the Linear and RBF kernels.
Figure 5 : Model Comparison Bar Charts
Compares test accuracy and cross validation scores across all three models.

# How to Reproduce
1. Clone this repository
2. Install required libraries:
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import pandas as pd        
import numpy as np        
import matplotlib.pyplot as plt

4. 3. Open `code/MeatClassification.py` in Spyder or any Python IDE
5. Update the file path in Section 1 to match your local directory
6. Run each section in order from top to bottom

## Dependencies
- Python 3.x
- numpy
- pandas
- matplotlib
- scikit-learn

## Author
**Sinchal**  
Postgraduate Diploma in Science (Physics)  
University of Auckland, New Zealand
