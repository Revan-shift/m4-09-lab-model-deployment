# Iris Classifier Model Deployment API

## What the Model Does
This API serves a machine learning model (Random Forest Classifier) trained to predict the species of an Iris flower (`setosa`, `versicolor`, or `virginica`). It takes four numerical features as input: sepal length, sepal width, petal length, and petal width, and returns the predicted class along with the probability distribution for each class.

## How to Run

**1. Install Dependencies:**
Ensure you have the required Python libraries installed in your environment:
```bash
pip install flask joblib requests scikit-learn numpy pandas