# Kepler Exoplanet Classifier

A Decision Tree classifier that predicts the disposition of Kepler Objects of Interest (confirmed exoplanet, candidate, or false positive) using NASA's Kepler cumulative dataset.

## Overview
This project uses supervised machine learning to classify exoplanet candidates based on numerical features from the Kepler mission's cumulative dataset. The model is trained with scikit-learn's Decision Tree Classifier and evaluated using accuracy score and a confusion matrix.

## Dataset
[Kepler Exoplanet Search Results](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results) — download from Kaggle and place as cumulative.csv in the project folder before running.

## Features
- Data preprocessing with LabelEncoder
- Automatic selection of numeric features
- Train/test split (80/20)
- Decision Tree classification
- Accuracy evaluation
- Confusion matrix visualization (heatmap)

## Requirements
pandas
scikit-learn
matplotlib
seaborn

## How to Run
1. Download the dataset and place it as cumulative.csv in the same folder as the script
2. Install dependencies: pip install pandas scikit-learn matplotlib seaborn
3. Run: python ClassificationOfStars.py

## Output
The script prints the dataset columns, model accuracy, and displays a confusion matrix heatmap showing predicted vs. true classifications.