# TASK - 1 : Data Immersion & Wrangling
# Python Code using Pandas

# Import Libraries
import pandas as pd
import numpy as np

# -------------------------------
# 1. DATA ACCESS & FAMILIARIZATION
# -------------------------------

# Load Dataset
# Replace 'dataset.csv' with your actual dataset file name
df = pd.read_csv(r"C:\Users\tammi\Downloads\customer_churn_data.csv")

# Display first 5 rows
print("First 5 Rows:\n")
print(df.head())

# Dataset Information
print("\nDataset Info:\n")
print(df.info())

# Column Names
print("\nColumn Names:\n")
print(df.columns)

# Data Dictionary
print("\nData Types:\n")
print(df.dtypes)

# -------------------------------
# 2. DATA QUALITY ASSESSMENT
# -------------------------------

# Check Missing Values
print("\nMissing Values:\n")
print(df.isnull().sum())

# Check Duplicate Rows
print("\nDuplicate Rows:", df.duplicated().sum())

# Statistical Summary
print("\nStatistical Summary:\n")
print(df.describe())

# Detect Outliers (Numerical Columns)
numerical_cols = df.select_dtypes(include=np.number).columns

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    print(f"\nOutliers in {col}: {len(outliers)}")

# -------------------------------
# 3. DATA CLEANING & TRANSFORMATION
# -------------------------------

# Remove Duplicate Rows
df = df.drop_duplicates()

# Handle Missing Values
# Numerical columns -> fill with mean
for col in numerical_cols:
    df[col].fillna(df[col].mean(), inplace=True)

# Categorical columns -> fill with mode
categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Convert Date Columns (Example)
# Replace 'Date' with your actual date column name
# df['Date'] = pd.to_datetime(df['Date'])

# Example Feature Engineering
# Create Customer Age from Birth Year
# df['Customer_Age'] = 2025 - df['Birth_Year']

# Standardize Text Columns
for col in categorical_cols:
    df[col] = df[col].str.strip().str.lower()

# -------------------------------
# 4. OUTPUT CLEANED DATASET
# -------------------------------

# Save Cleaned Dataset
df.to_csv("cleaned_dataset.csv", index=False)

print("\nData Cleaning Completed Successfully!")
print("Cleaned dataset saved as 'cleaned_dataset.csv'")