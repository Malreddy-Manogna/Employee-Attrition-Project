import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page title ---
st.title("Employee Attrition Analysis and Data Cleaning")
st.subheader("Data Analytics & Visualization Project")

# --- Load dataset ---
df = pd.read_csv("C:/II year 2025-26/DAV/Employee Attrition Analyser/HR-Employee-Attrition.csv")

# --- Dataset Overview ---
st.header("Dataset Summary and Information")
st.write(f"**Shape of Dataset:** {df.shape[0]} rows and {df.shape[1]} columns")

buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()
st.text(info_str)

st.subheader("Summary Statistics")
st.write(df.describe())

# --- Handle Missing Values ---
st.header("Handling Missing Values")

missing_before = df.isnull().sum()
st.write("**Missing values before handling:**")
st.write(missing_before[missing_before > 0])

# Fill missing values
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

missing_after = df.isnull().sum()
st.write("**Missing values after handling:**")
st.write(missing_after[missing_after > 0])

if missing_after.sum() == 0:
    st.success("✅ All missing values handled successfully (Forward and Backward fill).")
else:
    st.warning("⚠ Some missing values still remain.")

# --- Encode Categorical Columns ---
st.header("Data Preprocessing: Encoding Categorical Columns")
cat_cols = df.select_dtypes(include=['object']).columns
df_encoded = df.copy()
for col in cat_cols:
    df_encoded[col] = df_encoded[col].astype('category').cat.codes

st.write(f"**Converted {len(cat_cols)} categorical columns to numeric.**")
st.dataframe(df_encoded[cat_cols].head())

# --- Normalize Numeric Columns ---
st.header("Data Transformation and Normalization")
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
df_normalized = df_encoded.copy()
df_normalized[num_cols] = scaler.fit_transform(df_encoded[num_cols])

st.write("**Z-score normalization applied to numerical features (for modeling).**")
st.write(df_normalized[num_cols].head())

# --- Shuffle Dataset ---
st.header("Data Shuffling")
st.write("Preview before shuffling:")
st.dataframe(df_normalized.head(5))

df_shuffled = df_normalized.sample(frac=1, random_state=42).reset_index(drop=True)
st.write("Preview after shuffling:")
st.dataframe(df_shuffled.head(5))

st.success("✅ Dataset shuffled successfully.")

# --- Attrition Analysis with Descriptions ---
st.header("Attrition Analysis and Insights")

# Overall attrition rate
attrition_rate = df['Attrition'].value_counts(normalize=True) * 100
st.subheader("Overall Attrition Rate (%)")
st.write(attrition_rate)

# Department-wise attrition
st.subheader("Department-wise Attrition Count")
dept_attrition = df.groupby(['Department', 'Attrition']).size()
st.write(dept_attrition.unstack(fill_value=0))
most_attr_dept = dept_attrition.unstack(fill_value=0)['Yes'].idxmax()
st.info(f"📝 The department with the highest attrition is **{most_attr_dept}**.")

# Gender-wise attrition
st.subheader("Gender-wise Attrition Count")
gender_attr = df.groupby(['Gender', 'Attrition']).size()
st.write(gender_attr.unstack(fill_value=0))
st.info("📝 Typically, one gender may show higher attrition, highlighting retention differences.")

# Marital Status-wise attrition
st.subheader("Marital Status-wise Attrition Count")
marital_attr = df.groupby(['MaritalStatus', 'Attrition']).size()
st.write(marital_attr.unstack(fill_value=0))
st.info("📝 Observing attrition by marital status can indicate life-stage influences on leaving.")

# Business Travel-wise attrition
st.subheader("Attrition based on Business Travel")
travel_attr = df.groupby(['BusinessTravel', 'Attrition']).size()
st.write(travel_attr.unstack(fill_value=0))
st.info("📝 Employees who travel frequently may have higher attrition due to work-life balance challenges.")

# Work-life balance and attrition
st.subheader("Attrition based on Work-Life Balance")
worklife_attr = df.groupby(['WorkLifeBalance', 'Attrition']).size()
st.write(worklife_attr.unstack(fill_value=0))
st.info("📝 Poor work-life balance is often associated with higher attrition.")

# --- Visualizations ---
st.header("Visualizations")

# Gender-wise attrition plot
st.subheader("Gender-wise Attrition Count")
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Gender', hue='Attrition')
plt.title("Gender-wise Attrition Count")
st.pyplot(plt)

# Department-wise attrition plot
st.subheader("Department-wise Attrition Count")
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Department', hue='Attrition')
plt.title("Department-wise Attrition Count")
st.pyplot(plt)

# Marital Status-wise attrition plot
st.subheader("Marital Status-wise Attrition Count")
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='MaritalStatus', hue='Attrition')
plt.title("Marital Status-wise Attrition Count")
st.pyplot(plt)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
plt.figure(figsize=(12,10))
sns.heatmap(df_encoded.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap of All Features")
st.pyplot(plt)

# --- Statistical Insights ---
st.header("Statistical Insights")
mean_age_left = np.mean(df[df['Attrition']=='Yes']['Age'])
mean_income = np.mean(df['MonthlyIncome'])
median_income = np.median(df['MonthlyIncome'])
mean_years = np.mean(df['YearsAtCompany'])

st.write(f"**Average Age of Employees Who Left:** {mean_age_left:.2f}")
st.write(f"**Mean Monthly Income:** {mean_income:.2f}")
st.write(f"**Median Monthly Income:** {median_income:.2f}")
st.write(f"**Mean Years at Company:** {mean_years:.2f}")

# --- Save cleaned dataset ---
cleaned_path = "Cleaned_Employee_Attrition.csv"
df_shuffled.to_csv(cleaned_path, index=False)
st.success(f"✅ Final cleaned dataset saved as: `{cleaned_path}`")
st.info("Data cleaning, preprocessing, transformation, visualization, and analysis completed successfully.")
