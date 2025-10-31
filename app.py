import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Title ---
st.title("Employee Attrition Analysis and Data Cleaning")
st.subheader("Data Analytics & Visualization Project")

# --- File Uploader ---
st.header("Upload Dataset")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # --- Load Dataset ---
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")

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

    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    missing_after = df.isnull().sum()
    st.write("**Missing values after handling:**")
    st.write(missing_after[missing_after > 0])

    if missing_after.sum() == 0:
        st.success("✅ All missing values handled successfully (Forward and Backward fill).")
    else:
        st.warning("⚠ Some missing values still remain.")

    # --- Filter by Department (if exists) ---
    if 'Department' in df.columns:
        department = st.selectbox("Select Department", df['Department'].unique())
        st.write(df[df['Department'] == department].head())

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

    # --- Attrition Analysis ---
    if 'Attrition' in df.columns:
        st.header("Attrition Analysis and Insights")

        # Overall Attrition Rate
        attrition_rate = df['Attrition'].value_counts(normalize=True) * 100
        st.subheader("Overall Attrition Rate (%)")
        st.write(attrition_rate)

        # Pie Chart for overall attrition
        st.subheader("Attrition Distribution (Pie Chart)")
        plt.figure(figsize=(5, 5))
        df['Attrition'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightgreen', 'salmon'])
        plt.title("Overall Employee Attrition")
        plt.ylabel('')
        st.pyplot(plt)

        # Department-wise attrition
        if 'Department' in df.columns:
            st.subheader("Department-wise Attrition Count")
            dept_attrition = df.groupby(['Department', 'Attrition']).size()
            st.write(dept_attrition.unstack(fill_value=0))
            most_attr_dept = dept_attrition.unstack(fill_value=0)['Yes'].idxmax()
            st.info(f"📝 The department with the highest attrition is **{most_attr_dept}**.")

        # Gender-wise attrition
        if 'Gender' in df.columns:
            st.subheader("Gender-wise Attrition Count")
            gender_attr = df.groupby(['Gender', 'Attrition']).size().unstack(fill_value=0)
            st.write(gender_attr)
            high_gender = gender_attr['Yes'].idxmax()
            st.info(f"📝 Higher attrition is observed among **{high_gender}** employees.")

        # Marital Status-wise attrition
        if 'MaritalStatus' in df.columns:
            st.subheader("Marital Status-wise Attrition Count")
            marital_attr = df.groupby(['MaritalStatus', 'Attrition']).size().unstack(fill_value=0)
            st.write(marital_attr)
            high_marital = marital_attr['Yes'].idxmax()
            st.info(f"📝 Employees who are **{high_marital}** show higher attrition rates.")

        # Business Travel-wise attrition
        if 'BusinessTravel' in df.columns:
            st.subheader("Attrition based on Business Travel")
            travel_attr = df.groupby(['BusinessTravel', 'Attrition']).size().unstack(fill_value=0)
            st.write(travel_attr)
            high_travel = travel_attr['Yes'].idxmax()
            st.info(f"📝 Employees with **{high_travel}** frequency of travel tend to leave more.")

        # Work-life balance and attrition
        if 'WorkLifeBalance' in df.columns:
            st.subheader("Attrition based on Work-Life Balance")
            worklife_attr = df.groupby(['WorkLifeBalance', 'Attrition']).size().unstack(fill_value=0)
            st.write(worklife_attr)
            high_worklife = worklife_attr['Yes'].idxmax()
            st.info(f"📝 Employees with **WorkLifeBalance = {high_worklife}** have the highest attrition.")

        # --- Visualizations ---
        st.header("Visualizations")

        # Gender-wise attrition plot
        if 'Gender' in df.columns:
            st.subheader("Gender-wise Attrition Count")
            plt.figure(figsize=(6, 4))
            sns.countplot(data=df, x='Gender', hue='Attrition')
            plt.title("Gender-wise Attrition Count")
            st.pyplot(plt)

        # Department-wise attrition plot
        if 'Department' in df.columns:
            st.subheader("Department-wise Attrition Count")
            plt.figure(figsize=(8, 5))
            sns.countplot(data=df, x='Department', hue='Attrition')
            plt.title("Department-wise Attrition Count")
            st.pyplot(plt)

        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        plt.figure(figsize=(12, 10))
        sns.heatmap(df_encoded.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap of All Features")
        st.pyplot(plt)

        # Additional Visualizations
        if 'Age' in df.columns:
            st.subheader("Distribution of Age")
            plt.figure(figsize=(6, 4))
            sns.histplot(df['Age'], bins=20, kde=True, color='skyblue')
            plt.title("Age Distribution")
            st.pyplot(plt)

        if 'MonthlyIncome' in df.columns:
            st.subheader("Monthly Income Distribution")
            plt.figure(figsize=(6, 4))
            sns.histplot(df['MonthlyIncome'], bins=20, kde=True, color='orange')
            plt.title("Monthly Income Distribution")
            st.pyplot(plt)

        if 'Attrition' in df.columns and 'MonthlyIncome' in df.columns:
            st.subheader("Monthly Income vs Attrition")
            plt.figure(figsize=(6, 4))
            sns.boxplot(data=df, x='Attrition', y='MonthlyIncome', palette="Set2")
            plt.title("Monthly Income by Attrition")
            st.pyplot(plt)

        # Scatter plot (Age vs MonthlyIncome)
        if 'Age' in df.columns and 'MonthlyIncome' in df.columns:
            st.subheader("Scatter Plot: Age vs Monthly Income")
            plt.figure(figsize=(6, 4))
            sns.scatterplot(data=df, x='Age', y='MonthlyIncome', hue='Attrition', palette="coolwarm")
            plt.title("Age vs Monthly Income (Colored by Attrition)")
            st.pyplot(plt)

        # Boxplot: Years at Company vs Attrition
        if 'YearsAtCompany' in df.columns and 'Attrition' in df.columns:
            st.subheader("Years at Company vs Attrition")
            plt.figure(figsize=(6, 4))
            sns.boxplot(data=df, x='Attrition', y='YearsAtCompany', palette="Set3")
            plt.title("Years at Company by Attrition")
            st.pyplot(plt)

        # --- Statistical Insights ---
        st.header("Statistical Insights")

        if 'Attrition' in df.columns and 'Age' in df.columns:
            mean_age_left = np.mean(df[df['Attrition'] == 'Yes']['Age'])
            st.write(f"**Average Age of Employees Who Left:** {mean_age_left:.2f}")

        if 'MonthlyIncome' in df.columns:
            mean_income = np.mean(df['MonthlyIncome'])
            median_income = np.median(df['MonthlyIncome'])
            st.write(f"**Mean Monthly Income:** {mean_income:.2f}")
            st.write(f"**Median Monthly Income:** {median_income:.2f}")

        if 'YearsAtCompany' in df.columns:
            mean_years = np.mean(df['YearsAtCompany'])
            st.write(f"**Mean Years at Company:** {mean_years:.2f}")

        # --- Download Button ---
        st.download_button(
            label="Download Cleaned Dataset",
            data=df_shuffled.to_csv(index=False),
            file_name='Cleaned_Employee_Attrition.csv',
            mime='text/csv'
        )
        st.success("✅ Data cleaning, preprocessing, visualization, and analysis completed successfully!")

else:
    st.info("👆 Please upload a CSV file to begin analysis.")
