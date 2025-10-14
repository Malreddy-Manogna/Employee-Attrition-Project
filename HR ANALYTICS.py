import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("C:/II year 2025-26/DAV/Employee Attrition Analyser/HR-Employee-Attrition.csv")
df = pd.DataFrame(data)
#print(df.head())
#print(df.tail())

#summary
#print("Summary statistics: ")
#print(df.describe())

#info
#print("Dataset Info")
#print(df.info())

#total attrition rate
#attrition_rate = df['Attrition'].value_counts(normalize=True)*100
#print(attrition_rate)

#department wise attrition count
#dept_attrition = df.groupby(['Department', 'Attrition']).size()
#print("\nDepartment-wise Attrition Count")
#print(dept_attrition)

#attrition count based on business travel
#business_travel = df.groupby(['BusinessTravel','Attrition']).size()
#print("\nAttrition count based on business travel")
#print(business_travel)

#attrition count based on gender
#gender_attr=df.groupby(['Gender', 'Attrition']).size()
#print("\nAttrition count based on gender")
#print(gender_attr)

#Attrition count by education level
#print("\nAttrition count by Education Level")
#print(df.groupby(['Education', 'Attrition']).size())

#Attrition count by Education Field
#print("\nAttrition count by Education Field")
#print(df.groupby(['EducationField', 'Attrition']).size())

#Attrition count by Job Role
#print("\nAttrition count by Job Role")
#print(df.groupby(['JobRole', 'Attrition']).size())

#marital attrition and which category has the highest attrition
#marital_attrition = df.groupby(['MaritalStatus', 'Attrition']).size()
#left_counts = marital_attrition.xs('Yes', level='Attrition')
#highest_attrition_status = left_counts.idxmax()
#highest_count = left_counts.max()
#print(f"\nMarital Status with highest attrition: {highest_attrition_status} ({highest_count} employees left)")

#Attrition count by WorkLifeBalance
#worklife_attrition = df.groupby(['WorkLifeBalance', 'Attrition']).size()
#print("\nAttrition count by WorkLifeBalance")
#print(worklife_attrition)

#NUMPY

#average age of employees leaving the company
#mean_age_left = np.mean(df[df['Attrition'] == 'Yes']['Age'])
#print("The average of employees leaving the company is: ")
#print(f"{mean_age_left:.2f}")

#skewness based on age
#median_age = np.median(df['Age'])
#print(f"Median Age: {median_age}")
#if median_age < mean_age_left:
  #  print("Age distribution is slightly skewed towards younger employees.")
#else:
    #print("Age distribution is fairly balanced.")

#average income of all the employees
#mean_income = np.mean(df['MonthlyIncome'])
#print(f"\nMean Monthly Income: {mean_income:.2f}")
#if mean_income < 5000:
    #print("Employees are earning on the lower side on average.")
#else:
    #print("Employees are earning a good salary on average.")

#median of income (skewness)
#median_income = np.median(df['MonthlyIncome'])
#print(f"Median Monthly Income: {median_income}")
#if median_income < mean_income:
    #print("A few employees earn very high salaries, Salaries are not evenly spread")
#else:
 #   print("Salaries are more evenly spread among employees.")

#mean of years at company
#mean_years = np.mean(df['YearsAtCompany'])
#print(f"\nMean Years at Company: {mean_years:.2f}")
#if mean_years < 3:
#    print("Employees don't stay very long.")
#else:
#    print("Employees stay for a moderate period of time.")

#median
#median_years = np.median(df['YearsAtCompany'])
#print(f"Median Years at Company: {median_years}")
#if median_years < mean_years:
#    print("Majority of employees are new.")
#else:
#    print("Employees tenure is balanced.")

