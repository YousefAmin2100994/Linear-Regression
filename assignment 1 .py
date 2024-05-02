# Import necessary libraries
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Set the seaborn style for better visualizations
plt.style.use('ggplot')

# Read the data from the CSV file into a DataFrame
data_frame = pd.read_csv(r'H:\semi_colon.csv')
print(data_frame.isnull().sum())
print(data_frame.duplicated().sum())
# Display the first 5 rows of the DataFrame
print(data_frame.head())

# Rename the 'Churn' column to 'Left_Last_Month' for clarity
data_frame.rename(columns={'Churn': 'Left_Last_Month'}, inplace=True)

# Get basic information about the DataFrame
shape = data_frame.shape
columns_list = data_frame.columns.tolist()
index_list = data_frame.index.tolist()

# Plot a pie chart to show the distribution of genders 'What is the gender ratio in the company'
gender_counts = data_frame['gender'].value_counts()  # Series containing the frequency of each gender
labels = gender_counts.index.tolist()
values = gender_counts.values.tolist()

# Choose a relevant color palette for the pie chart
colors = sns.color_palette("pastel")

# Plot the pie chart
plt.pie(values, labels=labels, autopct='%1.1f%%', colors=colors)

# Add title and show the plot
plt.title('Distribution of Genders')
plt.show()

# Plot count distribution of Senior Citizens by gender
sns.countplot(x='SeniorCitizen', hue='gender', data=data_frame)

# Add title and labels
plt.title('Count of Senior Citizens by Gender')
plt.xlabel('Senior Citizen Status')
plt.ylabel('Count')

# Show the plot
plt.show()

# Plot count distribution of Partners by gender 'How does gender affect churn rate? Is there a significant difference between males and females churn rate'
sns.countplot(x='Left_Last_Month', hue='gender', data=data_frame)

# Add title and labels
plt.title('Count of Partners by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')

# Show the plot
plt.show()

# Does being a senior citizen influence the likelihood of churn
sns.countplot(x='Left_Last_Month', hue='SeniorCitizen', data=data_frame)
plt.title("Senior Citizen Effect on the Churn Rate ")
plt.xlabel("Senior Citizen")
plt.ylabel("Count")
plt.show()

# Convert 'TotalCharges' and 'MonthlyCharges' columns to numeric format
data_frame["TotalCharges"] = pd.to_numeric(data_frame["TotalCharges"], errors="coerce")
data_frame['TotalCharges'].fillna(data_frame['TotalCharges'].mean())

data_frame["MonthlyCharges"] = pd.to_numeric(data_frame["MonthlyCharges"], errors="coerce")
data_frame['MonthlyCharges'].fillna(data_frame['MonthlyCharges'].mean())

# How much charges do the custmoers mostly pay
data_frame['TotalCharges'].plot(kind='box', color='blue')
plt.title("Detecting Total Charges Outliers")
plt.xlabel("Total Charges")

data_frame['TotalCharges'].plot(kind='hist', color='green')
plt.title("Total Charges Distribution")
plt.xlabel("Total Charges")

# What is the overall churn rate in the company
temp = data_frame['Left_Last_Month'].value_counts()
y = temp.values.tolist()
x = temp.index.tolist()
plt.pie(y, labels=x, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
plt.title("Churn Rate")
plt.show()

#Is there a correleation between tenure and churn rate
x=data_frame['tenure']
y=data_frame['Left_Last_Month']
plt.bar(x,y)
plt.title("Churn Rate by Tenure")
plt.xlabel("Churn")
plt.ylabel("Tenure")
plt.show()