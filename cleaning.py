# Importing necessary libraries for data cleaning
import pandas as pd
import numpy as np

# Reading the raw Netflix dataset
df = pd.read_csv("netflix_titles.csv")
df.head()
df.info()
df.isnull().sum()

df.duplicated().sum()

# Filling missing values with appropriate replacements
df['director'].fillna("Unknown", inplace=True)
df['cast'].fillna("Not-Mentioned", inplace=True)
df['country'].fillna("Unknown", inplace=True)
df['rating'].fillna(df['rating'].mode()[0], inplace=True)
df.dropna(subset=['duration'], inplace=True)
df.dropna(subset=['date_added'], inplace=True)

df.info()
df.isnull().sum()

df['date_added'].dtype

# Converting the 'date_added' column to datetime format
df['date_added'] = df['date_added'].str.strip()
df['date_added'] = pd.to_datetime(df['date_added'])

df['date_added'].dtype

# Extracting year and month from the 'date_added' column
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month


# Cleaning and extracting primary country and genre information
df['country'] = df['country'].str.split(',').str[0]
df['country'] = df['country'].str.strip()

df['primary_genre'] = df['listed_in'].str.split(',').str[0]
df['primary_genre'] = df['primary_genre'].str.strip()

df['rating'] = df['rating'].str.strip()

df.info()

df.head()

# Saving the cleaned dataset to a new CSV file
df.to_csv("Cleaned_netflix_titles.csv", index=False)