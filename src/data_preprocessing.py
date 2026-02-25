# ==============================
# Netflix Content Strategy Analyzer
# ==============================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------------------
# 1. Load Dataset
# ------------------------------
df = pd.read_csv("data/netflix_titles.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# Create output folder for saving visualizations
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# ------------------------------
# 2. Data Cleaning
# ------------------------------

# Remove duplicates
df.drop_duplicates(inplace=True)

# Fill missing values
df.fillna("Unknown", inplace=True)

# Convert date_added to datetime
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

print("\nData Cleaning Completed!")

# ------------------------------
# 3. Content Type Analysis
# ------------------------------

plt.figure()
sns.countplot(data=df, x='type')
plt.title("Movies vs TV Shows on Netflix")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/content_type.png")
plt.close()

# ------------------------------
# 4. Top 10 Countries
# ------------------------------

country_counts = df['country'].value_counts().head(10)

plt.figure()
country_counts.plot(kind='bar')
plt.title("Top 10 Content Producing Countries")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/top_countries.png")
plt.close()

# ------------------------------
# 5. Content Added Per Year
# ------------------------------

df['year_added'] = df['date_added'].dt.year
year_counts = df['year_added'].value_counts().sort_index()

plt.figure()
year_counts.plot(kind='line')
plt.title("Content Added Per Year")
plt.tight_layout()
plt.savefig("outputs/content_per_year.png")
plt.close()

# ------------------------------
# 6. Top 10 Genres
# ------------------------------

df['listed_in'] = df['listed_in'].str.split(',')
all_genres = df.explode('listed_in')

genre_counts = all_genres['listed_in'].value_counts().head(10)

plt.figure()
genre_counts.plot(kind='bar')
plt.title("Top 10 Genres on Netflix")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/top_genres.png")
plt.close()

print("\nAll Visualizations Saved in 'outputs' folder!")
print("Project Execution Completed Successfully!")