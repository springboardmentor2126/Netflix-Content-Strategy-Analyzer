"""
Project: Netflix Content Strategy Analyzer – Insights into Global Streaming Trends

Milestone 1:
- Define project scope & success metrics
- Load Netflix dataset
- Clean dataset (handle missing values, remove duplicates)
- Normalize categorical features

Milestone 2:
- Perform exploratory data analysis
- Visualize trends
- Generate insights

Success Metrics:
- Identify top genres
- Identify top producing countries
- Analyze rating distribution
- Study yearly content growth trend
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# 1. Define File Path
# ==============================

DATA_PATH = os.path.join("data", "netflix_titles.csv")
OUTPUT_PATH = os.path.join("data", "netflix_cleaned.csv")

# ==============================
# 2. Load Dataset
# ==============================

def load_dataset(path):
    print("Loading dataset...")
    df = pd.read_csv(path)
    print("Dataset loaded successfully!")
    return df

# ==============================
# 3. Explore Data
# ==============================

def explore_data(df):
    print("\nDataset Shape:", df.shape)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDuplicate Rows:", df.duplicated().sum())

# ==============================
# 4. Clean Data (Milestone 1)
# ==============================

def clean_data(df):
    print("\nCleaning data...")

    df = df.drop_duplicates()

    df["director"] = df["director"].fillna("Unknown")
    df["cast"] = df["cast"].fillna("Not Available")
    df["country"] = df["country"].fillna("Unknown")
    df["rating"] = df["rating"].fillna("Not Rated")

    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df = df.dropna(subset=["title"])

    print("Cleaning completed.")
    return df

# ==============================
# 5. Normalize Features (Milestone 1)
# ==============================

def normalize_features(df):
    print("\nNormalizing categorical features...")

    df["listed_in"] = df["listed_in"].str.lower().str.strip()
    df["country"] = df["country"].str.lower().str.strip()
    df["rating"] = df["rating"].str.upper().str.strip()

    df["genre_list"] = df["listed_in"].str.split(",")
    df["country_list"] = df["country"].str.split(",")

    print("Normalization completed.")
    return df

# ==============================
# 6. Feature Engineering (Milestone 2)
# ==============================

def feature_engineering(df):
    print("\nCreating derived features...")

    df["duration_int"] = df["duration"].str.extract(r"(\d+)").astype(float)

    df["content_length_category"] = np.where(
        (df["type"] == "Movie") & (df["duration_int"] < 90),
        "Short Movie",
        np.where(
            (df["type"] == "Movie") & (df["duration_int"] >= 90),
            "Long Movie",
            "TV Show"
        )
    )

    df["original_vs_licensed"] = np.where(
        df["director"].str.contains("Netflix", case=False, na=False),
        "Netflix Original",
        "Licensed"
    )

    print("Feature engineering completed.")
    return df

# ==============================
# 7. EDA Visualizations (Milestone 2)
# ==============================

def analyze_content_growth(df):
    print("\nAnalyzing content growth over time...")

    df["year_added"] = df["date_added"].dt.year
    yearly_growth = df["year_added"].value_counts().sort_index()

    plt.figure(figsize=(10,5))
    yearly_growth.plot(kind='line')
    plt.title("Netflix Content Growth Over Time")
    plt.xlabel("Year")
    plt.ylabel("Number of Titles Added")
    plt.tight_layout()
    plt.show()

def visualize_genre_distribution(df):
    print("\nVisualizing genre distribution...")

    genre_counts = df["listed_in"].value_counts().head(10)

    plt.figure(figsize=(10,5))
    genre_counts.plot(kind='bar')
    plt.title("Top 10 Genres")
    plt.xlabel("Genre")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def visualize_rating_distribution(df):
    print("\nVisualizing rating distribution...")

    plt.figure(figsize=(10,5))
    sns.countplot(data=df, x="rating", order=df["rating"].value_counts().index)
    plt.xticks(rotation=45)
    plt.title("Rating Distribution")
    plt.tight_layout()
    plt.show()

def visualize_content_type(df):
    print("\nVisualizing content type distribution...")

    df["type"].value_counts().plot(kind="pie", autopct='%1.1f%%')
    plt.title("Movies vs TV Shows")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

def analyze_country_contribution(df):
    print("\nAnalyzing country-level contributions...")

    country_counts = df["country"].value_counts().head(10)

    plt.figure(figsize=(10,5))
    country_counts.plot(kind='bar')
    plt.title("Top 10 Content Producing Countries")
    plt.xlabel("Country")
    plt.ylabel("Number of Titles")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ==============================
# 8. Save Cleaned Data
# ==============================

def save_cleaned_data(df):
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nCleaned dataset saved at: {OUTPUT_PATH}")

# ==============================
# MAIN FUNCTION
# ==============================

def main():
    print("Starting Netflix Data Analysis Project...\n")

    df = load_dataset(DATA_PATH)
    explore_data(df)

    df = clean_data(df)
    df = normalize_features(df)
    df = feature_engineering(df)

    save_cleaned_data(df)

    analyze_content_growth(df)
    visualize_genre_distribution(df)
    visualize_rating_distribution(df)
    visualize_content_type(df)
    analyze_country_contribution(df)

    print("\n📊 Key Insights:")
    print("1. Movies dominate Netflix content compared to TV Shows.")
    print("2. Certain countries contribute significantly more content.")
    print("3. Drama and International genres are among the most common.")
    print("4. Content production has increased rapidly after 2015.")
    print("5. TV-MA and TV-14 are common rating categories.")

    print("\nFinal Dataset Shape:", df.shape)
    print("\nMilestone 1 & 2 Completed Successfully! 🎉")

# ==============================
# RUN SCRIPT
# ==============================

if __name__ == "__main__":
    main()