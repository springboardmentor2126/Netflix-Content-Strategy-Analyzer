import pandas as pd
import numpy as np
import os

# ==============================
# 1. Define File Path
# ==============================

DATA_PATH = os.path.join("data", "netflix_titles.csv")


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
# 4. Clean Data
# ==============================

def clean_data(df):
    print("\nCleaning data...")

    # Remove duplicates
    df = df.drop_duplicates()

    # Fill missing values
    df["director"] = df["director"].fillna("Unknown")
    df["cast"] = df["cast"].fillna("Not Available")
    df["country"] = df["country"].fillna("Unknown")
    df["rating"] = df["rating"].fillna("Not Rated")

    # Convert date column
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")

    # Drop rows where title is missing
    df = df.dropna(subset=["title"])

    print("Cleaning completed.")
    return df


# ==============================
# 5. Normalize Features
# ==============================

def normalize_features(df):
    print("\nNormalizing categorical features...")

    df["listed_in"] = df["listed_in"].str.lower().str.strip()
    df["country"] = df["country"].str.lower().str.strip()
    df["rating"] = df["rating"].str.upper().str.strip()

    # Split multiple values into lists
    df["genre_list"] = df["listed_in"].str.split(",")
    df["country_list"] = df["country"].str.split(",")

    print("Normalization completed.")
    return df


# ==============================
# 6. Save Cleaned Data
# ==============================

def save_cleaned_data(df):
    output_path = os.path.join("data", "netflix_cleaned.csv")
    df.to_csv(output_path, index=False)
    print(f"\nCleaned dataset saved at: {output_path}")


# ==============================
# MAIN FUNCTION
# ==============================

def main():
    print("Starting Netflix Data Preparation...\n")

    df = load_dataset(DATA_PATH)
    explore_data(df)

    df = clean_data(df)
    df = normalize_features(df)

    save_cleaned_data(df)

    print("\nMilestone 1 Completed Successfully!")


# ==============================
# RUN SCRIPT
# ==============================



def main():
    print("Starting Netflix Data Preparation...\n")

    df = load_dataset(DATA_PATH)
    explore_data(df)

    df = clean_data(df)
    df = normalize_features(df)

    save_cleaned_data(df)

    print("\nData shape after cleaning:", df.shape)
    print("\nSample data:")
    print(df.head())

    print("\nMilestone 1 Completed Successfully!")

if __name__ == "__main__":
    main()



    