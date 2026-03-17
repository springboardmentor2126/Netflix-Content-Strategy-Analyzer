# Milestone 2 - Feature Engineering

# importing libraries
import pandas as pd
import os

# setting up file paths
base_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_path = os.path.join(base_folder, "data", "netflix_titles_normalized.csv")
output_path = os.path.join(base_folder, "data", "netflix_titles_featured.csv")


# this function will load the dataset from csv file
def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path)
        print("dataset loaded!")
        print("shape is:", data.shape)
        return data
    except Exception as e:
        print("could not load file, error:", e)
        return None


# Feature 1 - Release Speed
# checking how fast content was added to netflix after release
# hypothesis: is that newer content gets added faster
def release_speed_category(gap):

    if gap < 0:
        return "Unknown"

    if gap <= 1:
        return "Fast"
    elif gap <= 5:
        return "Medium"
    else:
        return "Slow"


# Feature 2 - Genre Group
# checking if content has single or multiple genres
# hypothesis: netflix has more multi-genre content
def genre_group(n):

    # handle missing values
    if pd.isna(n):
        return "Unknown"

    if n == 1:
        return "Single Genre"
    elif n <= 3:
        return "Multi Genre"
    else:
        return "Highly Multi Genre"


# Feature 3 - Production Type
# checking if content is from one country or multiple countries
# hypothesis: most content comes from a single country
def production_type(country):
    country = str(country)

    parts = [c.strip() for c in country.split(",") if c.strip()]

    if len(parts) > 1:
        return "Multi Country"
    else:
        return "Single Country"


# Feature 4 - Time Period
# grouping content by the era it was released in
# hypothesis: most netflix content is from modern era
def time_period(year):

    if pd.isna(year):
        return "Unknown"

    if year < 2010:
        return "Old Era"
    elif year < 2016:
        return "Growth Era"
    else:
        return "Modern Era"


# Feature 5 - Duration Category
# grouping movies by how long they are
# hypothesis: most content is medium duration
def duration_category(value):
    try:
        value = float(value)

        if value <= 60:
            return "Short"
        elif value <= 120:
            return "Medium"
        else:
            return "Long"

    except (ValueError, TypeError):
        return "Unknown"


# this is the main function that creates all features
def create_features(df):

    print("\nstarting feature engineering...")

    # converting date_added to proper datetime format so i can extract year
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

    # Feature 6 - extracting the year content was added to netflix
    df['year_added'] = df['date_added'].dt.year

    # Feature 7 - calculating gap between release year and year added
    df['release_gap'] = df['year_added'] - df['release_year']

    # applying all the feature functions i wrote above
    df['release_speed'] = df['release_gap'].apply(release_speed_category)
    df['genre_group'] = df['num_genres'].apply(genre_group)
    df['production_type'] = df['countries'].apply(production_type)
    df['time_period'] = df['release_year'].apply(time_period)
    df['duration_category'] = df['duration_value'].apply(duration_category)

    # quick check to see how many unknowns we got in duration
    unknown_count = (df['duration_category'] == 'Unknown').sum()
    print(f"unknown duration values: {unknown_count}")

    print("feature engineering done!")
    print("new shape:", df.shape)

    return df


# saving the final dataset
def save_dataset(df, save_path):
    try:
        df.to_csv(save_path, index=False)
        print("\nfile saved at:", save_path)
    except Exception as e:
        print("error saving file:", e)


# main block
if __name__ == "__main__":

    print("input file:", input_path)

    df = load_dataset(input_path)

    if df is not None:

        featured_df = create_features(df)

        # showing first few rows to check
        print("\npreview of data:")
        print(featured_df.head())

        print("\ndataset info:")
        print(featured_df.info())

        save_dataset(featured_df, output_path)