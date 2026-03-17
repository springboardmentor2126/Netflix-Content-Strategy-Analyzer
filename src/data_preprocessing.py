# Importing libraries
import pandas as pd
import os

# Getting the file paths
base_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_file = os.path.join(base_folder, "data", "netflix_titles.csv")
output_file = os.path.join(base_folder, "data", "netflix_titles_normalized.csv")


# Loading the dataset
def load_dataset(file_path):
    """This function loads the CSV file"""
    try:
        data = pd.read_csv(file_path)
        print("\nDataset loaded!")
        print("Rows and columns:", data.shape)
        return data
    except Exception as error:
        print("Couldn't load the file:", error)
        return None


# Cleaning the data
def clean_dataset(data):
    """Cleaning up missing values and fixing data types"""
    print("\nStarting to clean the data...")
    
    # Remove any duplicate rows
    data = data.drop_duplicates()
    
    # Fill missing values with 'unknown'
    data['director'] = data['director'].fillna('unknown')
    data['cast'] = data['cast'].fillna('unknown')
    data['country'] = data['country'].fillna('unknown')
    data['rating'] = data['rating'].fillna('not rated')
    data['duration'] = data['duration'].fillna('unknown')
    data['listed_in'] = data['listed_in'].fillna('unknown')
    
    # Convert date_added to datetime format
    data['date_added'] = pd.to_datetime(data['date_added'], errors='coerce')
    
    print("Cleaning completed")
    return data


# Function to count how many items are in a comma-separated string
def count_items(text):
    if text == 'unknown':
        return 0
    # Split by comma and count
    items = text.split(',')
    count = 0
    for item in items:
        if item.strip():
            count += 1
    return count


# Function to get the first genre
def get_primary_genre(genres):
    if genres == 'unknown':
        return 'unknown'
    # Just get the first one before the comma
    first_genre = genres.split(',')[0].strip()
    return first_genre


# Function to extract the number from duration (like "90 min" -> 90)
def extract_duration_value(duration):
    try:
        parts = duration.split(' ')
        number = int(parts[0])
        return number
    except:
        return None


# Function to extract the unit from duration (like "90 min" -> "min")
def extract_duration_unit(duration):
    try:
        parts = duration.split(' ')
        unit = parts[1].lower()
        return unit
    except:
        return 'unknown'


# Function to categorize ratings
def classify_rating(rating):
    # Different rating categories
    kids_ratings = ['tv-y', 'tv-y7', 'g']
    teen_ratings = ['tv-pg', 'pg', 'pg-13']
    adult_ratings = ['tv-ma', 'r', 'nc-17']
    family_ratings = ['tv-g']
    
    rating = rating.lower()
    
    if rating in kids_ratings:
        return 'kids'
    elif rating in teen_ratings:
        return 'teens'
    elif rating in family_ratings:
        return 'family'
    elif rating in adult_ratings:
        return 'adults'
    else:
        return 'not rated'


# Normalizing the dataset
def normalize_dataset(data):
    """Creating a normalized version with new columns"""
    print("\nNormalizing the dataset...")
    
    # Create a new empty dataframe
    new_data = pd.DataFrame()
    
    # Copy basic columns
    new_data['show_id'] = data['show_id']
    new_data['type'] = data['type']
    new_data['title'] = data['title']
    
    # Director columns
    new_data['directors'] = data['director']
    new_data['num_directors'] = data['director'].apply(count_items)
    
    # Cast columns
    new_data['cast_members'] = data['cast']
    new_data['num_cast'] = data['cast'].apply(count_items)
    
    # Genre columns
    new_data['genres'] = data['listed_in']
    new_data['num_genres'] = data['listed_in'].apply(count_items)
    new_data['primary_genre'] = data['listed_in'].apply(get_primary_genre)
    
    # Release information
    new_data['release_year'] = data['release_year']
    new_data['date_added'] = data['date_added']
    
    # Rating columns
    new_data['rating'] = data['rating']
    new_data['rating_category'] = data['rating'].apply(classify_rating)
    
    # Duration columns
    new_data['duration_value'] = data['duration'].apply(extract_duration_value)
    new_data['duration_unit'] = data['duration'].apply(extract_duration_unit)
    
    # Country column
    new_data['countries'] = data['country']
    
    # Description
    new_data['description'] = data['description']
    
    print("Normalization done!")
    print("New shape:", new_data.shape)
    
    return new_data


# Function to save the dataset
def save_dataset(data, save_path):
    """Saves the dataframe to a CSV file"""
    try:
        data.to_csv(save_path, index=False)
        print("\nFile saved successfully!")
        print("Location:", save_path)
    except Exception as error:
        print("Error saving file:", error)


# Main code
if __name__ == "__main__":
    print("Base directory:", base_folder)
    print("Input file:", input_file)
    
    # Load the data
    df = load_dataset(input_file)
    
    if df is not None:
        # Clean it
        df = clean_dataset(df)
        
        # Normalize it
        normalized_df = normalize_dataset(df)
        
        # Show a preview
        print("\nFirst few rows:")
        print(normalized_df.head())
        
        print("\nDataset information:")
        print(normalized_df.info())
        
        # Save the result
        save_dataset(normalized_df, output_file)
        
        print("\nMilestone 1 completed.")