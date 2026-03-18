import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
df = pd.read_csv("data/processed/netflix_cleaned_featured.csv")

# Select needed columns
df = df[['type', 'listed_in', 'primary_country', 'rating']].dropna()

# Label Encoding
le_genre = LabelEncoder()
le_country = LabelEncoder()
le_rating = LabelEncoder()
le_type = LabelEncoder()

df['listed_in'] = le_genre.fit_transform(df['listed_in'])
df['primary_country'] = le_country.fit_transform(df['primary_country'])
df['rating'] = le_rating.fit_transform(df['rating'])
df['type'] = le_type.fit_transform(df['type'])

# Features & target
X = df[['listed_in', 'primary_country', 'rating']]
y = df['type']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save files
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le_genre, open("le_genre.pkl", "wb"))
pickle.dump(le_country, open("le_country.pkl", "wb"))
pickle.dump(le_rating, open("le_rating.pkl", "wb"))
pickle.dump(le_type, open("le_type.pkl", "wb"))

print("✅ Model trained successfully!")