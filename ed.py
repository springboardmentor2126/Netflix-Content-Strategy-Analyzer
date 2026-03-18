# Importing necessary libraries for data analysis and visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the cleaned Netflix dataset
df = pd.read_csv("Cleaned_netflix_titles.csv")
df.head()

# Displaying basic information about the dataset
df.info()

# Analyzing the distribution of content types
df['type'].value_counts()

sns.countplot(data=df, x='type')
plt.title("Distribution of Movies vs TV Shows")
plt.xlabel("Content Type")
plt.ylabel("Count")
plt.show()

# Visualizing the top 10 genres on Netflix
top_genres = df['primary_genre'].value_counts().head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=top_genres.values, y=top_genres.index)
plt.title("Top 10 Genres on Netflix")
plt.xlabel("Number of Titles")
plt.ylabel("Genre")
plt.show()

# Analyzing the distribution of content ratings
plt.figure(figsize=(10,5))
sns.countplot(data=df, y='rating', order=df['rating'].value_counts().index)
plt.title("Distribution of Content Ratings")
plt.xlabel("Count")
plt.ylabel("Rating")
plt.show()

# Visualizing the addition of movies and TV shows over time
plt.figure(figsize=(10,5))
sns.countplot(data=df, x='year_added', hue='type')
plt.xticks(rotation=45)
plt.title("Movies vs TV Shows Added Over Time")
plt.show()

# Analyzing the top 10 content-producing countries
top_countries = df['country'].value_counts().head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=top_countries.values, y=top_countries.index)
plt.title("Top 10 Content Producing Countries")
plt.xlabel("Number of Titles")
plt.ylabel("Country")
plt.show()

# Comparing movies and TV shows by top countries
country_type = pd.crosstab(df['country'], df['type'])
top_countries = df['country'].value_counts().head(10).index

country_type.loc[top_countries].plot(kind='bar', figsize=(10,5))
plt.title("Movies vs TV Shows by Top Countries")
plt.xlabel("Country")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Analyzing the distribution of top genres by rating
plt.figure(figsize=(12,6))
sns.countplot(data=df, y='primary_genre', hue='rating',
              order=df['primary_genre'].value_counts().head(8).index)
plt.title("Top Genres vs Rating Distribution")
plt.show()

# Categorizing content length for movies and TV shows
df['duration_num'] = df['duration'].str.extract('(\d+)').astype(int)
df[['duration', 'duration_num']].head()

df['length_category'] = None

# For Movies → categorize by minutes
df.loc[df['type'] == 'Movie', 'length_category'] = pd.cut(
    df.loc[df['type'] == 'Movie', 'duration_num'],
    bins=[0, 60, 120, 1000],
    labels=["Short", "Medium", "Long"]
)

# For TV Shows → categorize by seasons
df.loc[df['type'] == 'TV Show', 'length_category'] = pd.cut(
    df.loc[df['type'] == 'TV Show', 'duration_num'],
    bins=[0, 1, 3, 50],
    labels=["Single Season", "Few Seasons", "Many Seasons"]
)

df['length_category'].value_counts()
pd.crosstab(df['type'], df['length_category'])

df['length_category'].isna().sum()

# Visualizing content length categories by type
plt.figure(figsize=(10,5))
sns.countplot(data=df, x='length_category', hue='type')
plt.xticks(rotation=45)
plt.title("Content Length Category by Type")
plt.show()

