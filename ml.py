# Importing necessary libraries for data manipulation, visualization, and machine learning
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Reading the cleaned Netflix dataset
df = pd.read_csv("../data/Cleaned_netflix_titles.csv")

df.head()
df['duration_num'] = df['duration'].str.extract(r'(\d+)').astype(int)

df[['duration', 'duration_num']].head()

# Selecting features for the model
features = df[['release_year', 'duration_num', 'rating', 'primary_genre', 'country', 'type']].copy()
features.head()

# Encoding categorical variables into numerical values
le = LabelEncoder()
features['rating_encoded'] = le.fit_transform(features['rating'])
features['genre_encoded'] = le.fit_transform(features['primary_genre'])
features['country_encoded'] = le.fit_transform(features['country'])
features['type_encoded'] = le.fit_transform(features['type'])

features.head()

# Splitting the data into training and testing sets
X = features[['release_year', 'duration_num', 'rating_encoded', 'genre_encoded', 'country_encoded']]
y = features['type_encoded']
X.head()
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

# Training a Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Evaluating the model's accuracy
cm = confusion_matrix(y_test, y_pred)

print(cm)

# Creating a confusion matrix to analyze predictions
# ...existing code...

# Preparing features for clustering
cluster_features = features[['release_year', 'duration_num', 'rating_encoded', 'genre_encoded']]

cluster_features.head()

# Applying KMeans clustering to group similar content
kmeans = KMeans(n_clusters=4, random_state=42)

features['cluster'] = kmeans.fit_predict(cluster_features)

features[['release_year','duration_num','cluster']].head()

# Visualizing the clusters using a scatter plot
fig = px.scatter(
    features,
    x="release_year",
    y="duration_num",
    color="cluster",
    title="Netflix Content Clusters",
    hover_data=["primary_genre", "rating"]
)

fig.show()

# Calculating the average release year and duration for each cluster
features.groupby('cluster')[['release_year','duration_num']].mean()