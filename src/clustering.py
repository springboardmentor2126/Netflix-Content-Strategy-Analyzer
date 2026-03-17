# importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import os

# getting the file paths
base_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# file paths
input_path = os.path.join(base_folder, "data", "netflix_titles_featured.csv")
output_path = os.path.join(base_folder, "data", "netflix_titles_clustered.csv")


# load the csv file
def load_dataset(path):
    try:
        df = pd.read_csv(path)
        print("loaded the file!")
        print("shape:", df.shape)
        return df
    except Exception as e:
        print("something went wrong:", e)
        return None


# picking the columns that i need and converting stuff to numbers
def prepare_clustering_data(df):
    print("preparing data...")

    clustering_df = df[["num_genres", "duration_value", "rating_category"]].copy()

    le = LabelEncoder()
    clustering_df["rating_encoded"] = le.fit_transform(clustering_df["rating_category"])

    clustering_df.drop("rating_category", axis=1, inplace=True)

    clustering_df.fillna(0, inplace=True)

    return clustering_df


# this is where the actual clustering happens
def perform_clustering(data, original_df):
    print("running kmeans...")

    # scale the data first
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)

    # add the cluster labels back to the original dataframe
    original_df["cluster"] = clusters

    print("done clustering!")
    return original_df


# make a scatter plot to see the clusters
def visualize_clusters(df):
    plt.scatter(df["num_genres"], df["duration_value"], c=df["cluster"])
    plt.xlabel("Number of Genres")
    plt.ylabel("Duration")
    plt.title("Netflix Titles Clustering")
    plt.colorbar(label="Cluster")
    plt.show()


# save to csv
def save_dataset(df, path):
    try:
        df.to_csv(path, index=False)
        print("saved! location:", path)
    except Exception as e:
        print("couldnt save:", e)


# main
if __name__ == "__main__":

    print("starting...\n")

    dataset = load_dataset(input_path)

    if dataset is not None:
        clustering_data = prepare_clustering_data(dataset)
        clustered_dataset = perform_clustering(clustering_data, dataset)
        visualize_clusters(clustered_dataset)
        save_dataset(clustered_dataset, output_path)
        print("\nall done!")