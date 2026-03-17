# importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import os

# getting file path
base_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_path = os.path.join(base_folder, "data", "netflix_titles_featured.csv")


def load_dataset(path):
    try:
        df = pd.read_csv(path)
        print("dataset loaded!")
        print("shape:", df.shape)
        return df

    except Exception as e:
        print("something went wrong while loading the file:", e)
        return None


def analyze_top_countries(df):

    print("\nfinding the top countries...")

    # splitting multi-country entries so each country gets its own row
    country_counts = (
        df["countries"]
        .str.split(", ")
        .explode()
        .value_counts()
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    country_counts.plot(kind="bar", ax=ax, color="steelblue")

    ax.set_title("Top 10 Countries Producing Netflix Content", fontsize=14)
    ax.set_xlabel("Country", fontsize=12)
    ax.set_ylabel("Number of Titles", fontsize=12)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show(block=False)

    return fig


def analyze_genres(df):

    print("\nfinding popular genres...")

    genre_counts = df["primary_genre"].value_counts().head(10)

    fig, ax = plt.subplots(figsize=(12, 6))

    genre_counts.plot(kind="bar", ax=ax, color="coral")

    ax.set_title("Top 10 Most Popular Netflix Genres", fontsize=14)
    ax.set_xlabel("Genre", fontsize=12)
    ax.set_ylabel("Number of Titles", fontsize=12)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show(block=False)

    return fig


def analyze_country_genre(df):

    print("\nlooking at genres per country...")

    df_exploded = df.copy()
    df_exploded["countries"] = df_exploded["countries"].str.split(", ")
    df_exploded = df_exploded.explode("countries")

    pivot = pd.crosstab(df_exploded["countries"], df_exploded["primary_genre"])

    top_countries = pivot.sum(axis=1).sort_values(ascending=False).head(10)
    pivot = pivot.loc[top_countries.index]

    fig, ax = plt.subplots(figsize=(14, 7))

    pivot.plot(kind="bar", stacked=True, ax=ax)

    ax.set_title("Genre Distribution Across Top 10 Countries", fontsize=14)
    ax.set_xlabel("Country", fontsize=12)
    ax.set_ylabel("Number of Titles", fontsize=12)

    plt.xticks(rotation=45, ha="right")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.show(block=False)

    return fig


if __name__ == "__main__":

    print("starting the country and genre analysis...\n")

    dataset = load_dataset(input_path)

    if dataset is not None:

        fig1 = analyze_top_countries(dataset)
        fig2 = analyze_genres(dataset)
        fig3 = analyze_country_genre(dataset)

        print("\nall done! close the chart windows to exit.")
        plt.show()

    else:
        print("could not load dataset, please check the file path.")