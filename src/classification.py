# importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

# getting the file paths
base_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_path = os.path.join(base_folder, "data", "netflix_titles_featured.csv")
output_path = os.path.join(base_folder, "data", "netflix_titles_classified.csv")


# load the dataset
def load_dataset(path):
    try:
        df = pd.read_csv(path)
        print("file loaded!")
        print("shape:", df.shape)
        return df
    except Exception as e:
        print("error loading file:", e)
        return None


# prepare the data for the model
def prepare_data(df):
    print("preparing data...")

    # grabbing only the columns i need
    data = df[["num_genres", "duration_value", "rating_category", "type"]].copy()

    le = LabelEncoder()

    # encode rating
    data["rating_encoded"] = le.fit_transform(data["rating_category"])

    # encode the target column (Movie vs TV Show)
    data["type_encoded"] = le.fit_transform(data["type"])

    data.drop(["rating_category", "type"], axis=1, inplace=True)

    data.fillna(0, inplace=True)

    print("data ready!")
    return data


# train the random forest model
def train_model(data):
    print("training model...")

    # split features and target
    X = data.drop("type_encoded", axis=1)
    y = data["type_encoded"]

    # 80/20 split for train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # using random forest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # checking accuracy
    accuracy = accuracy_score(y_test, predictions)
    print("\naccuracy:", accuracy)

    print("\nclassification report:")
    print(classification_report(y_test, predictions))

    return model


# save results to csv
def save_dataset(df, path):
    try:
        df.to_csv(path, index=False)
        print("saved to:", path)
    except Exception as e:
        print("couldnt save file:", e)


# run everything
if __name__ == "__main__":

    print("starting classification...\n")

    dataset = load_dataset(input_path)

    if dataset is not None:
        prepared_data = prepare_data(dataset)
        model = train_model(prepared_data)
        save_dataset(prepared_data, output_path)
        print("\nall done!")