# importing the libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# get the base project folder
base_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_path = os.path.join(base_folder, "data", "netflix_titles_featured.csv")


def load_dataset(path):
    # loads csv file and returns a dataframe
    try:
        df = pd.read_csv(path)
        print("Dataset loaded successfully")
        print("Dataset shape:", df.shape)
        return df
    except Exception as error:
        print("Error loading dataset:", error)
        return None


def prepare_features(df):
    # selects features, encodes categoricals, and returns X, y plus encoders
    required_cols = ["num_genres", "duration_value", "rating_category", "type"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    print("\nPreparing features...")

    features = df[["num_genres", "duration_value", "rating_category"]].copy()
    target = df["type"].copy()

    rating_encoder = LabelEncoder()
    target_encoder = LabelEncoder()

    features["rating_encoded"] = rating_encoder.fit_transform(features["rating_category"])
    features.drop("rating_category", axis=1, inplace=True)

    target_encoded = target_encoder.fit_transform(target)

    return features, target_encoded, rating_encoder, target_encoder


def train_model(X, y):
    # trains a random forest classifier and prints accuracy on the test set
    print("\nTraining Random Forest model...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model training completed — Test Accuracy: {accuracy:.2f}")

    return model


def plot_feature_importance(model, feature_names, save_path=None):
    # plots a bar chart of feature importances, saves to file if path is given
    print("\nGenerating feature importance graph...")

    importance = model.feature_importances_

    plt.figure()
    plt.bar(feature_names, importance)
    plt.title("Feature Importance for Content Type Prediction")
    plt.xlabel("Features")
    plt.ylabel("Importance Score")

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":

    print("Starting Feature Importance Analysis...\n")

    dataset = load_dataset(input_path)

    if dataset is not None:
        X, y, _, _ = prepare_features(dataset)

        model = train_model(X, y)

        plot_feature_importance(model, X.columns)

        print("\nFeature Importance Analysis Completed")