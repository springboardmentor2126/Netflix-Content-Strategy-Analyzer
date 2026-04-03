import ast
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="StreamScope | Netflix Dashboard", page_icon="N", layout="wide")

ROOT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
RECOMMENDER_ARTIFACT = ARTIFACTS_DIR / "recommendation_model.pkl"
PREDICTOR_ARTIFACT = ARTIFACTS_DIR / "predictor_model.pkl"

COUNTRY_COORDS = {
    "Argentina": (-34.0, -64.0),
    "Australia": (-25.0, 133.0),
    "Austria": (47.5, 14.5),
    "Belgium": (50.8, 4.5),
    "Brazil": (-14.2, -51.9),
    "Bulgaria": (42.7, 25.4),
    "Canada": (56.1, -106.3),
    "Chile": (-35.7, -71.5),
    "China": (35.9, 104.2),
    "Colombia": (4.6, -74.1),
    "Croatia": (45.1, 15.2),
    "Czech Republic": (49.8, 15.5),
    "Denmark": (56.0, 10.0),
    "Egypt": (26.8, 30.8),
    "Finland": (61.9, 25.7),
    "France": (46.2, 2.2),
    "Germany": (51.2, 10.4),
    "Greece": (39.1, 21.8),
    "Hong Kong": (22.3, 114.2),
    "Hungary": (47.2, 19.5),
    "India": (20.6, 78.9),
    "Indonesia": (-0.8, 113.9),
    "Iran": (32.4, 53.7),
    "Ireland": (53.4, -8.0),
    "Israel": (31.0, 35.0),
    "Italy": (41.9, 12.6),
    "Japan": (36.2, 138.2),
    "Kenya": (0.0, 37.9),
    "Malaysia": (4.2, 102.0),
    "Mexico": (23.6, -102.5),
    "Netherlands": (52.1, 5.3),
    "New Zealand": (-40.9, 174.9),
    "Nigeria": (9.1, 8.7),
    "Norway": (60.5, 8.5),
    "Pakistan": (30.4, 69.3),
    "Peru": (-9.1, -75.0),
    "Philippines": (12.9, 122.8),
    "Poland": (52.1, 19.1),
    "Portugal": (39.4, -8.2),
    "Romania": (45.9, 24.9),
    "Russia": (61.5, 105.3),
    "Saudi Arabia": (23.9, 45.1),
    "Singapore": (1.3, 103.8),
    "South Africa": (-30.6, 22.9),
    "South Korea": (35.9, 127.8),
    "Spain": (40.5, -3.7),
    "Sweden": (60.1, 18.6),
    "Switzerland": (46.8, 8.2),
    "Taiwan": (23.7, 121.0),
    "Thailand": (15.9, 101.0),
    "Turkey": (39.0, 35.2),
    "United Arab Emirates": (24.3, 54.4),
    "United Kingdom": (55.4, -3.4),
    "United States": (37.1, -95.7),
    "Uruguay": (-32.5, -55.8),
    "Venezuela": (6.4, -66.6),
    "Vietnam": (14.1, 108.3),
}


@st.cache_data
def parse_list_like(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_duration_minutes(value):
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if "min" not in text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else None


def parse_seasons(value):
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if "season" not in text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else None


@st.cache_data
def load_data():
    enhanced_path = ROOT_DIR / "netflix_titles_enhanced.csv"
    cleaned_path = ROOT_DIR / "netflix_titles_cleaned.csv"

    if enhanced_path.exists():
        data = pd.read_csv(enhanced_path)
    elif cleaned_path.exists():
        data = pd.read_csv(cleaned_path)
    else:
        raise FileNotFoundError("No input dataset found.")

    if "genre_list_parsed" in data.columns:
        data["genre_items"] = data["genre_list_parsed"].apply(parse_list_like)
    elif "genre_list" in data.columns:
        data["genre_items"] = data["genre_list"].apply(parse_list_like)
    else:
        data["genre_items"] = data["listed_in"].fillna("").apply(parse_list_like)

    if "country_list_parsed" in data.columns:
        data["country_items"] = data["country_list_parsed"].apply(parse_list_like)
    elif "country_list" in data.columns:
        data["country_items"] = data["country_list"].apply(parse_list_like)
    else:
        data["country_items"] = data["country"].fillna("").apply(parse_list_like)

    data["release_year"] = pd.to_numeric(data.get("release_year"), errors="coerce")

    if "year_added" not in data.columns:
        data["date_added_dt"] = pd.to_datetime(data.get("date_added"), errors="coerce")
        data["year_added"] = data["date_added_dt"].dt.year
    else:
        data["year_added"] = pd.to_numeric(data.get("year_added"), errors="coerce")
        data["date_added_dt"] = pd.to_datetime(data.get("date_added"), errors="coerce")

    data["rating"] = data.get("rating", "Unknown").fillna("Unknown").astype(str)
    data["type"] = data.get("type", "Unknown").fillna("Unknown").astype(str)
    data["content_origin"] = data.get("content_origin", "Unknown").fillna("Unknown").astype(str)
    data["title"] = data.get("title", "Untitled").fillna("Untitled").astype(str)
    data["description"] = data.get("description", "").fillna("").astype(str)
    data["cast"] = data.get("cast", "").fillna("").astype(str)

    if "primary_country" not in data.columns:
        data["primary_country"] = data["country_items"].apply(lambda x: x[0] if x else "Unknown")
    else:
        data["primary_country"] = data["primary_country"].fillna("Unknown").astype(str)

    data["movie_minutes"] = data.get("duration", "").apply(parse_duration_minutes)
    data["tv_seasons"] = data.get("duration", "").apply(parse_seasons)

    data["mood_bucket"] = "General Pick"
    data.loc[(data["type"] == "Movie") & (data["movie_minutes"].fillna(0) < 90), "mood_bucket"] = "Short and Sweet"
    data.loc[(data["type"] == "TV Show") & (data["tv_seasons"].fillna(0) > 5), "mood_bucket"] = "Deep Dives"

    return data


@st.cache_resource
def build_recommendation_model(data):
    if RECOMMENDER_ARTIFACT.exists():
        saved = joblib.load(RECOMMENDER_ARTIFACT)
        return saved["vectorizer"], saved["tfidf_matrix"], saved["title_to_idx"]

    corpus = data["description"].fillna("").astype(str)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    title_to_idx = pd.Series(data.index.values, index=data["title"]).drop_duplicates()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "tfidf_matrix": tfidf_matrix,
            "title_to_idx": title_to_idx,
        },
        RECOMMENDER_ARTIFACT,
    )
    return vectorizer, tfidf_matrix, title_to_idx


@st.cache_resource
def build_predictor_model(data):
    if PREDICTOR_ARTIFACT.exists():
        saved = joblib.load(PREDICTOR_ARTIFACT)
        return saved["vectorizer"], saved["model"]

    train = data[(data["description"].str.len() > 0) & (data["type"].isin(["Movie", "TV Show"]))].copy()
    text = (train["description"].fillna("") + " " + train["cast"].fillna("")).str.strip()
    y = train["type"]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=4000, ngram_range=(1, 2), min_df=2)
    x = vectorizer.fit_transform(text)
    model = LogisticRegression(max_iter=1200)
    model.fit(x, y)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "model": model}, PREDICTOR_ARTIFACT)
    return vectorizer, model


def render_overview(data):
    st.subheader("Overview")
    st.caption("Global KPI view with quick filtering and data table.")

    min_year = int(data["release_year"].dropna().min())
    max_year = int(data["release_year"].dropna().max())
    years = st.slider("Release year range", min_year, max_year, (min_year, max_year), key="ov_year")

    type_options = sorted(data["type"].unique().tolist())
    sel_types = st.multiselect("Title type", type_options, default=type_options, key="ov_type")

    filtered = data[data["type"].isin(sel_types)]
    filtered = filtered[filtered["release_year"].between(years[0], years[1], inclusive="both")]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Titles", f"{len(filtered):,}")
    c2.metric("Movies", f"{int((filtered['type'] == 'Movie').sum()):,}")
    c3.metric("TV Shows", f"{int((filtered['type'] == 'TV Show').sum()):,}")

    top_country = (
        filtered["primary_country"].value_counts().index[0] if not filtered.empty else "NA"
    )
    st.metric("Top Country", top_country)

    trend = (
        filtered.dropna(subset=["year_added"])
        .groupby(["year_added", "type"], as_index=False)
        .size()
        .rename(columns={"size": "titles"})
    )
    fig = px.line(
        trend,
        x="year_added",
        y="titles",
        color="type",
        markers=True,
        title="Titles Added Each Year by Type",
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    ov_left, ov_right = st.columns(2)

    type_share = (
        filtered.groupby("type", as_index=False)
        .size()
        .rename(columns={"size": "titles"})
        .sort_values("titles", ascending=False)
    )
    fig_type_share = px.pie(
        type_share,
        names="type",
        values="titles",
        hole=0.45,
        title="Catalog Composition by Type",
    )
    fig_type_share.update_layout(template="plotly_white")
    ov_left.plotly_chart(fig_type_share, use_container_width=True)

    year_dist = filtered.dropna(subset=["release_year"]).copy()
    fig_year_dist = px.histogram(
        year_dist,
        x="release_year",
        color="type",
        barmode="overlay",
        nbins=30,
        title="Release Year Distribution",
        labels={"release_year": "Release Year", "count": "Titles"},
    )
    fig_year_dist.update_layout(template="plotly_white")
    ov_right.plotly_chart(fig_year_dist, use_container_width=True)

    st.dataframe(
        filtered[["title", "type", "primary_country", "release_year", "rating", "duration"]]
        .sort_values("release_year", ascending=False)
        .head(300),
        use_container_width=True,
        hide_index=True,
    )


def country_aggregate(data):
    country_df = (
        data.explode("country_items")
        .dropna(subset=["country_items"])
        .groupby("country_items", as_index=False)
        .size()
        .rename(columns={"country_items": "country", "size": "titles"})
    )
    country_df["lat"] = country_df["country"].map(lambda x: COUNTRY_COORDS.get(x, (None, None))[0])
    country_df["lon"] = country_df["country"].map(lambda x: COUNTRY_COORDS.get(x, (None, None))[1])
    return country_df.dropna(subset=["lat", "lon"])


def render_global_explorer(data):
    st.subheader("Global Content Explorer")
    st.caption("Map + drill-down + country comparison.")

    country_map_df = country_aggregate(data)
    if not country_map_df.empty:
        view_state = pdk.ViewState(latitude=15, longitude=10, zoom=1.2, pitch=20)
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=country_map_df,
            get_position="[lon, lat]",
            get_radius="titles * 2200",
            get_fill_color=[220, 38, 38, 140],
            pickable=True,
        )
        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": "{country}\nTitles: {titles}"},
                map_style="mapbox://styles/mapbox/light-v10",
            ),
            use_container_width=True,
        )
    else:
        st.warning("No countries with map coordinates are available in the current dataset.")

    countries = sorted({c for row in data["country_items"] for c in row})
    selected_country = st.selectbox("Drill-down country", countries, key="gx_country")
    selected_df = data[data["country_items"].apply(lambda x: selected_country in x)]

    d1, d2 = st.columns(2)
    top_genres = (
        selected_df.explode("genre_items")
        .groupby("genre_items", as_index=False)
        .size()
        .rename(columns={"genre_items": "genre", "size": "titles"})
        .sort_values("titles", ascending=False)
        .head(10)
    )
    fig_genre = px.bar(top_genres, x="titles", y="genre", orientation="h", title=f"Top Genres in {selected_country}")
    fig_genre.update_layout(template="plotly_white", yaxis={"categoryorder": "total ascending"})
    d1.plotly_chart(fig_genre, use_container_width=True)

    avg_duration = selected_df[selected_df["type"] == "Movie"]["movie_minutes"].dropna().mean()
    d2.metric("Average Movie Duration", f"{avg_duration:.1f} min" if pd.notna(avg_duration) else "NA")
    rating_mix = (
        selected_df.groupby("rating", as_index=False)
        .size()
        .rename(columns={"size": "titles"})
        .sort_values("titles", ascending=False)
        .head(12)
    )
    fig_rating = px.bar(rating_mix, x="rating", y="titles", title=f"Rating Mix in {selected_country}")
    fig_rating.update_layout(template="plotly_white")
    d2.plotly_chart(fig_rating, use_container_width=True)

    st.markdown("#### Country Comparison Tool")
    c1, c2 = st.columns(2)
    left_country = c1.selectbox("Country A", countries, index=max(0, countries.index("India") if "India" in countries else 0), key="cmp_a")
    right_country = c2.selectbox("Country B", countries, index=max(0, countries.index("United States") if "United States" in countries else 0), key="cmp_b")

    left_df = data[data["country_items"].apply(lambda x: left_country in x)]
    right_df = data[data["country_items"].apply(lambda x: right_country in x)]

    left_rate = left_df["rating"].value_counts(normalize=True).mul(100).rename(left_country)
    right_rate = right_df["rating"].value_counts(normalize=True).mul(100).rename(right_country)
    cmp = pd.concat([left_rate, right_rate], axis=1).fillna(0).reset_index().rename(columns={"index": "rating"})
    cmp_melt = cmp.melt(id_vars="rating", var_name="country", value_name="share_pct")

    fig_cmp = px.bar(cmp_melt, x="rating", y="share_pct", color="country", barmode="group", title=f"Rating Share Comparison: {left_country} vs {right_country}")
    fig_cmp.update_layout(template="plotly_white", yaxis_title="Share (%)")
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown("#### Additional Geographic Insights")
    geo_left, geo_right = st.columns(2)

    movie_country = (
        data[data["type"] == "Movie"]
        .explode("country_items")
        .dropna(subset=["country_items", "movie_minutes"])
        .groupby("country_items", as_index=False)
        .agg(avg_minutes=("movie_minutes", "mean"), titles=("title", "count"))
        .rename(columns={"country_items": "country"})
        .sort_values("titles", ascending=False)
        .head(20)
    )
    fig_duration_country = px.scatter(
        movie_country,
        x="titles",
        y="avg_minutes",
        size="titles",
        hover_name="country",
        title="Country Scale vs Average Movie Duration",
        labels={"titles": "Movie Titles", "avg_minutes": "Avg Duration (min)"},
    )
    fig_duration_country.update_layout(template="plotly_white")
    geo_left.plotly_chart(fig_duration_country, use_container_width=True)

    selected_movie_df = selected_df[selected_df["type"] == "Movie"].dropna(subset=["movie_minutes"]).copy()
    if not selected_movie_df.empty:
        top_ratings = selected_movie_df["rating"].value_counts().head(8).index
        selected_movie_df = selected_movie_df[selected_movie_df["rating"].isin(top_ratings)]
        fig_box = px.box(
            selected_movie_df,
            x="rating",
            y="movie_minutes",
            points="outliers",
            title=f"Movie Duration Spread by Rating in {selected_country}",
            labels={"movie_minutes": "Movie Duration (min)", "rating": "Rating"},
        )
        fig_box.update_layout(template="plotly_white")
        geo_right.plotly_chart(fig_box, use_container_width=True)
    else:
        geo_right.info("Not enough movie duration data for this country.")


def render_strategy_simulator(data):
    st.subheader("Content Strategy Simulator")
    st.caption("Gap analysis, growth trends, and monthly scorecards for planning.")

    combo = (
        data.explode("genre_items")
        .dropna(subset=["genre_items"])
        .groupby(["genre_items", "rating"], as_index=False)
        .size()
        .rename(columns={"genre_items": "genre", "size": "titles"})
    )
    combo["titles_log"] = (combo["titles"] + 1).apply(lambda x: np.log(x))
    features = combo[["titles", "titles_log"]].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    combo["cluster"] = km.fit_predict(scaled)

    cluster_rank = combo.groupby("cluster", as_index=False)["titles"].mean().sort_values("titles")
    under_cluster = int(cluster_rank.iloc[0]["cluster"]) if not cluster_rank.empty else 0
    gaps = combo[combo["cluster"] == under_cluster].sort_values("titles").head(15)

    st.markdown("#### Gap Analysis: Underrepresented Genre + Rating Combinations")
    st.dataframe(gaps[["genre", "rating", "titles"]], use_container_width=True, hide_index=True)

    growth = (
        data.dropna(subset=["year_added"])
        .groupby(["year_added", "type"], as_index=False)
        .size()
        .rename(columns={"size": "titles"})
    )
    fig_growth = px.line(growth, x="year_added", y="titles", color="type", markers=True, title="Growth Trend: TV Shows vs Movies by Year Added")
    fig_growth.update_layout(template="plotly_white")
    st.plotly_chart(fig_growth, use_container_width=True)

    heat = (
        data.explode("genre_items")
        .dropna(subset=["genre_items", "rating"])
        .groupby(["genre_items", "rating"], as_index=False)
        .size()
        .rename(columns={"genre_items": "genre", "size": "titles"})
    )
    top_genres = heat.groupby("genre", as_index=False)["titles"].sum().sort_values("titles", ascending=False).head(12)["genre"]
    top_ratings = heat.groupby("rating", as_index=False)["titles"].sum().sort_values("titles", ascending=False).head(10)["rating"]
    heat_small = heat[heat["genre"].isin(top_genres) & heat["rating"].isin(top_ratings)]

    fig_heat = px.density_heatmap(
        heat_small,
        x="rating",
        y="genre",
        z="titles",
        histfunc="sum",
        color_continuous_scale="Reds",
        title="Genre-Rating Density (Top Segments)",
    )
    fig_heat.update_layout(template="plotly_white")
    st.plotly_chart(fig_heat, use_container_width=True)

    month_df = data.dropna(subset=["date_added_dt"]).copy()
    month_df["month"] = month_df["date_added_dt"].dt.to_period("M").astype(str)
    month_count = month_df.groupby("month", as_index=False).size().rename(columns={"size": "titles"}).sort_values("month")

    if len(month_count) >= 2:
        current = month_count.iloc[-1]
        prev = month_count.iloc[-2]
        delta = int(current["titles"] - prev["titles"])
        st.metric(
            f"Titles Added in {current['month']}",
            int(current["titles"]),
            delta,
            help=f"Compared with {prev['month']}.",
        )
    else:
        st.metric("Titles Added (latest month)", 0)


def render_recommendations(data):
    st.subheader("Find Your Next Binge")
    st.caption("Similarity search powered by cosine similarity + mood-based filters.")

    _, tfidf_matrix, title_to_idx = build_recommendation_model(data)

    selected_title = st.selectbox("Choose a title you like", sorted(data["title"].unique().tolist()), key="rec_title")
    mood = st.selectbox("Mood filter", ["Any", "Short and Sweet", "Deep Dives"], key="rec_mood")

    idx = int(title_to_idx[selected_title])
    sim_vector = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).ravel()
    sim_scores = list(enumerate(sim_vector))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    rec_indices = [i for i, _ in sim_scores[1:200]]
    rec_df = data.iloc[rec_indices].copy()

    if mood != "Any":
        rec_df = rec_df[rec_df["mood_bucket"] == mood]

    rec_df = rec_df[["title", "type", "primary_country", "rating", "mood_bucket", "description"]].head(5)
    st.dataframe(rec_df, use_container_width=True, hide_index=True)


def render_predictor(data):
    st.subheader("Title Type Predictor")
    st.caption("Predicts Movie vs TV Show from description and cast details.")

    vectorizer, model = build_predictor_model(data)

    user_desc = st.text_area(
        "Description",
        value="A group of teenagers uncover strange supernatural events in their town.",
        height=100,
    )
    user_cast = st.text_input("Cast", value="Millie Bobby Brown, Finn Wolfhard")

    if st.button("Predict Title Type"):
        sample = f"{user_desc} {user_cast}".strip()
        x_user = vectorizer.transform([sample])

        if x_user.nnz == 0:
            st.warning("Try adding more descriptive words!")
            return

        pred = model.predict(x_user)[0]
        proba = model.predict_proba(x_user)[0]
        classes = model.classes_
        score_map = {classes[i]: proba[i] for i in range(len(classes))}

        st.success(f"Predicted Type: {pred}")
        st.write("Confidence:")
        st.write({k: round(v, 3) for k, v in score_map.items()})

        conf_df = pd.DataFrame({"type": list(score_map.keys()), "probability": list(score_map.values())})
        fig_conf = px.bar(
            conf_df,
            x="type",
            y="probability",
            title="Prediction Confidence by Class",
            labels={"type": "Title Type", "probability": "Probability"},
        )
        fig_conf.update_layout(template="plotly_white", yaxis_range=[0, 1])
        st.plotly_chart(fig_conf, use_container_width=True)

        feature_names = vectorizer.get_feature_names_out()
        coef = model.coef_[0]

        if pred == "TV Show":
            top_idx = coef.argsort()[-15:]
            values = coef[top_idx]
        else:
            top_idx = coef.argsort()[:15]
            values = -coef[top_idx]

        words = feature_names[top_idx]
        fi = pd.DataFrame({"word": words, "importance": values}).sort_values("importance")
        fig = px.bar(
            fi,
            x="importance",
            y="word",
            orientation="h",
            title="Most Influential Words for This Prediction",
            labels={"importance": "Influence Score", "word": "Word"},
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("StreamScope - Netflix Strategy Intelligence Dashboard")
    st.caption("Milestone 4: Fully functional interactive dashboard with analytics + ML features")

    data = load_data()

    with st.sidebar:
        st.header("Navigation")
        section = st.radio(
            "Go to",
            [
                "Overview",
                "Global Content Explorer",
                "Content Strategy Simulator",
                "Find Your Next Binge",
                "Title Type Predictor",
            ],
        )

    if section == "Overview":
        render_overview(data)
    elif section == "Global Content Explorer":
        render_global_explorer(data)
    elif section == "Content Strategy Simulator":
        render_strategy_simulator(data)
    elif section == "Find Your Next Binge":
        render_recommendations(data)
    elif section == "Title Type Predictor":
        render_predictor(data)


if __name__ == "__main__":
    main()
