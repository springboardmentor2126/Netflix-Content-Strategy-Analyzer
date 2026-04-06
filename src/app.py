import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

st.set_page_config(page_title="Netflix Dashboard", layout="wide")

st.markdown("""
<style>
body { background-color: #111; color: white; }
</style>
""", unsafe_allow_html=True)

# ---- GLOBAL PLOT STYLE ----
plt.rcParams.update({
    "figure.facecolor":  "none",
    "axes.facecolor":    "none",
    "axes.edgecolor":    "#555555",
    "axes.labelcolor":   "#cccccc",
    "xtick.color":       "#cccccc",
    "ytick.color":       "#cccccc",
    "text.color":        "#cccccc",
    "grid.color":        "#333333",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "legend.facecolor":  "none",
    "legend.edgecolor":  "#555555",
    "legend.labelcolor": "#cccccc",
    "figure.dpi":        120,
})

# ---- LOAD DATA ----
BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data", "netflix_titles_featured.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# ---- SIDEBAR FILTERS ----
st.sidebar.title("Filters")

year_min   = int(df["release_year"].min())
year_max   = int(df["release_year"].max())
year_range = st.sidebar.slider("Release Year", year_min, year_max, (2000, 2020))

content_type = st.sidebar.selectbox("Type", ["All", "Movie", "TV Show"])

genres     = sorted(df["primary_genre"].dropna().unique())
genre_pick = st.sidebar.multiselect("Genre", genres)

all_countries = sorted({
    c.strip()
    for entry in df["countries"].dropna()
    for c in entry.split(",")
})
country_pick = st.sidebar.multiselect("Country", all_countries)

ratings     = sorted(df["rating_category"].dropna().unique())
rating_pick = st.sidebar.multiselect("Rating", ratings)

# ---- APPLY FILTERS ----
fdf = df.copy()
fdf = fdf[(fdf["release_year"] >= year_range[0]) & (fdf["release_year"] <= year_range[1])]

if content_type != "All":
    fdf = fdf[fdf["type"] == content_type]

if genre_pick:
    fdf = fdf[fdf["primary_genre"].isin(genre_pick)]

if country_pick:
    mask = fdf["countries"].str.contains("|".join(country_pick), case=False, na=False)
    fdf  = fdf[mask]

if rating_pick:
    fdf = fdf[fdf["rating_category"].isin(rating_pick)]

# ---- PAGE TITLE ----
st.title("Netflix Dashboard")
st.write(f"Showing {len(fdf)} titles from {year_range[0]} to {year_range[1]}")

# ---- TABS ----
tab1, tab2, tab3 = st.tabs(["Overview", "Trends & Insights", "What to Watch Now?"])


# ==============================================================
# TAB 1 - OVERVIEW
# ==============================================================
with tab1:
    if fdf.empty:
        st.warning("No results/titles found. Try changing the filters.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Titles",         len(fdf))
        col2.metric("Movies Count",         int((fdf["type"] == "Movie").sum()))
        col3.metric("TV Shows Count",       int((fdf["type"] == "TV Show").sum()))
        col4.metric("Average Release Year", int(fdf["release_year"].mean()))

        st.divider()
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Most Popular Genres")
            top_genres = fdf["primary_genre"].value_counts().head(10)
            if not top_genres.empty:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.barh(top_genres.index[::-1], top_genres.values[::-1], color="#E50914")
                ax.set_xlabel("Count")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No genre data available.")

        with col_right:
            st.subheader("Top Producing Countries")
            country_series = fdf["countries"].dropna().str.split(",").explode().str.strip()
            top_countries  = country_series.value_counts().head(10)
            if not top_countries.empty:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.barh(top_countries.index[::-1], top_countries.values[::-1], color="#888888")
                ax.set_xlabel("Count")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No country data available.")

        st.divider()
        st.subheader("Content Released Over the Years")
        yearly = fdf.groupby(["release_year", "type"]).size().unstack(fill_value=0)
        if yearly.empty:
            st.info("Not enough data for the timeline.")
        else:
            fig, ax = plt.subplots(figsize=(10, 3))
            for col in yearly.columns:
                color = "#E50914" if col == "Movie" else "#888888"
                ax.plot(yearly.index, yearly[col], label=col, color=color,
                        marker="o", markersize=3, linewidth=1.5)
            ax.set_xlabel("Year")
            ax.legend()
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig)
            plt.close(fig)


# ==============================================================
# TAB 2 - TRENDS & INSIGHTS
# ==============================================================
with tab2:
    if fdf.empty:
        st.warning("No data available for current filters.")
    else:
        st.subheader("Top Genre Per Year")
        top_genre_yearly = (
            fdf.groupby(["release_year", "primary_genre"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .groupby("release_year")
            .first()
            .reset_index()
        )
        if top_genre_yearly.empty:
            st.info("No data for top genre per year.")
        else:
            fig, ax = plt.subplots(figsize=(12, 4))
            bars = ax.bar(
                top_genre_yearly["release_year"],
                top_genre_yearly["count"],
                color="#E50914"
            )
            for bar, genre in zip(bars, top_genre_yearly["primary_genre"]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    genre,
                    ha="center", fontsize=7, rotation=90,
                    color="#cccccc"
                )
            ax.set_xlabel("Year")
            ax.set_ylabel("Titles")
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig)
            plt.close(fig)

        st.divider()
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Audience Ratings Distribution")
            rating_counts = fdf["rating_category"].value_counts()
            if not rating_counts.empty:
                fig, ax = plt.subplots(figsize=(5, 5))
                wedges, texts, autotexts = ax.pie(
                    rating_counts,
                    labels=rating_counts.index,
                    autopct="%1.0f%%",
                    startangle=140,
                    textprops={"color": "#cccccc", "fontsize": 11}
                )
                for at in autotexts:
                    at.set_color("#ffffff")
                st.pyplot(fig)
                plt.close(fig)

                rating_table = fdf.groupby("rating_category").agg(
                    Total   =("title", "count"),
                    Movies  =("type", lambda x: (x == "Movie").sum()),
                    TV_Shows=("type", lambda x: (x == "TV Show").sum())
                ).reset_index()
                st.dataframe(rating_table, hide_index=True)
            else:
                st.info("No rating data available.")

        with col_right:
            st.subheader("Production Type")
            if "production_type" in fdf.columns:
                prod_counts = fdf["production_type"].value_counts()
                if not prod_counts.empty:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.barh(prod_counts.index[::-1], prod_counts.values[::-1], color="#888888")
                    ax.set_xlabel("Count")
                    ax.spines[["top", "right"]].set_visible(False)
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("No production type data available.")
            else:
                st.info("Production type column not found in dataset.")

        st.divider()
        st.subheader("Genre Trends Over Time")
        top8_genres  = fdf["primary_genre"].value_counts().head(8).index.tolist()
        genre_yearly = fdf.groupby(["release_year", "primary_genre"]).size().unstack(fill_value=0)
        valid_genres = [g for g in top8_genres if g in genre_yearly.columns]

        if not valid_genres or genre_yearly.empty:
            st.info("Not enough genre data for trend chart.")
        else:
            genre_yearly = genre_yearly[valid_genres]
            fig, ax = plt.subplots(figsize=(12, 4))
            genre_yearly.plot(kind="area", ax=ax, stacked=True, alpha=0.7, colormap="Reds")
            ax.set_xlabel("Year")
            ax.set_ylabel("Titles")
            ax.legend(loc="upper left", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig)
            plt.close(fig)

        st.divider()
        st.subheader("Advanced Distributions")
        col1, col2 = st.columns(2)

        with col1:
            if "release_gap" in fdf.columns:
                gap_data = fdf["release_gap"].dropna()
                st.markdown("**Release Gap Distribution (Years to Reach Netflix)**")
                if gap_data.empty:
                    st.info("No release gap data available.")
                else:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.hist(gap_data, bins=20, color="#E50914", alpha=0.85, edgecolor="#111111")
                    ax.set_xlabel("Years Between Release & Netflix Addition")
                    ax.set_ylabel("Number of Titles")
                    ax.spines[["top", "right"]].set_visible(False)
                    st.pyplot(fig)
                    plt.close(fig)

        with col2:
            if "duration_value" in fdf.columns:
                st.markdown("**Duration Distribution (Movies vs TV Shows)**")
                movies = fdf[fdf["type"] == "Movie"]["duration_value"].dropna()
                shows  = fdf[fdf["type"] == "TV Show"]["duration_value"].dropna()

                data   = [s for s in [movies, shows] if len(s) > 1]
                labels = [lbl for lbl, s in zip(["Movies", "TV Shows"], [movies, shows]) if len(s) > 1]

                if len(data) == 0:
                    st.info("Not enough duration data to plot.")
                else:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    vp = ax.violinplot(data, showmeans=True, showmedians=True)
                    for body in vp["bodies"]:
                        body.set_facecolor("#E50914")
                        body.set_alpha(0.6)
                    for part in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
                        if part in vp:
                            vp[part].set_color("#cccccc")
                    ax.set_xticks(range(1, len(labels) + 1))
                    ax.set_xticklabels(labels)
                    ax.set_ylabel("Duration (Minutes / Seasons)")
                    ax.spines[["top", "right"]].set_visible(False)
                    st.pyplot(fig)
                    plt.close(fig)

        if "release_speed" in fdf.columns:
            st.markdown("**Release Speed Distribution**")
            speed_counts = fdf["release_speed"].value_counts()
            if speed_counts.empty:
                st.info("No release speed data available.")
            else:
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.barh(speed_counts.index[::-1], speed_counts.values[::-1],
                        color="#888888", alpha=0.85)
                ax.set_xlabel("No. of Titles")
                ax.set_ylabel("Release Speed Category")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig)
                plt.close(fig)
            st.caption("Most content is added within a short time after release.")


# ==============================================================
# TAB 3 - WHAT TO WATCH + WORLD CHALLENGE
# ==============================================================
with tab3:
    st.title("What Should I Watch Right Now?")
    st.write("Suggestions based on your time of day. Optionally tune them by rating your current energy.")

    profiles = {
        "Peak Focus": {
            "hours":       list(range(8, 12)),
            "genres":      ["Documentary", "Drama", "Crime", "History", "Thriller"],
            "description": "Morning focus time — your brain can handle complex and heavy content.",
            "emoji":       "🔥"
        },
        "Post-Lunch Dip": {
            "hours":       list(range(13, 16)),
            "genres":      ["Comedy", "Animation", "Reality-TV"],
            "description": "After lunch slump — keep it light and easy.",
            "emoji":       "😌"
        },
        "Afternoon Recharge": {
            "hours":       list(range(16, 19)),
            "genres":      ["Action", "Adventure", "Sci-Fi", "Fantasy"],
            "description": "Energy picks up again — good time for fast-paced content.",
            "emoji":       "⚡"
        },
        "Evening Unwind": {
            "hours":       list(range(19, 22)),
            "genres":      ["Romantic", "Drama", "Music"],
            "description": "Time to relax a bit. Pick something calm, emotional, or feel-good.",
            "emoji":       "🌇"
        },
        "Wind-Down": {
            "hours":       list(range(22, 24)),
            "genres":      ["Documentary", "Animation", "Comedy"],
            "description": "Close to bedtime — avoid thrillers, keep it calm.",
            "emoji":       "🌙"
        },
        "Late Night": {
            "hours":       list(range(0, 8)),
            "genres":      ["Comedy", "Animation", "Reality-TV"],
            "description": "Late night mode — easy breezy content only.",
            "emoji":       "🌃"
        },
    }

    def get_profile(hour):
        for name, p in profiles.items():
            if hour in p["hours"]:
                return name
        return "Evening Unwind"

    current_hour     = datetime.now().hour
    detected_profile = get_profile(current_hour)
    current_time_str = datetime.now().strftime("%I:%M %p")
    prof             = profiles[detected_profile]

    st.write(f"Current time: **{current_time_str}**")
    st.info(prof["emoji"] + " **" + detected_profile + "** — " + prof["description"])

    st.divider()

    st.markdown("**Does this match how you actually feel?** *(optional)*")
    st.caption(
        "If the detected profile does not reflect your current energy, "
        "rate yourself below and the suggestions will update accordingly."
    )
    energy = st.feedback("stars")

    energy_map = {
        0: {"genres": ["Comedy", "Animation", "Reality-TV"],        "note": "Exhausted — keeping it mindless and easy."},
        1: {"genres": ["Comedy", "Romantic", "Music"],              "note": "Tired — something light and feel-good."},
        2: {"genres": ["Drama", "Action", "Adventure"],             "note": "Moderate energy — a decent story with some depth."},
        3: {"genres": ["Crime", "Thriller", "Sci-Fi", "Fantasy"],   "note": "Focused — your brain can handle something engaging."},
        4: {"genres": ["Documentary", "History", "Drama", "Crime"], "note": "Sharp — go for complex, thought-provoking content."},
    }

    if energy is not None:
        rating_info   = energy_map[energy]
        active_genres = rating_info["genres"]
        st.success("Got it! Adjusting suggestions based on your energy — " + rating_info["note"])
    else:
        active_genres = prof["genres"]
        st.caption("Showing suggestions based on your time profile. Rate your energy above to personalise further.")

    st.divider()
    st.write("**Recommended genres:** " + ", ".join(active_genres))

    st.subheader("Recommended Titles")

    # FIX 3: escape genre names before joining into regex pattern
    pattern = "|".join(re.escape(g) for g in active_genres)
    recs    = df[df["primary_genre"].str.contains(pattern, case=False, na=False)]
    recs    = recs.sort_values("release_year", ascending=False).head(8)

    if recs.empty:
        st.warning("No recommendations found for this mood.")
    else:
        cols = st.columns(4)
        for i, (_, row) in enumerate(recs.iterrows()):
            with cols[i % 4]:
                st.write(f"**{row['title']}**")
                st.caption(f"{row['type']} · {int(row['release_year'])}")
                st.caption(row.get("primary_genre", ""))

    st.divider()
    st.subheader("Better to Avoid Now")
    all_rec_genres  = set(active_genres)
    all_genres_list = list({g for pr in profiles.values() for g in pr["genres"]})
    avoid           = [g for g in all_genres_list if g not in all_rec_genres]
    st.write(", ".join(avoid[:8]))

    # ---- NOT SATISFIED PROMPT ----
    st.divider()
    st.markdown(
        """
        <div style='text-align:center; padding: 1.2rem 0 0.5rem;'>
            <span style='color:#aaaaaa; font-size:0.95rem;'>
                😕 Not satisfied with the suggestions? Want to try something new?
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    if "show_world_challenge" not in st.session_state:
        st.session_state.show_world_challenge = False

    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        if st.button("🌍 Let's Take a New Challenge!", use_container_width=True):
            st.session_state.show_world_challenge = True

    # ---- WATCH AROUND THE WORLD (inline) ----
    if st.session_state.show_world_challenge:
        st.divider()
        st.title("Watch Around the World")
        st.write("Explore top titles from different countries and track what you've watched.")

        country_flags = {
            "United States": "🇺🇸", "India": "🇮🇳", "United Kingdom": "🇬🇧",
            "Japan": "🇯🇵", "South Korea": "🇰🇷", "France": "🇫🇷",
            "Canada": "🇨🇦", "Spain": "🇪🇸", "Germany": "🇩🇪", "Mexico": "🇲🇽",
            "Brazil": "🇧🇷", "Australia": "🇦🇺", "Italy": "🇮🇹",
        }

        # FIX 1: rewritten build_passport — logic now correctly picks up to 3 titles per country
        @st.cache_data
        def build_passport(source):
            exploded = source.assign(
                country=source["countries"].str.split(",")
            ).explode("country")
            exploded["country"] = exploded["country"].str.strip()
            exploded = exploded[exploded["country"].notna() & (exploded["country"] != "")]

            rows = []
            for country, group in exploded.groupby("country"):
                picked     = []
                picked_idx = set()

                # try to get one Movie and one TV Show first
                for ctype in ["Movie", "TV Show"]:
                    sub = group[group["type"] == ctype].sort_values("release_year", ascending=False)
                    if not sub.empty:
                        row = sub.iloc[0]
                        picked.append(row)
                        picked_idx.add(row.name)

                # fill up to 3 with whatever else is available
                if len(picked) < 3:
                    rest = group[~group.index.isin(picked_idx)].sort_values("release_year", ascending=False)
                    for _, row in rest.iterrows():
                        picked.append(row)
                        if len(picked) >= 3:
                            break

                for r in picked:
                    rows.append({
                        "country": country,
                        "flag":    country_flags.get(country, "🌐"),
                        "title":   r["title"],
                        "type":    r["type"],
                        "genre":   r.get("primary_genre", "—"),
                        "year":    int(r["release_year"]),
                        "rating":  r.get("rating_category", "—"),
                    })

            return pd.DataFrame(rows).sort_values("country").reset_index(drop=True)

        if "passport_df" not in st.session_state:
            st.session_state.passport_df = build_passport(df)
        if "watched" not in st.session_state:
            st.session_state.watched = set()

        passport = st.session_state.passport_df
        watched  = st.session_state.watched

        total_titles      = len(passport)
        watched_count     = sum(1 for _, r in passport.iterrows() if f"{r['country']}||{r['title']}" in watched)
        total_countries   = passport["country"].nunique()
        watched_countries = len({
            r["country"] for _, r in passport.iterrows()
            if f"{r['country']}||{r['title']}" in watched
        })
        pct = int((watched_count / total_titles) * 100) if total_titles > 0 else 0

        st.write(
            f"**Progress:** {watched_count} / {total_titles} titles · "
            f"{watched_countries} / {total_countries} countries · {pct}% done"
        )
        st.progress(pct / 100)

        st.divider()

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search = st.text_input(
                "Search by country or title",
                placeholder="e.g. 'Japan' or 'Dark'",
                key="world_search"
            )
        with col2:
            show_filter = st.selectbox(
                "Show", ["All", "Not Watched", "Watched"],
                key="world_filter"
            )
        with col3:
            # FIX 2: also reset watched set when starting a new challenge
            if st.button("Start New Challenge 🔄", key="world_reset"):
                build_passport.clear()
                st.session_state.passport_df = build_passport(df)
                st.session_state.watched     = set()
                st.rerun()

        view = passport.copy()
        if search:
            q    = search.lower()
            view = view[
                view["country"].str.lower().str.contains(q, na=False) |
                view["title"].str.lower().str.contains(q, na=False)
            ]

        def is_watched(row):
            return f"{row['country']}||{row['title']}" in watched

        if show_filter == "Not Watched":
            view = view[~view.apply(is_watched, axis=1)]
        elif show_filter == "Watched":
            view = view[view.apply(is_watched, axis=1)]

        st.divider()

        if view.empty:
            st.info("No matches found. Try a different search.")
        else:
            changed         = False
            current_country = None

            for _, row in view.iterrows():
                uid             = f"{row['country']}||{row['title']}"
                already_watched = uid in watched

                if row["country"] != current_country:
                    current_country = row["country"]
                    st.markdown(f"**{row['flag']} {row['country']}**")

                checked = st.checkbox(
                    f"{row['title']} · {row['type']} · {row['genre']} · {row['year']}",
                    value=already_watched,
                    key=f"world_{uid}"
                )

                if checked and uid not in watched:
                    watched.add(uid)
                    changed = True
                elif not checked and uid in watched:
                    watched.discard(uid)
                    changed = True

            if changed:
                st.session_state.watched = watched
                st.rerun()