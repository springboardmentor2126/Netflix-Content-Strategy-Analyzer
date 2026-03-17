# importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os

# finding the data file/dataset
base_folder = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# getting the file
input_path = os.path.join(
    base_folder,
    "data",
    "netflix_titles_featured.csv"
)

# checking if the file exists, if not show an error
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Could not find the dataset at: {input_path}")

# loading the csv file into a dataframe
df = pd.read_csv(input_path)

# converting the date column to a proper date format
df['date_added'] = pd.to_datetime(
    df['date_added'],
    errors='coerce'
)

# creating the main window for graphs
fig, ax = plt.subplots(figsize=(10, 6))

# leaving space at the bottom for the buttons
plt.subplots_adjust(bottom=0.2)


# graph 1 - shows how many titles netflix added each year
def plot_growth(ax):

    ax.cla()  # clear the previous graph

    data = df['year_added'].value_counts().sort_index()

    ax.plot(data.index, data.values)

    ax.set_title("Netflix Content Growth Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Shows Added")

    plt.tight_layout(rect=[0, 0.15, 1, 1])


# graph 2 - compares number of movies vs tv shows
def plot_type(ax):

    ax.cla()

    data = df['type'].value_counts()

    ax.bar(data.index, data.values)

    ax.set_title("Movies vs TV Shows")
    ax.set_xlabel("Type")
    ax.set_ylabel("Count")

    plt.tight_layout(rect=[0, 0.15, 1, 1])


# graph 3 - shows the top 10 most common genres
def plot_genre(ax):

    ax.cla()

    data = df['primary_genre'].value_counts().head(10)

    ax.bar(data.index, data.values)

    ax.set_title("Top 10 Genres on Netflix")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Count")

    plt.xticks(rotation=45, ha='right')  # rotating labels to avoid overlap
    plt.tight_layout(rect=[0, 0.15, 1, 1])


# graph 4 - shows how content is split by rating like PG, TV-MA etc
def plot_rating(ax):

    ax.cla()

    data = df['rating_category'].value_counts()

    ax.bar(data.index, data.values)

    ax.set_title("Content Rating Categories")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(rect=[0, 0.15, 1, 1])


# graph 5 - shows if content is short, medium or long
def plot_duration(ax):

    ax.cla()

    data = df['duration_category'].value_counts()

    ax.bar(data.index, data.values)

    ax.set_title("Duration Categories (Short / Medium / Long)")
    ax.set_xlabel("Duration Category")
    ax.set_ylabel("Count")

    plt.tight_layout(rect=[0, 0.15, 1, 1])


# graph 6 - shows which countries produce the most netflix content
def plot_country(ax):

    ax.cla()

    data = df['countries'].value_counts().head(10)

    ax.bar(data.index, data.values)

    ax.set_title("Top 10 Countries Producing Netflix Content")
    ax.set_xlabel("Country")
    ax.set_ylabel("Count")

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(rect=[0, 0.15, 1, 1])


# putting all graph functions in a list
plots = [
    plot_growth,
    plot_type,
    plot_genre,
    plot_rating,
    plot_duration,
    plot_country
]

# names for each graph used in the window title
plot_labels = [
    "Content Growth",
    "Type Distribution",
    "Genre Distribution",
    "Rating Distribution",
    "Duration Categories",
    "Country Contribution"
]

# tracks which graph we are currently on
current_plot = [0]


# updates the window title to show which graph we are on
def update_window_title():

    num = current_plot[0] + 1
    total = len(plots)

    label = plot_labels[current_plot[0]]

    fig.canvas.manager.set_window_title(
        f"Netflix Dashboard - {label} ({num}/{total})"
    )


# greys out the prev button on first graph and next button on last graph
def update_buttons():

    if current_plot[0] == 0:
        btn_prev.set_active(False)
        btn_prev.label.set_color('lightgrey')
    else:
        btn_prev.set_active(True)
        btn_prev.label.set_color('black')

    if current_plot[0] == len(plots) - 1:
        btn_next.set_active(False)
        btn_next.label.set_color('lightgrey')
    else:
        btn_next.set_active(True)
        btn_next.label.set_color('black')


# runs when the next button is clicked
def next_plot(event):

    current_plot[0] += 1

    plots[current_plot[0]](ax)

    update_window_title()
    update_buttons()

    fig.canvas.draw_idle()  # refreshes the screen


# runs when the previous button is clicked
def previous_plot(event):

    current_plot[0] -= 1

    plots[current_plot[0]](ax)

    update_window_title()
    update_buttons()

    fig.canvas.draw_idle()


# setting the position of the buttons on screen
ax_prev = plt.axes([0.3, 0.05, 0.15, 0.075])
ax_next = plt.axes([0.55, 0.05, 0.15, 0.075])

# creating the buttons
btn_prev = Button(ax_prev, "< Previous")
btn_next = Button(ax_next, "Next >")

# connecting buttons to their functions
btn_prev.on_clicked(previous_plot)
btn_next.on_clicked(next_plot)

# drawing the first graph when the program starts
plots[0](ax)

update_window_title()

# disabling previous button at start since we are on the first graph
update_buttons()

plt.show()