#Pandas is a Python library used for working with data sets.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

import kagglehub

import os

warnings.filterwarnings('ignore')

os.makedirs("plots", exist_ok=True)


df_airBNB = pd.read_csv("Airbnb_Open_Data.csv")
# print(df_airBNB.tail())

#Data Preprocessing
#Examine the size of the facts (variety of rows and columns) to experience its length and complexity.

print("Number of Rows: ",df_airBNB.shape[0]) 
print("Number of columns: ", df_airBNB.shape[1])
print("Features in the frame: ",df_airBNB.columns)
# print(df_airBNB.shape)
print(df_airBNB.info())
print(df_airBNB.nunique()) #This will help us in deciding which type of encoding to choose for converting categorical columns into numerical columns
print(df_airBNB.describe(include ='all'))

# print(df_airBNB.isnull().sum())

#Handling Missing Values
df_airBNB['NAME'] = df_airBNB['NAME'].fillna("No Name")

#Categorical Data
cols = ["host_identity_verified", "instant_bookable", "cancellation_policy"]
df_airBNB[cols] = df_airBNB[cols].fillna("Unknown")
df_airBNB = df_airBNB.dropna(subset=["lat", "long"]) #drops the entire row


#NUMERICAL
df_airBNB["service fee"] = pd.to_numeric(
    df_airBNB["service fee"].str.replace("$", "", regex=False).str.replace(",", ""), errors = "coerce"
)

df_airBNB["price"] = pd.to_numeric(
    df_airBNB["price"].str.replace("$", "", regex=False).str.replace(",", ""),
    errors="coerce"
)

df_airBNB["service fee"] = df_airBNB["service fee"].fillna(df_airBNB["service fee"].median())
df_airBNB["price"] = df_airBNB["price"].fillna(df_airBNB["price"].median())


df_airBNB["minimum nights"] = df_airBNB["minimum nights"].fillna(df_airBNB["minimum nights"].median())

df_airBNB["review rate number"] = df_airBNB["review rate number"].fillna(0) #no rating = no reviews
df_airBNB["number of reviews"] = df_airBNB["number of reviews"].fillna(0)
df_airBNB["reviews per month"] = df_airBNB["reviews per month"].fillna(0)
df_airBNB["last review"] = df_airBNB["last review"].fillna("No Review")
df_airBNB = df_airBNB.drop(columns=["license", "house_rules"])
df_airBNB["host name"] = df_airBNB["host name"].fillna("Unknown Host")
df_airBNB["calculated host listings count"] = df_airBNB["calculated host listings count"].fillna(1) #host has to at least have one (the current AirBNB)
df_airBNB["neighbourhood group"] = df_airBNB["neighbourhood group"].fillna("Unknown Group") #host has to at least have one (the current AirBNB)

#fixing mispelled words, probably smarter to use NLP techniques, for now this is fine.
borough_fix = {
    "brookln":  "Brooklyn",
    "manhatan": "Manhattan",
}
df_airBNB["neighbourhood group"] = (
    df_airBNB["neighbourhood group"]
    .str.strip()
    .str.lower()
    .replace(borough_fix)
    .str.title() 
)

df_airBNB["neighbourhood"] = df_airBNB["neighbourhood"].fillna("Unknown Neighbourhood") #host has to at least have one (the current AirBNB)
df_airBNB["country"] = df_airBNB["country"].fillna("United States") #host has to at least have one (the current AirBNB)
df_airBNB["country code"] = df_airBNB["country code"].fillna("US") #host has to at least have one (the current AirBNB)
df_airBNB["Construction year"] = df_airBNB["Construction year"].fillna(df_airBNB["Construction year"].median()) #host has to at least have one (the current AirBNB)
df_airBNB["availability 365"] = df_airBNB["availability 365"].fillna(df_airBNB["availability 365"].median()) #host has to at least have one (the current AirBNB)

print(df_airBNB.isnull().sum())

#Exploring Data Characteristics
print("\n",df_airBNB["service fee"].skew())
print(df_airBNB["service fee"].kurtosis())


#Univariate Analysis 

sns.set_theme(style="whitegrid", palette="muted")
PINK = "#E91E8C"

#bar chart
fig, ax = plt.subplots(figsize=(8, 5))
room_counts = df_airBNB["room type"].value_counts()
ax.bar(room_counts.index, room_counts.values, color=PINK, edgecolor="white")
ax.set_title("Distribution of Room Types", fontsize=14, fontweight="bold")
ax.set_xlabel("Room Type")
ax.set_ylabel("Number of Listings")
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
            f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("plots/room_type.png", dpi=150)
plt.show()

# Neighbourhood Group  (bar chart)
# Shows geographic demand concentration across the five boroughs.
fig, ax = plt.subplots(figsize=(9, 5))
nb_counts = df_airBNB["neighbourhood group"].value_counts()
ax.bar(nb_counts.index, nb_counts.values, color="#6C63FF", edgecolor="white")
ax.set_title("Listings by Neighbourhood Group (Borough)", fontsize=14, fontweight="bold")
ax.set_xlabel("Borough")
ax.set_ylabel("Number of Listings")
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
            f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("plots/neighbourhood_group.png", dpi=150)
plt.show()



#  Review Rate Number in comparsion to number of listing
fig, ax = plt.subplots(figsize=(7, 5))
rr_counts = df_airBNB["review rate number"].value_counts().sort_index()
ax.bar(rr_counts.index.astype(str), rr_counts.values, color="#AB47BC", edgecolor="white")
ax.set_title("Review Rate Number Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Review Rate (0 = No Reviews)")
ax.set_ylabel("Number of Listings")
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
            f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("plots/review_rate.png", dpi=150)
plt.show()

# Price  (histogram) 
price_clean = df_airBNB["price"][df_airBNB["price"] <= 1000]
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(price_clean, bins=60, color=PINK, edgecolor="white", alpha=0.85)
ax.set_title("Price Distribution (≤ $1,000 / night)", fontsize=14, fontweight="bold")
ax.set_xlabel("Price (USD)")
ax.set_ylabel("Number of Listings")
ax.axvline(price_clean.median(), color="#333", linestyle="--", linewidth=1.5,
           label=f"Median: ${price_clean.median():.0f}")
ax.legend()
plt.tight_layout()
plt.savefig("plots/price_distribution.png", dpi=150)
plt.show()


# Number of Reviews in comparsion to amount of Number of Listings
reviews_clean = df_airBNB["number of reviews"][df_airBNB["number of reviews"] > 0]
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(reviews_clean, bins=60, color="#5C6BC0", edgecolor="white", alpha=0.85)
ax.set_yscale("log")
ax.set_title("Number of Reviews per Listing (log scale)", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Reviews")
ax.set_ylabel("Number of Listings (log scale)")
ax.axvline(reviews_clean.median(), color="#333", linestyle="--", linewidth=1.5,
           label=f"Median: {reviews_clean.median():.0f}")
ax.legend()
plt.tight_layout()
plt.savefig("plots/number_of_reviews.png", dpi=150)
plt.show()

print("\nAll univariate plots saved to plots/")


#Bivariate Analysis

# Room Type vs Price
# Reveals how listing structure drives price range and outliers per category.
biv_price = df_airBNB[df_airBNB["price"] <= 1000]
fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(data=biv_price, x="room type", y="price", palette="Set2", ax=ax)
ax.set_title("Price Distribution by Room Type", fontsize=14, fontweight="bold")
ax.set_xlabel("Room Type")
ax.set_ylabel("Price (USD, ≤ $1,000)")
plt.tight_layout()
plt.savefig("plots/biv_roomtype_vs_price.png", dpi=150)
plt.show()

# Neighbourhood Group vs Price  (boxplot)
# Compares median price and spread across NYC boroughs.
fig, ax = plt.subplots(figsize=(10, 5))
borough_order = biv_price.groupby("neighbourhood group")["price"].median().sort_values(ascending=False).index
sns.boxplot(data=biv_price, x="neighbourhood group", y="price", order=borough_order, palette="coolwarm", ax=ax)
ax.set_title("Price Distribution by Borough", fontsize=14, fontweight="bold")
ax.set_xlabel("Borough")
ax.set_ylabel("Price (USD, ≤ $1,000)")
plt.tight_layout()
plt.savefig("plots/biv_borough_vs_price.png", dpi=150)
plt.show()

# Room Type vs Average Review Rate  (bar chart of means)
# Shows whether guests rate entire homes differently from private/shared rooms.
fig, ax = plt.subplots(figsize=(8, 5))
avg_review = df_airBNB[df_airBNB["review rate number"] > 0].groupby("room type")["review rate number"].mean().sort_values(ascending=False)
ax.bar(avg_review.index, avg_review.values, color="#AB47BC", edgecolor="white")
ax.set_title("Average Review Rate by Room Type", fontsize=14, fontweight="bold")
ax.set_xlabel("Room Type")
ax.set_ylabel("Average Review Rate")
ax.set_ylim(0, 5)
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig("plots/biv_roomtype_vs_reviewrate.png", dpi=150)
plt.show()

# Price vs Number of Reviews  (scatter plot, sampled)
# Tests whether cheaper listings accumulate more reviews (proxy for more bookings).
sample = df_airBNB[(df_airBNB["price"] <= 1000) & (df_airBNB["number of reviews"] > 0)].sample(3000, random_state=42)
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(sample["price"], sample["number of reviews"], alpha=0.3, color=PINK, s=15, edgecolors="none")
ax.set_title("Price vs Number of Reviews (random sample of 3,000)", fontsize=14, fontweight="bold")
ax.set_xlabel("Price (USD)")
ax.set_ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig("plots/biv_price_vs_reviews.png", dpi=150)
plt.show()

# Neighbourhood Group vs Average Availability  (bar chart of means)
# Highlights which boroughs have more listings actively available throughout the year.
fig, ax = plt.subplots(figsize=(9, 5))
avg_avail = df_airBNB.groupby("neighbourhood group")["availability 365"].mean().sort_values(ascending=False)
ax.bar(avg_avail.index, avg_avail.values, color="#26A69A", edgecolor="white")
ax.set_title("Average Availability (Days/Year) by Borough", fontsize=14, fontweight="bold")
ax.set_xlabel("Borough")
ax.set_ylabel("Average Days Available (out of 365)")
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("plots/biv_borough_vs_availability.png", dpi=150)
plt.show()

print("\nAll bivariate plots saved to plots/")


#Multivariate Analysis

# Correlation Heatmap (numeric features)
# Shows which numeric variables move together — key for spotting price drivers.
num_cols = ["price", "service fee", "minimum nights", "availability 365",
            "number of reviews", "reviews per month", "review rate number",
            "calculated host listings count"]
corr = df_airBNB[num_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, ax=ax, annot_kws={"size": 9})
ax.set_title("Correlation Heatmap — Numeric Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/multi_corr_heatmap.png", dpi=150)
plt.show()

# Borough × Room Type → Median Price  (pivot heatmap)
# Reveals which borough–room-type combos command the highest prices.
pivot = df_airBNB[df_airBNB["price"] <= 1000].pivot_table(
    values="price", index="neighbourhood group", columns="room type", aggfunc="median"
)
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", linewidths=0.5, ax=ax,
            annot_kws={"size": 10})
ax.set_title("Median Price by Borough & Room Type", fontsize=14, fontweight="bold")
ax.set_xlabel("Room Type")
ax.set_ylabel("Borough")
plt.tight_layout()
plt.savefig("plots/multi_borough_roomtype_price.png", dpi=150)
plt.show()

# Price vs Reviews colored by Room Type  (scatter, sampled)
# Three-way view: does the price–review relationship shift by room type?
sample_mv = df_airBNB[
    (df_airBNB["price"] <= 1000) & (df_airBNB["number of reviews"] > 0)
].sample(3000, random_state=42)
fig, ax = plt.subplots(figsize=(10, 6))
palette_rt = {"Entire home/apt": "#E91E8C", "Private room": "#6C63FF",
              "Shared room": "#26A69A", "Hotel room": "#FFA726"}
for rt, grp in sample_mv.groupby("room type"):
    ax.scatter(grp["price"], grp["number of reviews"], alpha=0.3, s=15,
               edgecolors="none", label=rt, color=palette_rt.get(rt, "#999"))
ax.set_title("Price vs Reviews by Room Type (sample 3,000)", fontsize=14, fontweight="bold")
ax.set_xlabel("Price (USD)")
ax.set_ylabel("Number of Reviews")
ax.legend(title="Room Type", fontsize=9)
plt.tight_layout()
plt.savefig("plots/multi_price_reviews_roomtype.png", dpi=150)
plt.show()

# Price vs Availability colored by Borough  (scatter, sampled)
# Checks if high-availability listings cluster at certain price points per borough.
sample_avail = df_airBNB[df_airBNB["price"] <= 1000].sample(3000, random_state=7)
fig, ax = plt.subplots(figsize=(10, 6))
boroughs = sample_avail["neighbourhood group"].unique()
palette_b = sns.color_palette("Set2", len(boroughs))
for borough, color in zip(boroughs, palette_b):
    grp = sample_avail[sample_avail["neighbourhood group"] == borough]
    ax.scatter(grp["availability 365"], grp["price"], alpha=0.3, s=15,
               edgecolors="none", label=borough, color=color)
ax.set_title("Price vs Availability by Borough (sample 3,000)", fontsize=14, fontweight="bold")
ax.set_xlabel("Availability (Days / Year)")
ax.set_ylabel("Price (USD, ≤ $1,000)")
ax.legend(title="Borough", fontsize=9)
plt.tight_layout()
plt.savefig("plots/multi_price_avail_borough.png", dpi=150)
plt.show()

# Room Type & Instant Bookable vs Price  (grouped boxplot)
# Adds instant_bookable as a third dimension to the room-type–price relationship.
fig, ax = plt.subplots(figsize=(11, 5))
sns.boxplot(data=biv_price, x="room type", y="price", hue="instant_bookable",
            palette="Set2", ax=ax)
ax.set_title("Price by Room Type & Instant Bookable", fontsize=14, fontweight="bold")
ax.set_xlabel("Room Type")
ax.set_ylabel("Price (USD, ≤ $1,000)")
ax.legend(title="Instant Bookable", fontsize=9)
plt.tight_layout()
plt.savefig("plots/multi_roomtype_bookable_price.png", dpi=150)
plt.show()

print("\nAll multivariate plots saved to plots/")
