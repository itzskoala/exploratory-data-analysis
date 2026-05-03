'''
How are ratings distributed?
What are the top categories?
Which cities dominate?
What correlates with high ratings?
What features are usable vs messy?
'''

# Import the pandas library as pd for data manipulation
import pandas as pd

# Import the numpy library as np for numerical operations
import numpy as np

# Import the seaborn library as sns for statistical data visualization
import seaborn as sns

# Import the matplotlib.pyplot module as plt for plotting graphs and visualizations
import matplotlib.pyplot as plt

# Import the warnings module to suppress any warnings that might occur during code execution
import warnings

import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px

# Suppress warnings to improve code readability
warnings.filterwarnings('ignore')

df_yelp = pd.read_json("business.json", lines=True)
# print(df_yelp.head())
# print(df_yelp.tail())

#Data Preprocessing
#Check the shape of the DataFrame df_yelp
print("Number of Rows",df_yelp.shape[0]) # 0 prints the number of rows
print("Number of Columns",df_yelp.shape[1]) # 1 prints the number of columns

#Let's see all the columns/features in the DataFrame
print("Features in the frame: ",df_yelp.columns)

#For the features in attributes
print("Features in attributes: ",df_yelp.attributes.keys()) 

#Identify missing values
#Table shows true/false for each cell in the DataFrame
#if true, there is a missing value, else it's false
print("Missing values in the frame: ",df_yelp.isnull())


#the .sum() counts the number of true (MISSING VALUES) in the DataFrame
print("Missing values in the frame: ",df_yelp.isnull().sum())

#Identify missing values.
#What do I have to fill in for what I'm researching for?
# categories → there is only a small number of missing values, so we can drop them
df_yelp = df_yelp.dropna(subset=["categories"])

df_yelp["attributes"] = df_yelp["attributes"].fillna("{}")
df_yelp["hours"] = df_yelp["hours"].fillna("Unknown")

#Verify the data type of each variable.
print("Data type of the variables\n",df_yelp.dtypes)

#Look for errors, invalid values or unusual data points.
# describe() -> generates a statistical summary, useful for checking a range, (1-5 stars)
print(df_yelp["stars"].describe()) 
print((df_yelp["review_count"] < 0).sum()) #review count can't be less than 0 
#unique() returns the items that are unqiue, this var only has a bool, so should be either 1 or 2 
print((df_yelp["is_open"].unique())) 
print(df_yelp["categories"].head())
print(df_yelp["review_count"].sort_values(ascending=False).head())
#df_yelp.describe() shows us there is a large right skew for the STD, 
# meaning a few smaller business have very large review, 75% of businesses have 25 or fewer reviews
#Top 25% of businesses have more than 25 reviews, only a tiny fraction have over 25 reviews
#median is 9 which means heavyyyy right skew
print(df_yelp.describe())


#Handling missing data
#Understand why data is missing, as this helps in selecting the right approach.
#Common for businesses to not share ALL of their details
#https://www.geeksforgeeks.org/data-analysis/handling-missing-values-machine-learning/
print(df_yelp.info())


cleaned_missing_values = df_yelp.isnull().sum()
print(cleaned_missing_values)
#Yah, there are no more True values, no missing values

#check for duplicates via Buisness ID
# df_yelp["business_id"].duplicated().sum()
print("The number of duplicated business are", df_yelp["business_id"].duplicated().sum())


#Exploring Data Characteristics
'''
Central Tendency in Statistics- Mean, Median, Mode
Exploring Data Distribution
Standard Dev -> It helps us understand how spread out the values in a dataset are compared to the mean (average). Varience is just spread of the data in the dataset
Skew -> https://www.geeksforgeeks.org/data-science/difference-between-skewness-and-kurtosis/
'''

print(df_yelp["review_count"].skew()) #The distribution is extremely right-skewed
print(df_yelp["review_count"].kurtosis()) #The dataset has extreme outliers and very heavy tails,
#Leptokurtic: A leptokurtic distribution has kurtosis greater than 3. It has a sharp, narrow peak and heavy tails, meaning the data clusters tightly near the mean but also produces more

#Performing Data Transformation
#variation can impact negatively on the performance of algorithms like KNN, SVM or Logistic Regression
#So we have high variation in the review_count field and the solution is to standardize via feature scaling
#okay so realizing this is only for ML, I'm not doing that here?


# Visualizing Relationship of Data
'''
Bar charts and pie charts help analyze categorical data distribution.
Histograms, box plots and density plots show distribution and detect outliers in numerical data.
Scatter plots and correlation measures help analyze relationships between variables.

1. Univariate Analysis
Univariate analysis studies one variable at a time to understand its characteristics and distribution.

Histograms: Show how data values are distributed.
Box plots: Help detect outliers and show data spread.
Bar charts: Used for categorical variables.


2. Bivariate Analysis
Bivariate analysis examines the relationship between two variables to understand how they interact or influence each other. Common techniques include:

Scatter plots: Show the relationship between two numerical variables.
Correlation coefficient: Measures the strength of the relationship between variables .
Cross-tabulation: Displays the relationship between two categorical variables.
Line graphs: Compare two variables over time to identify trends.
Covariance: Shows how two variables change together.

3. Multivariate Analysis
Multivariate analysis studies three or more variables together to understand complex relationships within the dataset. Common techniques include:

Pair plots: Show relationships between multiple variables at once.
Principal Component Analysis (PCA): Reduces dimensionality while preserving important information.
Spatial analysis: Analyzes geographical patterns using maps and location-based data.

'''

os.makedirs("plots", exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.2)
BLUE   = "#4C72B0"
GREEN  = "#55A868"
ORANGE = "#DD8452"

# ═══════════════════════════════════════════════════════════════════════════════
#  UNIVARIATE ANALYSIS
#  We now look at ONE variable at a time 
# ═══════════════════════════════════════════════════════════════════════════════


# HISTOGRAM: Star Rating Distribution  
# Stars only come in 0.5 increments, so we use discrete bins centered on each increment to avoid gaps.
fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(
    df_yelp["stars"],
    bins=[0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25],
    color=BLUE,
    edgecolor="white",
    linewidth=0.8,
    ax=ax,
)
ax.set_title("Distribution of Star Ratings", fontsize=16, fontweight="bold", pad=14)
ax.set_xlabel("Stars")
ax.set_ylabel("Number of Businesses")
ax.xaxis.set_ticks([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
median_stars = df_yelp["stars"].median()
ax.axvline(median_stars, color=ORANGE, linestyle="--", linewidth=1.8, label=f"Median: {median_stars}")
ax.legend()
plt.tight_layout()
plt.savefig("plots/hist_stars.png", dpi=150)
plt.show()

# Review Count — Raw vs Log-Transformed
# Raw review_count is heavily right-skewed (a few businesses
# have thousands of reviews). The log view reveals the true shape.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df_yelp["review_count"], bins=60, color=BLUE,
             edgecolor="white", linewidth=0.5, ax=axes[0])
axes[0].set_title("Review Count — Raw", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Review Count")
axes[0].set_ylabel("Number of Businesses")

sns.histplot(np.log1p(df_yelp["review_count"]), bins=50, color=GREEN,
             edgecolor="white", linewidth=0.5, kde=True, ax=axes[1])
axes[1].set_title("Review Count — Log Transformed", fontsize=14, fontweight="bold")
axes[1].set_xlabel("log(1 + Review Count)")
axes[1].set_ylabel("")

fig.suptitle("Review Count Distribution", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("plots/hist_review_count.png", dpi=150, bbox_inches="tight")
plt.show()


# Top 15 Cities by Business Count
top_cities = (df_yelp["city"].value_counts().head(15) .sort_values())  # ascending so the longest bar is at the top

#df_yelp["city"].value_counts() = How many businesses are in each city

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(top_cities.index, top_cities.values,
               color=sns.color_palette("viridis", len(top_cities)))
ax.bar_label(bars, padding=4, fontsize=10)
ax.set_title("Top 15 Cities by Number of Businesses", fontsize=16,
             fontweight="bold", pad=14)
ax.set_xlabel("Number of Businesses")
ax.set_xlim(0, top_cities.max() * 1.15)
ax.grid(axis="x", linestyle="--", alpha=0.5)
ax.grid(axis="y", visible=False)
plt.tight_layout()
plt.savefig("plots/bar_top_cities.png", dpi=150)
plt.show()

# Top 15 Business Categories 
# Categories are stored as comma-separated strings — so I should split and explode to count each individual tag.
top_cats = (
    df_yelp["categories"]
    .str.split(", ")
    .explode()
    .str.strip()
    .value_counts()
    .head(15)
    .sort_values()
)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(top_cats.index, top_cats.values,
               color=sns.color_palette("magma", len(top_cats)))
ax.bar_label(bars, padding=4, fontsize=10)
ax.set_title("Top 15 Business Categories", fontsize=16,
             fontweight="bold", pad=14)
ax.set_xlabel("Number of Businesses")
ax.set_xlim(0, top_cats.max() * 1.15)
ax.grid(axis="x", linestyle="--", alpha=0.5)
ax.grid(axis="y", visible=False)
plt.tight_layout()
plt.savefig("plots/bar_top_categories.png", dpi=150)
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
#  BIVARIATE ANALYSIS
#  We now look at TWO variables at a time to see how they relate to each other.
# ═══════════════════════════════════════════════════════════════════════════════

# A derived column I'll reuse throughout this section
df_yelp["log_reviews"] = np.log1p(df_yelp["review_count"])


# CORRELATION HEATMAP 
# The correlation coefficient (r) is a single number between -1 and +1
# that summarises how strongly two variables move together:
#
#   r ≈ +1  → they rise together  (e.g. height and weight)
#   r ≈ -1  → one rises while the other falls
#   r ≈  0  → no linear relationship
#
# A heatmap shows ALL pairs at once — each cell is one r value,
# and colour makes it instant to spot strong relationships.
#


num_cols = ["stars", "review_count", "log_reviews", "is_open"]
corr_matrix = df_yelp[num_cols].corr()

print("\n--- Covariance matrix (hard to interpret because scales differ) ---")
print(df_yelp[num_cols].cov().round(3))
print("\n--- Correlation matrix (scale-free, easier to compare) ---")
print(corr_matrix.round(3))

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1, vmax=1,
    linewidths=0.5,
    square=True,
    ax=ax,
)
ax.set_title("Correlation Matrix — Numerical Features", fontsize=16, fontweight="bold", pad=14)
plt.tight_layout()
plt.savefig("plots/biv_correlation_heatmap.png", dpi=150)
plt.show()



# ═══════════════════════════════════════════════════════════════════════════════
#  MULTIVARIATE ANALYSIS
#  Now we look at THREE OR MORE variables at once to spot complex patterns
#  that pair-wise charts would miss.
# ═══════════════════════════════════════════════════════════════════════════════

# SPATIAL ANALYSIS — Average Rating by State
#Are West Coast businesses rated higher than Midwest? Do Sun Belt cities rate differently?

US_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC",
}
df_us = df_yelp[df_yelp["state"].isin(US_STATES)].copy()

state_stats = (
    df_us.groupby("state")
    .agg(businesses=("business_id", "count"),
         avg_stars=("stars", "mean"),
         lat=("latitude", "mean"),
         lon=("longitude", "mean"))
    .reset_index()
)
state_stats["avg_stars"] = state_stats["avg_stars"].round(2)

fig_states = px.choropleth(
    state_stats,
    locations="state",
    locationmode="USA-states",
    color="avg_stars",
    scope="usa",
    color_continuous_scale="RdYlGn",
    range_color=[3.2, 4.2],
    hover_name="state",
    hover_data={"businesses": True, "avg_stars": True},
    title="Average Yelp Star Rating by US State",
    labels={"avg_stars": "Avg Rating ", "businesses": "# Businesses"},
    template="plotly_white",
)
fig_states.update_layout(
    title_font_size=20,
    margin={"r": 0, "t": 50, "l": 0, "b": 0},
    coloraxis_colorbar=dict(title="Avg Rating"),
)
fig_states.write_image("plots/map_avg_stars_by_state.png", width=1400, height=800, scale=2)
print("State rating map saved → plots/map_avg_stars_by_state.png")


df_yelp["log_reviews"] = np.log1p(df_yelp["review_count"]) 

# DO MORE POPULAR BUSINESSES GET HARSHER RATINGS?
# Bin businesses into review-volume tiers, then examine how star ratings
# distribute within each tier.  A drop in ratings at higher tiers would
# support the "harsher expectations" hypothesis.

review_labels = ["1–5\n(Very Few)", "6–15\n(Few)", "16–40\n(Moderate)",
                 "41–100\n(Many)", "101–500\n(Very Many)", "500+\n(Top)"]
df_yelp["review_bin"] = pd.cut(
    df_yelp["review_count"],
    bins=[0, 5, 15, 40, 100, 500, df_yelp["review_count"].max() + 1],
    labels=review_labels,
)

bin_stats = (
    df_yelp.groupby("review_bin", observed=True)["stars"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"mean": "mean_stars"})
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.violinplot(data=df_yelp, x="review_bin", y="stars",
               order=review_labels, palette="viridis",
               inner="quartile", cut=0, ax=axes[0])
axes[0].set_title("Star Rating Distribution by Review Volume", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Review Volume Tier")
axes[0].set_ylabel("Star Rating")

bars = axes[1].bar(bin_stats["review_bin"], bin_stats["mean_stars"],
                   color=sns.color_palette("viridis", len(bin_stats)), edgecolor="white")
for bar, row in zip(bars, bin_stats.itertuples()):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"n={row.count:,}", ha="center", va="bottom", fontsize=8)
axes[1].axhline(df_yelp["stars"].mean(), color=ORANGE, linestyle="--", linewidth=1.5,
                label=f"Overall mean: {df_yelp['stars'].mean():.2f}")
axes[1].set_title("Mean Star Rating by Review Volume Tier", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Review Volume Tier")
axes[1].set_ylabel("Mean Star Rating")
axes[1].set_ylim(0, 5.5)
axes[1].legend()

fig.suptitle("Do More Popular Businesses Get Harsher Ratings?",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("plots/rating_vs_review_volume.png", dpi=150, bbox_inches="tight")
plt.show()


# HOW IS ATTENTION DISTRIBUTED? (LORENZ CURVE) 
# The Lorenz curve reveals how unequally reviews are distributed.
# The Gini coefficient (0 = perfect equality, 1 = all reviews to one business)
# summarises the concentration in a single number.

rc_sorted = np.sort(df_yelp["review_count"].values)
cumshare_biz = np.linspace(0, 1, len(rc_sorted) + 1)[1:]
cumshare_rev = np.cumsum(rc_sorted) / rc_sorted.sum()
gini = 1 - 2 * np.trapezoid(cumshare_rev, cumshare_biz)

top_pcts   = [1, 5, 10, 25, 50]
top_shares = []
for p in top_pcts:
    cutoff = int((1 - p / 100) * len(rc_sorted))
    top_shares.append(rc_sorted[cutoff:].sum() / rc_sorted.sum() * 100)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].plot(cumshare_biz * 100, cumshare_rev * 100, color=BLUE, linewidth=2, label="Lorenz curve")
axes[0].plot([0, 100], [0, 100], "--", color="gray", linewidth=1.2, label="Perfect equality")
axes[0].fill_between(cumshare_biz * 100, cumshare_rev * 100, cumshare_biz * 100,
                     alpha=0.15, color=BLUE)
for pct in [50, 80, 90]:
    idx = int(pct / 100 * len(rc_sorted)) - 1
    rev_pct = cumshare_rev[idx] * 100
    axes[0].annotate(
        f"Bottom {pct}%:\n{rev_pct:.1f}% of reviews",
        xy=(pct, rev_pct), xytext=(pct - 25, rev_pct + 12),
        fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8),
    )
axes[0].set_title(f"Lorenz Curve — Review Distribution\nGini = {gini:.3f}",
                  fontsize=13, fontweight="bold")
axes[0].set_xlabel("% of Businesses (sorted by review count)")
axes[0].set_ylabel("% of Total Reviews")
axes[0].legend()
axes[0].grid(alpha=0.3)

bars = axes[1].bar([f"Top {p}%" for p in top_pcts], top_shares,
                   color=sns.color_palette("Blues_r", len(top_pcts)), edgecolor="white")
axes[1].bar_label(bars, fmt="%.1f%%", padding=3, fontsize=11, fontweight="bold")
axes[1].set_title("Share of All Reviews Held\nby Top X% of Businesses", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Business Percentile (by review count)")
axes[1].set_ylabel("% of All Reviews")
axes[1].set_ylim(0, 110)
axes[1].grid(axis="y", linestyle="--", alpha=0.4)

fig.suptitle("How Is Attention Distributed Across Businesses?",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("plots/review_concentration.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Gini coefficient (review concentration): {gini:.3f}")


# WHICH TYPES OF BUSINESSES PERFORM BEST? 

cat_stats = (
    df_yelp[["categories", "stars", "review_count"]]
    .assign(category=df_yelp["categories"].str.split(", "))
    .explode("category")
    .assign(category=lambda x: x["category"].str.strip())
    .groupby("category")
    .agg(avg_stars=("stars", "mean"), count=("stars", "count"),
         avg_reviews=("review_count", "mean"))
    .query("count >= 200")
    .sort_values("avg_stars")
)

top20 = cat_stats.tail(20)
bot10 = cat_stats.head(10)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

bars_top = axes[0].barh(top20.index, top20["avg_stars"],
                        color=sns.color_palette("Greens_d", 20))
for bar, (_, row) in zip(bars_top, top20.iterrows()):
    axes[0].text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{row.avg_stars:.2f}  (n={int(row['count']):,})", va="center", fontsize=8)
axes[0].axvline(df_yelp["stars"].mean(), color=ORANGE, linestyle="--", linewidth=1.2,
                label=f"Overall avg {df_yelp['stars'].mean():.2f}")
axes[0].set_xlim(0, 5.8)
axes[0].set_title("Top 20 Highest-Rated Categories\n(min 200 businesses)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Average Star Rating")
axes[0].legend(fontsize=9)
axes[0].grid(axis="x", linestyle="--", alpha=0.4)
axes[0].grid(axis="y", visible=False)

bars_bot = axes[1].barh(bot10.index, bot10["avg_stars"],
                        color=sns.color_palette("Reds_d", 10))
for bar, (_, row) in zip(bars_bot, bot10.iterrows()):
    axes[1].text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{row.avg_stars:.2f}  (n={int(row['count']):,})", va="center", fontsize=8)
axes[1].axvline(df_yelp["stars"].mean(), color=ORANGE, linestyle="--", linewidth=1.2,
                label=f"Overall avg {df_yelp['stars'].mean():.2f}")
axes[1].set_xlim(0, 5.8)
axes[1].set_title("Bottom 10 Lowest-Rated Categories\n(min 200 businesses)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Average Star Rating")
axes[1].legend(fontsize=9)
axes[1].grid(axis="x", linestyle="--", alpha=0.4)
axes[1].grid(axis="y", visible=False)

fig.suptitle("Which Types of Businesses Perform Best?",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("plots/categories_by_rating.png", dpi=150, bbox_inches="tight")
plt.show()


# HOW CONSISTENT ARE RATINGS? 
# Standard deviation of stars measures rating disagreement.
# Low SD → customers agree; high SD → polarising business.

overall_mean = df_yelp["stars"].mean()
overall_std  = df_yelp["stars"].std()

cat_consistency = (
    df_yelp[["categories", "stars"]]
    .assign(category=df_yelp["categories"].str.split(", "))
    .explode("category")
    .assign(category=lambda x: x["category"].str.strip())
    .groupby("category")["stars"]
    .agg(["mean", "std", "count"])
    .query("count >= 500")
    .rename(columns={"mean": "avg_stars", "std": "std_stars"})
    .sort_values("std_stars")
    .reset_index()
)

most_consistent  = cat_consistency.head(10)
least_consistent = cat_consistency.tail(10)
combined_c8 = pd.concat([
    most_consistent.assign(group="Most Consistent (low SD)"),
    least_consistent.assign(group="Least Consistent (high SD)"),
]).sort_values("std_stars", ascending=False)
bar_colors_c8 = [GREEN if g == "Most Consistent (low SD)" else "#E06060"
                 for g in combined_c8["group"]]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

star_counts = df_yelp["stars"].value_counts().sort_index()
axes[0].bar(star_counts.index, star_counts.values, width=0.4, color=BLUE, edgecolor="white")
for bar in axes[0].patches:
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 300,
                 f"{int(bar.get_height()):,}", ha="center", fontsize=7)
axes[0].axvline(overall_mean, color=ORANGE, linestyle="--", linewidth=2,
                label=f"Mean: {overall_mean:.2f}")
axes[0].axvline(overall_mean - overall_std, color="gray", linestyle=":", linewidth=1.5,
                label=f"±1 SD: {overall_std:.2f}")
axes[0].axvline(overall_mean + overall_std, color="gray", linestyle=":", linewidth=1.5)
axes[0].set_title(f"Overall Star Distribution\nMean={overall_mean:.2f}, SD={overall_std:.2f}",
                  fontsize=13, fontweight="bold")
axes[0].set_xlabel("Star Rating"); axes[0].set_ylabel("Number of Businesses")
axes[0].xaxis.set_ticks([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
axes[0].legend()

axes[1].barh(combined_c8["category"], combined_c8["std_stars"],
             color=bar_colors_c8, alpha=0.85)
axes[1].axvline(overall_std, color=ORANGE, linestyle="--", linewidth=1.5,
                label=f"Overall SD: {overall_std:.2f}")
from matplotlib.patches import Patch
axes[1].legend(handles=[
    Patch(color=GREEN, label="Most Consistent (low SD)"),
    Patch(color="#E06060", label="Least Consistent (high SD)"),
    plt.Line2D([0], [0], color=ORANGE, linestyle="--", label=f"Overall SD: {overall_std:.2f}"),
], fontsize=9, loc="lower right")
axes[1].set_title("Rating Consistency by Category\n(SD of stars, min 500 businesses)",
                  fontsize=13, fontweight="bold")
axes[1].set_xlabel("Standard Deviation of Stars")
axes[1].grid(axis="x", linestyle="--", alpha=0.4)

fig.suptitle("How Consistent Are Ratings Across Businesses?",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("plots/rating_consistency.png", dpi=150, bbox_inches="tight")
plt.show()

# DOES INFORMATION COMPLETENESS MATTER?
# Score each business: 1pt for having hours, 1pt for having attributes.
# Then compare average stars and median review count per tier.

df_yelp["has_hours"] = df_yelp["hours"].apply(
    lambda h: 0 if str(h) in {"None", "Unknown", "nan", "{}"} or h is None else 1
)
df_yelp["has_attributes"] = df_yelp["attributes"].apply(
    lambda a: 0 if a is None or a == {} or str(a) in {"{}", "nan", "None"} else 1
)
df_yelp["attr_count"] = df_yelp["attributes"].apply(
    lambda a: len(a) if isinstance(a, dict) else 0
)
df_yelp["completeness"] = df_yelp["has_hours"] + df_yelp["has_attributes"]

tier_labels = {
    0: "Neither\n(no hours, no attrs)",
    1: "Partial\n(hours OR attrs)",
    2: "Complete\n(hours AND attrs)",
}
tier_order = list(tier_labels.values())
df_yelp["completeness_tier"] = df_yelp["completeness"].map(tier_labels)

tier_stats = (
    df_yelp.groupby("completeness_tier")
    .agg(avg_stars=("stars", "mean"),
         median_reviews=("review_count", "median"),
         count=("stars", "count"))
    .reindex(tier_order)
    .reset_index()
)

colors_comp = [ORANGE, BLUE, GREEN]

fig, axes = plt.subplots(1, 3, figsize=(17, 6))

bars = axes[0].bar(tier_stats["completeness_tier"], tier_stats["avg_stars"],
                   color=colors_comp, edgecolor="white")
axes[0].bar_label(bars, fmt="%.2f", padding=3, fontsize=10, fontweight="bold")
axes[0].axhline(df_yelp["stars"].mean(), color="gray", linestyle="--",
                label=f"Overall {df_yelp['stars'].mean():.2f}")
axes[0].set_ylim(0, 5.5)
axes[0].set_title("Avg Star Rating\nby Info Completeness", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Completeness Tier"); axes[0].set_ylabel("Avg Stars")
axes[0].legend(fontsize=9)

bars2 = axes[1].bar(tier_stats["completeness_tier"], tier_stats["median_reviews"],
                    color=colors_comp, edgecolor="white")
axes[1].bar_label(bars2, fmt="%.0f reviews", padding=3, fontsize=9, fontweight="bold")
for bar, row in zip(bars2, tier_stats.itertuples()):
    axes[1].text(bar.get_x() + bar.get_width() / 2, 0.5,
                 f"n={row.count:,}", ha="center", fontsize=9, color="white", fontweight="bold")
axes[1].axhline(df_yelp["review_count"].median(), color="gray", linestyle="--",
                label=f"Overall median: {df_yelp['review_count'].median():.0f}")
axes[1].set_title("Median Review Count\nby Info Completeness", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Completeness Tier"); axes[1].set_ylabel("Median Review Count")
axes[1].legend(fontsize=9)

attr_stars = (
    df_yelp.groupby("attr_count")["stars"]
    .agg(["mean", "count"])
    .query("count >= 50")
    .rename(columns={"mean": "avg_stars"})
    .reset_index()
)
axes[2].scatter(attr_stars["attr_count"], attr_stars["avg_stars"],
                s=attr_stars["count"] / attr_stars["count"].max() * 400 + 20,
                color=BLUE, alpha=0.65, edgecolors="white")
axes[2].axhline(df_yelp["stars"].mean(), color=ORANGE, linestyle="--", linewidth=1.5,
                label=f"Overall mean {df_yelp['stars'].mean():.2f}")
axes[2].set_title("Avg Stars vs. # Attributes Listed\n(bubble = # businesses)",
                  fontsize=12, fontweight="bold")
axes[2].set_xlabel("Number of Attributes Provided"); axes[2].set_ylabel("Average Star Rating")
axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

fig.suptitle("Does Business Information Completeness Matter?",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("plots/info_completeness.png", dpi=150, bbox_inches="tight")
plt.show()


#SUMMARY DASHBOARD 
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Yelp Business Dataset — Summary Dashboard", fontsize=16, fontweight="bold")

star_counts = df_yelp["stars"].value_counts().sort_index()
axes[0, 0].bar(star_counts.index, star_counts.values, width=0.4, color=BLUE, edgecolor="white")
axes[0, 0].set_title("Star Rating Distribution", fontweight="bold")
axes[0, 0].set_xlabel("Stars"); axes[0, 0].set_ylabel("# Businesses")
axes[0, 0].xaxis.set_ticks([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])

rc_all = np.sort(df_yelp["review_count"].values)
axes[0, 1].plot(rc_all, np.arange(1, len(rc_all) + 1) / len(rc_all) * 100,
                color=GREEN, linewidth=1.5)
axes[0, 1].set_xscale("log")
axes[0, 1].axvline(df_yelp["review_count"].median(), color=ORANGE, linestyle="--",
                   label=f"Median: {df_yelp['review_count'].median():.0f}")
axes[0, 1].set_title("CDF of Review Count (log scale)", fontweight="bold")
axes[0, 1].set_xlabel("Review Count (log)"); axes[0, 1].set_ylabel("Cumulative %")
axes[0, 1].legend()

oc_counts = df_yelp["is_open"].value_counts()
axes[0, 2].pie([oc_counts.get(1, 0), oc_counts.get(0, 0)],
               labels=["Open", "Closed"], colors=[GREEN, "#E06060"],
               autopct="%1.1f%%", startangle=90, wedgeprops=dict(edgecolor="white"))
axes[0, 2].set_title("Open vs Closed Businesses", fontweight="bold")

comp_counts = df_yelp["completeness_tier"].value_counts().reindex(tier_order)
axes[1, 0].bar(comp_counts.index, comp_counts.values, color=colors_comp, edgecolor="white")
for bar in axes[1, 0].patches:
    axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 300,
                    f"{int(bar.get_height()):,}", ha="center", fontsize=8)
axes[1, 0].set_title("Info Completeness Breakdown", fontweight="bold")
axes[1, 0].set_ylabel("# Businesses")
axes[1, 0].tick_params(axis="x", labelsize=8)

top5_cats = (
    df_yelp[["categories"]]
    .assign(category=df_yelp["categories"].str.split(", "))
    .explode("category")
    .assign(category=lambda x: x["category"].str.strip())
    ["category"].value_counts().head(5)
)
axes[1, 1].barh(top5_cats.index[::-1], top5_cats.values[::-1],
                color=sns.color_palette("viridis", 5))
axes[1, 1].set_title("Top 5 Categories by Business Count", fontweight="bold")
axes[1, 1].set_xlabel("# Businesses")

stats_text = (
    f"Total Businesses:  {len(df_yelp):,}\n"
    f"Avg Star Rating:   {df_yelp['stars'].mean():.2f} \n"
    f"Median Stars:      {df_yelp['stars'].median():.1f} \n"
    f"SD of Stars:       {df_yelp['stars'].std():.2f}\n"
    f"Median Reviews:    {df_yelp['review_count'].median():.0f}\n"
    f"Mean Reviews:      {df_yelp['review_count'].mean():.1f}\n"
    f"% Open:            {df_yelp['is_open'].mean() * 100:.1f}%\n"
    f"Review Gini:       {gini:.3f}"
)
axes[1, 2].axis("off")
axes[1, 2].text(0.08, 0.5, stats_text, transform=axes[1, 2].transAxes,
                fontsize=12, verticalalignment="center", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=1", facecolor="aliceblue",
                          edgecolor=BLUE, linewidth=1.5))
axes[1, 2].set_title("Key Statistics — Typical Business", fontweight="bold")

plt.tight_layout()
plt.savefig("plots/summary_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()

