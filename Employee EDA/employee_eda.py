import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import warnings

import kagglehub

import os

os.makedirs("plots", exist_ok=True)

warnings.filterwarnings('ignore')

'''
Which city is each job title from?
Roles between men and women?



'''
df_employee = pd.read_csv("Employee_dataset.csv")

print("Number of Rows: ",df_employee.shape[0]) 
print("Number of columns: ", df_employee.shape[1])
print("Features in the frame: ",df_employee.columns)

print(df_employee.info())
print(df_employee.nunique()) #This will help us in deciding which type of encoding to choose for converting categorical columns into numerical columns
print(df_employee.describe(include ='all'))
print(df_employee.isnull().sum())


#Handling Missing Values
# Zero missing values across all 18 columns! :)

#Check for duplicate values
print(df_employee.duplicated().sum())

#print every unique value and its count for each text column
categorical_cols = df_employee.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\n--- {col} ---")
    print(df_employee[col].value_counts())


plt.rcParams.update({
    "figure.facecolor":   "#FAFAFA",
    "axes.facecolor":     "#FAFAFA",
    "savefig.facecolor":  "#FAFAFA",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.edgecolor":     "#CCCCCC",
    "axes.grid":          True,
    "axes.axisbelow":     True,
    "grid.color":         "#E8E8E8",
    "grid.linewidth":     0.9,
    "font.family":        "sans-serif",
    "axes.titleweight":   "bold",
    "axes.titlesize":     14,
    "axes.labelsize":     11,
    "axes.labelcolor":    "#444444",
    "xtick.color":        "#666666",
    "ytick.color":        "#666666",
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
})

JEWEL = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51",
         "#457B9D", "#6A4C93", "#1982C4", "#8AC926", "#FF595E"]
FEMALE_COLOR   = "#E76F51"
MALE_COLOR     = "#457B9D"
GENDER_PALETTE = {"Female": FEMALE_COLOR, "Male": MALE_COLOR}

def grad_colors(n, cmap_name, lo=0.25, hi=0.90):
    cmap = cm.get_cmap(cmap_name)
    return [cmap(lo + (hi - lo) * i / max(n - 1, 1)) for i in range(n)]

#Univariate Analysis 

#bar chart
fig, ax = plt.subplots(figsize=(9, 5))
employee_count = df_employee["BUSINESS_UNIT"].value_counts()
ax.bar(employee_count.index, employee_count.values,
       color=JEWEL[:len(employee_count)], edgecolor="white", linewidth=1.5, width=0.55)
ax.set_title("Distribution of Business Units", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Business Unit")
ax.set_ylabel("Number of Employees")
_ymax = max(employee_count.values)
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + _ymax * 0.015,
            f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333333")
plt.tight_layout(pad=2)
plt.savefig("plots/business_unit.png", dpi=200, bbox_inches="tight")
plt.show()

#bar chart for gender distribution
fig, ax = plt.subplots(figsize=(7, 5))
gender_count = df_employee["gender_full"].value_counts()
colors_g = [GENDER_PALETTE.get(g, "#888888") for g in gender_count.index]
ax.bar(gender_count.index, gender_count.values, color=colors_g, edgecolor="white", linewidth=1.5, width=0.45)
ax.set_title("Gender Distribution", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Gender")
ax.set_ylabel("Number of Employees")
_ymax = max(gender_count.values)
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + _ymax * 0.015,
            f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333333")
plt.tight_layout(pad=2)
plt.savefig("plots/gender_distribution.png", dpi=200, bbox_inches="tight")
plt.show()

#bar chart for top 10 birth years
df_employee["birthdate_key"] = pd.to_datetime(df_employee["birthdate_key"], errors="coerce").dt.year
birth_year_count = df_employee["birthdate_key"].dropna().astype(int).value_counts().head(10).sort_index()
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(birth_year_count.index.astype(str), birth_year_count.values,
       color=grad_colors(len(birth_year_count), "cool"), edgecolor="white", linewidth=1.2, width=0.65)
ax.set_title("Top 10 Birth Years Among Employees", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Birth Year")
ax.set_ylabel("Number of Employees")
_ymax = max(birth_year_count.values)
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + _ymax * 0.015,
            f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333333")
plt.tight_layout(pad=2)
plt.savefig("plots/birth_year_distribution.png", dpi=200, bbox_inches="tight")
plt.show()

#bar chart for top 10 ages
current_year = pd.Timestamp.now().year
employee_age = (current_year - df_employee["birthdate_key"]).dropna().astype(int)
age_count = employee_age.value_counts().head(10).sort_index()
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(age_count.index.astype(str), age_count.values,
       color=grad_colors(len(age_count), "autumn_r"), edgecolor="white", linewidth=1.2, width=0.65)
ax.set_title("Top 10 Ages Among Employees", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Age")
ax.set_ylabel("Number of Employees")
_ymax = max(age_count.values)
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + _ymax * 0.015,
            f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333333")
plt.tight_layout(pad=2)
plt.savefig("plots/age_distribution_top10.png", dpi=200, bbox_inches="tight")
plt.show()


#Bivariate Analysis

#Use STATUS to identify terminated employees
df_employee["is_terminated"] = df_employee["STATUS"].eq("TERMINATED")

#1) Does termination rate differ by gender?
termination_by_gender = (df_employee.groupby("gender_full")["is_terminated"].mean() * 100).sort_values(ascending=False)
print("\n1) Termination rate by gender (%):")
print(termination_by_gender.round(2))
fig, ax = plt.subplots(figsize=(7, 5))
colors_tg = [GENDER_PALETTE.get(g, "#888888") for g in termination_by_gender.index]
ax.bar(termination_by_gender.index, termination_by_gender.values, color=colors_tg, edgecolor="white", linewidth=1.5, width=0.45)
ax.set_title("Termination Rate by Gender", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Gender")
ax.set_ylabel("Termination Rate (%)")
_ymax = max(termination_by_gender.values)
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + _ymax * 0.015,
            f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333333")
plt.tight_layout(pad=2)
plt.savefig("plots/termination_rate_by_gender.png", dpi=200, bbox_inches="tight")
plt.show()

#2) Do employees in certain job titles leave more?
#Filter to larger job groups so rates are not dominated by tiny sample sizes.
job_title_termination = df_employee.groupby("job_title").agg(
    employees=("EmployeeID", "count"),
    termination_rate=("is_terminated", "mean")
)
job_title_termination["termination_rate_pct"] = job_title_termination["termination_rate"] * 100
high_attrition_titles = (
    job_title_termination[job_title_termination["employees"] >= 100]
    .sort_values("termination_rate_pct", ascending=False)
    [["employees", "termination_rate_pct"]]
    .head(10)
)
print("\n2) Job titles with highest termination rate (min 200 employees):")
print(high_attrition_titles.round(2))
fig, ax = plt.subplots(figsize=(13, 6))
# Highest risk = darkest color, so reverse after sampling
colors_jt = grad_colors(len(high_attrition_titles), "YlOrRd")[::-1]
ax.bar(high_attrition_titles.index, high_attrition_titles["termination_rate_pct"],
       color=colors_jt, edgecolor="white", linewidth=1.2, width=0.6)
ax.set_title("Top Job Titles by Termination Rate (min 200 employees)", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Job Title")
ax.set_ylabel("Termination Rate (%)")
ax.tick_params(axis="x", rotation=35)
_ymax = max(high_attrition_titles["termination_rate_pct"])
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + _ymax * 0.015,
            f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333333")
plt.tight_layout(pad=2)
plt.savefig("plots/termination_rate_by_job_title.png", dpi=200, bbox_inches="tight")
plt.show()

#3) Do older employees stay longer (age vs length of service)?
age_service_corr = df_employee["age"].corr(df_employee["length_of_service"])
print(f"\n3) Correlation between age and length of service: {age_service_corr:.3f}")
fig, ax = plt.subplots(figsize=(9, 6))
sns.regplot(data=df_employee, x="age", y="length_of_service",
            scatter_kws={"alpha": 0.12, "s": 12, "color": "#457B9D", "rasterized": True},
            line_kws={"color": "#E63946", "linewidth": 2.5}, ax=ax)
ax.set_title(f"Age vs Length of Service (corr = {age_service_corr:.3f})", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Age")
ax.set_ylabel("Length of Service (years)")
plt.tight_layout(pad=2)
plt.savefig("plots/age_vs_length_of_service.png", dpi=200, bbox_inches="tight")
plt.show()


#4) Do certain cities skew younger or older?
city_age = df_employee.groupby("city_name")["age"].mean().sort_values()
youngest_cities = city_age.head(10)
oldest_cities = city_age.tail(10)

print("\n4) Cities with youngest average age:")
print(youngest_cities.round(2))
print("\n4) Cities with oldest average age:")
print(oldest_cities.round(2))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor("#FAFAFA")

# Youngest cities — cool aqua/blue gradient
axes[0].barh(youngest_cities.index, youngest_cities.values,
             color=grad_colors(len(youngest_cities), "GnBu"), edgecolor="white", linewidth=1.2, height=0.65)
axes[0].set_title("10 Youngest Cities (Avg Age)", fontsize=12, fontweight="bold", pad=12)
axes[0].set_xlabel("Average Age")
axes[0].set_ylabel("City")
axes[0].set_facecolor("#FAFAFA")
axes[0].spines[["top", "right"]].set_visible(False)
_xmax = max(youngest_cities.values)
for bar in axes[0].patches:
    axes[0].text(bar.get_width() + _xmax * 0.008, bar.get_y() + bar.get_height() / 2,
                 f"{bar.get_width():.1f}", va="center", fontsize=8, fontweight="bold", color="#333333")

# Oldest cities — warm amber/red gradient
axes[1].barh(oldest_cities.index, oldest_cities.values,
             color=grad_colors(len(oldest_cities), "OrRd"), edgecolor="white", linewidth=1.2, height=0.65)
axes[1].set_title("10 Oldest Cities (Avg Age)", fontsize=12, fontweight="bold", pad=12)
axes[1].set_xlabel("Average Age")
axes[1].set_ylabel("City")
axes[1].set_facecolor("#FAFAFA")
axes[1].spines[["top", "right"]].set_visible(False)
_xmax = max(oldest_cities.values)
for bar in axes[1].patches:
    axes[1].text(bar.get_width() + _xmax * 0.008, bar.get_y() + bar.get_height() / 2,
                 f"{bar.get_width():.1f}", va="center", fontsize=8, fontweight="bold", color="#333333")

plt.tight_layout(pad=2)
plt.savefig("plots/city_age_skew.png", dpi=200, bbox_inches="tight")
plt.show()

#Multivariate Analysis

#1) Does termination rate differ by gender?
termination_gender_dept = (
    df_employee.groupby(["department_name", "gender_full"])["is_terminated"]
    .mean()
    .mul(100)
    .reset_index(name="termination_rate_pct")
)
termination_pivot = termination_gender_dept.pivot(
    index="department_name", columns="gender_full", values="termination_rate_pct"
)
print("\nMultivariate 1) Termination rate by department and gender (%):")
print(termination_pivot.round(2))

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(termination_pivot, annot=True, fmt=".2f", cmap="YlOrRd",
            linewidths=0.6, linecolor="#FFFFFF",
            annot_kws={"size": 11, "weight": "bold"},
            cbar_kws={"shrink": 0.8, "label": "Termination Rate (%)"},
            ax=ax)
ax.set_title("Termination Rate (%) by Department and Gender", fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Gender")
ax.set_ylabel("Department")
ax.tick_params(axis="x", rotation=0)
ax.tick_params(axis="y", rotation=0)
plt.tight_layout(pad=2)
plt.savefig("plots/mv_termination_by_gender_dept.png", dpi=200, bbox_inches="tight")
plt.show()

#2) Are young employees leaving voluntarily? (age + termtype_desc + department_name)
terminated_df = df_employee[df_employee["is_terminated"]].copy()
terminated_df["age_group"] = pd.cut(
    terminated_df["age"],
    bins=[0, 24, 34, 44, 54, 64, 100],
    labels=["<=24", "25-34", "35-44", "45-54", "55-64", "65+"]
)
voluntary_rate = (
    terminated_df.groupby(["department_name", "age_group"])["termtype_desc"]
    .apply(lambda s: (s == "Voluntary").mean() * 100)
    .reset_index(name="voluntary_pct")
)
voluntary_pivot = voluntary_rate.pivot(
    index="department_name", columns="age_group", values="voluntary_pct"
)
print("\nMultivariate 2) Voluntary termination share (%) among terminated by department and age group:")
print(voluntary_pivot.round(2))

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(voluntary_pivot, annot=True, fmt=".1f", cmap="PuBuGn",
            linewidths=0.6, linecolor="#FFFFFF",
            annot_kws={"size": 10, "weight": "bold"},
            cbar_kws={"shrink": 0.8, "label": "Voluntary Termination (%)"},
            ax=ax)
ax.set_title("Voluntary Termination Share (%) by Department and Age Group", fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Age Group")
ax.set_ylabel("Department")
ax.tick_params(axis="x", rotation=0)
ax.tick_params(axis="y", rotation=0)
plt.tight_layout(pad=2)
plt.savefig("plots/mv_voluntary_attrition_age_dept.png", dpi=200, bbox_inches="tight")
plt.show()

#3) What does the highest-risk employee look like?
risk_profile = (
    df_employee.groupby(["department_name", "termtype_desc"])
    .agg(
        termination_rate_pct=("is_terminated", lambda s: s.mean() * 100),
        avg_age=("age", "mean"),
        avg_length_of_service=("length_of_service", "mean"),
        employees=("EmployeeID", "count")
    )
    .reset_index()
    .query("employees >= 100")
    .sort_values("termination_rate_pct", ascending=False)
)
top_risk_profiles = risk_profile.head(10)
print("\nMultivariate 3) Highest-risk profile combinations:")
print(top_risk_profiles.round(2))

fig, ax = plt.subplots(figsize=(13, 6))
profile_labels = top_risk_profiles["department_name"] + "\n" + top_risk_profiles["termtype_desc"]
_n_rp = len(top_risk_profiles)
# Highest risk = darkest red
ax.bar(range(_n_rp), top_risk_profiles["termination_rate_pct"],
       color=grad_colors(_n_rp, "Reds")[::-1], edgecolor="white", linewidth=1.2, width=0.6)
ax.set_xticks(range(_n_rp))
ax.set_xticklabels(profile_labels, rotation=30, ha="right", fontsize=8.5)
ax.set_title("Top Risk Profiles (Department + Term Type)", fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Profile")
ax.set_ylabel("Termination Rate (%)")
_ymax = max(top_risk_profiles["termination_rate_pct"])
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + _ymax * 0.015,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#333333")
plt.tight_layout(pad=2)
plt.savefig("plots/mv_highest_risk_profiles.png", dpi=200, bbox_inches="tight")
plt.show()

#4) What predicts how long someone stays? (age + gender_full + department_name + length_of_service)
retention_predictors = (
    df_employee.groupby(["department_name", "gender_full"])
    .agg(
        avg_length_of_service=("length_of_service", "mean"),
        avg_age=("age", "mean"),
        employees=("EmployeeID", "count")
    )
    .reset_index()
    .query("employees >= 100")
    .sort_values("avg_length_of_service", ascending=False)
)
print("\nMultivariate 4) Retention patterns by department and gender:")
print(retention_predictors.head(15).round(2))
age_los_corr = df_employee["age"].corr(df_employee["length_of_service"])
print(f"\nOverall age vs length_of_service correlation: {age_los_corr:.3f}")

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    data=retention_predictors,
    x="avg_age",
    y="avg_length_of_service",
    hue="department_name",
    style="gender_full",
    size="employees",
    sizes=(60, 350),
    palette="tab10",
    alpha=0.88,
    ax=ax
)
ax.set_title("Retention Predictors: Avg Age vs Avg Length of Service", fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Average Age")
ax.set_ylabel("Average Length of Service (years)")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0,
          frameon=True, framealpha=0.92, fontsize=8.5)
plt.tight_layout(pad=2)
plt.savefig("plots/mv_retention_predictors.png", dpi=200, bbox_inches="tight")
plt.show()

#5) How has attrition changed over time by dept/gender?
attrition_trend = (
    df_employee.groupby(["STATUS_YEAR", "department_name", "gender_full"])["is_terminated"]
    .mean()
    .mul(100)
    .reset_index(name="attrition_rate_pct")
)
top_departments = (
    df_employee["department_name"].value_counts().head(4).index.tolist()
)
trend_filtered = attrition_trend[attrition_trend["department_name"].isin(top_departments)]
print("\nMultivariate 5) Attrition trend sample (top departments):")
print(trend_filtered.head(20).round(2))

g = sns.relplot(
    data=trend_filtered,
    x="STATUS_YEAR",
    y="attrition_rate_pct",
    hue="gender_full",
    col="department_name",
    kind="line",
    col_wrap=2,
    height=4,
    aspect=1.4,
    palette={"Female": FEMALE_COLOR, "Male": MALE_COLOR},
    linewidth=2.5,
    markers=True,
)
g.set_axis_labels("Status Year", "Attrition Rate (%)")
g.figure.suptitle("Attrition Over Time by Department and Gender", y=1.03, fontsize=13, fontweight="bold")
g.figure.patch.set_facecolor("#FAFAFA")
for _ax in g.axes.flat:
    _ax.set_facecolor("#FAFAFA")
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.grid(True, color="#E8E8E8", linewidth=0.9)
g.legend.set_title("Gender")
plt.tight_layout()
plt.savefig("plots/mv_attrition_trend_dept_gender.png", dpi=200, bbox_inches="tight")
plt.show()

