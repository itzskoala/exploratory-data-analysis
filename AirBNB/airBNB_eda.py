#Pandas is a Python library used for working with data sets.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

import kagglehub


warnings.filterwarnings('ignore')

df_airBNB = pd.read_csv("Airbnb_Open_Data.csv")
# print(df_airBNB.tail())

'''
What is the commercial enterprise goal or research question you are trying to address?
What are the variables inside the information, and what do they mean?
What are the data sorts (numerical, categorical, textual content, etc.) ?
Is there any known information on first-class troubles or obstacles?
Are there any relevant area-unique issues or constraints?

Dictonary of variable info: https://docs.google.com/spreadsheets/d/1b_dvmyhb_kAJhUmv81rAxl4KcXn0Pymz/edit?gid=1967362979#gid=1967362979

🧠 If you want it to sound REALLY smart in your project:

Frame it like:

“What drives pricing dynamics in the Airbnb marketplace?”
“What factors contribute to high-performing listings?”
“How is demand distributed across geographic regions?”
“What host behaviors correlate with success on the platform?”
💡 My recommendation (best 4–5 to focus on)

If you don’t want to overdo it, pick:

What drives price?
What drives high ratings?
How does location impact price + demand?
Do superhosts outperform regular hosts?
What differentiates top listings?
'''


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
df_airBNB["neighbourhood"] = df_airBNB["neighbourhood"].fillna("Unknown Neighbourhood") #host has to at least have one (the current AirBNB)
df_airBNB["country"] = df_airBNB["country"].fillna("United States") #host has to at least have one (the current AirBNB)
df_airBNB["country code"] = df_airBNB["country code"].fillna("US") #host has to at least have one (the current AirBNB)
df_airBNB["Construction year"] = df_airBNB["Construction year"].fillna(df_airBNB["Construction year"].median()) #host has to at least have one (the current AirBNB)
df_airBNB["availability 365"] = df_airBNB["availability 365"].fillna(df_airBNB["availability 365"].median()) #host has to at least have one (the current AirBNB)


print(df_airBNB.isnull().sum())
