import pandas as pd

# Assuming your CSV files are named 'individuals.csv' and 'spouse.csv'
file_path_individuals = '../spark/data/individuals.csv'
file_path_spouse = '../spark/data/spouse.csv'
file_path_house = '../spark/data/house_pricing.csv'

# Read the CSV files into pandas DataFrames
df_ind = pd.read_csv(file_path_individuals)
df_spouse = pd.read_csv(file_path_spouse)
df_house = pd.read_csv(file_path_house)

# Fill empty values in the 'has_alimony' column with FALSE
df_ind['has_alimony'].fillna(False, inplace=True)
df_spouse['has_alimony'].fillna(False, inplace=True)

df_ind.to_csv('../spark/data/individuals_updated.csv', index=False)
df_spouse.to_csv('../spark/data/spouse_updated.csv', index=False)
print(df_ind.dtypes)
print(df_spouse.dtypes)
print(df_house.dtypes)