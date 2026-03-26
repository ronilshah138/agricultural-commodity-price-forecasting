import pandas as pd
df = pd.read_csv("backend/data/processed/cleaned_prices.csv")
print("Top Valid Pairs:")
print(df.groupby(['commodity', 'state']).size().sort_values(ascending=False).head(10))
