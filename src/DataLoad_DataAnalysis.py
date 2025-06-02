import pandas as pd

df = pd.read_csv(r'C:\Users\nyaka\Downloads\DATAQUEST\dq_recsys_challenge_2025_cleaned.csv', encoding='ascii')

print(df.head(10))
print()
print(df.info())
print(df.describe().T)
print(df.describe(include='object').T)
