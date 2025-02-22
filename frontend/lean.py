import pandas as pd

# Load your dataset
file_path = "./qa-pairs - Copy - empty removed.xlsx"  # Replace with the actual path to your dataset
df = pd.read_excel(file_path)

# Drop all rows with any missing values (NaN) in any column
df_cleaned = df.dropna()

# Optionally, save the cleaned dataset back to a new Excel file
df_cleaned.to_excel("cleaned_qa_pairs.xlsx", index=False)

# Print the first few rows to verify
print(df_cleaned.head())
