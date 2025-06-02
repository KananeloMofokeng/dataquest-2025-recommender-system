import pandas as pd
import matplotlib.pyplot as plt

# ✅ Load the dataset
input_path = r'C:\Users\nyaka\Downloads\DATAQUEST\dq_recsys_challenge_2025(in).csv'
df = pd.read_csv(input_path, encoding='ascii')

# Store original row count
original_row_count = df.shape[0]

# ✅ Convert 'int_date' to datetime format
df['int_date'] = pd.to_datetime(df['int_date'], format='%d-%b-%y')

# ✅ Fill missing 'item_descrip' with a placeholder
missing_item_descrip_before = df['item_descrip'].isnull().sum()
df['item_descrip'] = df['item_descrip'].fillna('NO_ITEM_DISPLAY_ONLY')
missing_item_descrip_after = df['item_descrip'].isnull().sum()

# ✅ Standardize categorical columns (strip & lowercase)
cat_cols = ['interaction', 'page', 'tod', 'item_type', 'segment', 'beh_segment', 'active_ind']
df[cat_cols] = df[cat_cols].apply(lambda col: col.str.strip().str.lower())

# ✅ Smarter duplicate removal: based on user, item, and interaction
duplicate_count = df.duplicated(subset=['idcol', 'item', 'interaction']).sum()
df = df.drop_duplicates(subset=['idcol', 'item', 'interaction'])
final_row_count = df.shape[0]

# ✅ Create time-based features from 'int_date'
df['year'] = df['int_date'].dt.year
df['month'] = df['int_date'].dt.month
df['day'] = df['int_date'].dt.day
df['weekday'] = df['int_date'].dt.day_name()

# ✅ Save cleaned data to a new CSV file
output_path = r'C:\Users\nyaka\Downloads\DATAQUEST\dq_recsys_challenge_2025_cleaned.csv'
df.to_csv(output_path, index=False)

# ✅ Analyze and visualize remaining missing values
missing_data = df.isnull().sum()
missing_percentage = (missing_data[missing_data > 0] / df.shape[0]) * 100
missing_percentage.sort_values(ascending=True, inplace=True)

# Plot missing values if any remain
if not missing_percentage.empty:
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.barh(missing_percentage.index, missing_percentage, color='#ff6200')

    for i, (value, name) in enumerate(zip(missing_percentage, missing_percentage.index)):
        ax.text(value + 0.5, i, f"{value:.2f}%", ha='left', va='center',
                fontweight='bold', fontsize=18, color='black')

    ax.set_xlim([0, 100])
    plt.title("Percentage of Missing Values", fontweight='bold', fontsize=22)
    plt.xlabel("Percentages (%)", fontsize=16)
    plt.tight_layout()
    plt.show()
else:
    print("✅ No missing values remain after cleaning.")

# ✅ Final Summary
print("\n========== DATA CLEANING SUMMARY ==========")
print(f"Original dataset size: {original_row_count} rows")
print(f"Final dataset size: {final_row_count} rows")
print(f"Rows removed based on ['idcol', 'item', 'interaction'] duplicates: {duplicate_count}")
print(f"Missing 'item_descrip' before: {missing_item_descrip_before}")
print(f"Missing 'item_descrip' after: {missing_item_descrip_after}")

print("\n✔ Cleaned the following:")
print("- Converted 'int_date' to datetime format")
print("- Filled missing 'item_descrip' with 'NO_ITEM_DISPLAY_ONLY'")
print("- Standardized categorical text columns (lowercased & stripped)")
print("- Removed duplicate interactions per user-item-action")
print("- Created new time-based features: year, month, day, weekday")
print(f"\n✅ Cleaned dataset saved to:\n{output_path}")
print("===========================================")



