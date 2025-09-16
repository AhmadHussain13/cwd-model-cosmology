import pandas as pd

# ====== 1. File paths ======
# Update these paths if your file is somewhere else
input_path = "/storage/emulated/0/Download/draco_all_3000_walker_04012025.txt"
output_path = "/storage/emulated/0/Download/draco_cleaned.csv"

# ====== 2. Read the TXT file ======
# sep=',' because the file seems comma-separated, skip comment lines starting with '#'
df = pd.read_csv(input_path, sep=',', comment='#')

# ====== 3. Convert all columns to numeric where possible ======
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # non-numeric entries become NaN

# ====== 4. Optional: Keep only useful columns ======
# For rotation curve, we mainly need: 'ra', 'dec', 'vlos', 'vlos_error', 'teff', 'logg', 'feh'
columns_to_keep = ['ra', 'dec', 'vlos', 'vlos_error', 'teff', 'logg', 'feh']
df_cleaned = df[columns_to_keep]

# ====== 5. Save cleaned data as CSV ======
df_cleaned.to_csv(output_path, index=False)

print(f"Draco data cleaned and saved as:\n{output_path}")