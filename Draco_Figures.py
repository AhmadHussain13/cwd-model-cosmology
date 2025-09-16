import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to your cleaned Draco CSV
csv_path = "/storage/emulated/0/Download/draco_cleaned.csv"

# Load CSV
df = pd.read_csv(csv_path)

# Make sure numeric columns are properly typed
numeric_cols = ['ra','dec','vlos','vlos_error','teff','logg','feh']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Output directory for figures
output_dir = "/storage/emulated/0/Download/draco_figures/"
os.makedirs(output_dir, exist_ok=True)

# 1. Velocity (vlos) vs RA
plt.figure(figsize=(8,5))
plt.errorbar(df['ra'], df['vlos'], yerr=df['vlos_error'], fmt='o', markersize=3, alpha=0.7)
plt.xlabel('RA [deg]')
plt.ylabel('v_los [km/s]')
plt.title('Draco Line-of-Sight Velocities vs RA')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'vlos_vs_ra.png'))
plt.close()

# 2. Histogram of velocities
plt.figure(figsize=(8,5))
plt.hist(df['vlos'], bins=30, color='skyblue', edgecolor='black')
plt.xlabel('v_los [km/s]')
plt.ylabel('Number of Stars')
plt.title('Histogram of Draco Line-of-Sight Velocities')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'vlos_histogram.png'))
plt.close()

# 3. Metallicity (feh) vs RA
plt.figure(figsize=(8,5))
plt.scatter(df['ra'], df['feh'], c='red', s=15, alpha=0.7)
plt.xlabel('RA [deg]')
plt.ylabel('[Fe/H]')
plt.title('Draco Metallicity vs RA')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feh_vs_ra.png'))
plt.close()

# 4. HR-like plot: Teff vs logg
plt.figure(figsize=(8,5))
plt.scatter(df['teff'], df['logg'], c=df['feh'], cmap='viridis', s=15)
plt.colorbar(label='[Fe/H]')
plt.xlabel('Teff [K]')
plt.ylabel('log(g) [cgs]')
plt.title('Draco HR-like Diagram')
plt.gca().invert_xaxis()  # HR diagram convention
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'teff_vs_logg.png'))
plt.close()

print(f"All figures saved to: {output_dir}")