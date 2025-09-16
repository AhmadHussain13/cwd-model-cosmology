import pandas as pd
import matplotlib.pyplot as plt

# ====== 1. File path ======
mw_file = "/storage/emulated/0/Download/milky_way_rotation_combined.csv"

# ====== 2. Load data ======
df = pd.read_csv(mw_file)

print(df.head())  # sanity check

# ====== 3. Plot rotation curve ======
plt.errorbar(df['r_kpc'], df['v_km_s'], yerr=df['sigma_km_s'], fmt='o',
             markersize=4, capsize=3, label="Milky Way data")

plt.xlabel("Radius R (kpc)")
plt.ylabel("Velocity (km/s)")
plt.title("Milky Way Rotation Curve")
plt.legend()
plt.grid(True)

# ====== 4. Save and show ======
save_path = "/storage/emulated/0/Download/milkyway_rotationcurve.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')  # high quality
plt.show()

print(f"Figure saved at: {save_path}")