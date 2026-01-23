import pandas as pd
import matplotlib.pyplot as plt

# ======================
# WCZYTANIE DANYCH
# ======================
df = pd.read_csv(
    "frogger_eval_extended_to_10M_rounded.csv",
    sep=",",
    skipinitialspace=True
)

# upewnij się, że liczby są liczbami
for col in ["timestep", "mean_reward", "std_reward"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["timestep", "mean_reward", "std_reward"])

# ======================
# WYGŁADZANIE
# ======================
WINDOW = 5   # evale są rzadkie → małe okno wystarczy

df["mean_reward_smooth"] = df["mean_reward"].rolling(
    window=WINDOW, min_periods=1
).mean()

df["std_reward_smooth"] = df["std_reward"].rolling(
    window=WINDOW, min_periods=1
).mean()

# ======================
# WYKRESY
# ======================
plt.figure(figsize=(12, 8))

# ---- MEAN REWARD ----
plt.subplot(2, 1, 1)
plt.plot(df["timestep"], df["mean_reward_smooth"], linewidth=2)
plt.title("Mean reward vs timestep")
plt.ylabel("Mean reward")
plt.grid(True)

# ---- EP ERROR (STD) ----
plt.subplot(2, 1, 2)
plt.plot(df["timestep"], df["std_reward_smooth"], linewidth=2)
plt.title("Episode reward variability (std)")
plt.xlabel("Timestep")
plt.ylabel("Std reward")
plt.grid(True)

plt.tight_layout()
plt.show()
