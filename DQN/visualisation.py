import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(
    "models_results/best_nie_normalizowane.txt", sep=",", skipinitialspace=True
)

WINDOW = 50  # im większe = gładsza linia

df["reward_smooth"] = df["ep_real_reward"].rolling(window=WINDOW, min_periods=1).mean()

df["loss_smooth"] = df["loss"].rolling(window=WINDOW, min_periods=1).mean()

# ======================
# WYKRESY
# ======================
plt.figure(figsize=(12, 8))

# ---- REWARD ----
plt.subplot(2, 1, 1)
plt.plot(df["episode"], df["reward_smooth"], linewidth=2)
plt.title("Reward per episode")
plt.ylabel("Reward")
plt.grid(True)

# ---- LOSS ----
plt.subplot(2, 1, 2)
plt.plot(df["episode"], df["loss_smooth"], linewidth=2)
plt.title("Training loss per episode")
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.grid(True)

plt.tight_layout()
plt.show()
