import tensorflow as tf
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

fPath = "stats"
csv = {}
offenders, victims = {}, {}

for file in os.listdir(fPath):
    if file.endswith(".csv"):
        file_path = os.path.join(fPath, file)
        df = pd.read_csv(file_path)
        name = os.path.splitext(file)[0]
        csv[name] = df

        lname = name.lower()
        if "offender" in lname:
            offenders[name] = df
        elif "victim" in lname:
            victims[name] = df

def normalize_name(name):
    for suffix in ["_offender", "_victim", "_race", "_stats"]:
        name = name.lower().replace(suffix, "")
    return name.replace("__", "_").strip("_")

# Merge matching pairs
merged_dfs = {}
for off_name, off_df in offenders.items():
    off_base = normalize_name(off_name)
    for vic_name, vic_df in victims.items():
        vic_base = normalize_name(vic_name)
        if off_base == vic_base:
            print(f"Merging: {off_name} + {vic_name}")
            merged = pd.merge(off_df, vic_df, on="key", suffixes=("_offender", "_victim"))
            merged_dfs[off_base] = merged

offenders_df_list = []
for name, df in offenders.items():
    temp_df = df.copy()
    temp_df["dataset"] = name  # optional: track original CSV name
    offenders_df_list.append(temp_df)

offenders_df = pd.concat(offenders_df_list, ignore_index=True)

# Convert victims dictionary to a single DataFrame
victims_df_list = []
for name, df in victims.items():
    temp_df = df.copy()
    temp_df["dataset"] = name  # optional: track original CSV name
    victims_df_list.append(temp_df)

victims_df = pd.concat(victims_df_list, ignore_index=True)

offender_agg = offenders_df.groupby("key", as_index=False)["value"].sum()

# Aggregate victim counts by race
victim_agg = victims_df.groupby("key", as_index=False)["value"].sum()

# Inspect
print("Offenders DataFrame:")
print(offenders_df.head())
print("Victims DataFrame:")
print(victims_df.head())

# Example: inspect one merged DataFrame
# example_key = list(merged_dfs.keys())[0]
# print(merged_dfs[example_key].head())


'''
Predicting victim count by offenders based on race

'''


merge = pd.merge(offender_agg, victim_agg, on="key", how="outer", suffixes=("_offender", "_victim"))

scaler_X = StandardScaler()
scaler_Y = StandardScaler()

raceE = pd.get_dummies(merge["key"], prefix="race").values
X_scaledN = scaler_X.fit_transform(merge[["value_offender"]])
X_scaled = np.concatenate([X_scaledN, raceE], 1)
y_scaled = scaler_Y.fit_transform(merge[["value_victim"]])
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y_scaled, test_size=.25, random_state=42)

tf.random.set_seed(42)
np.random.seed(42)
rModel = tf.keras.Sequential([
    # tf.keras.layers.Dense(64, activation='softplus', input_shape=(X_train.shape[1],)),
    # tf.keras.layers.Dense(32, activation='softplus'),
    tf.keras.layers.Dense(16, activation='softplus'),
    tf.keras.layers.Dense(8, activation='softplus'),
    tf.keras.layers.Dense(1)

])

rModel.compile(
    loss = "mse",
    optimizer = tf.keras.optimizers.Adam(.001),
    metrics =["mse"]
)

# early_stop = tf.keras.callbacks.EarlyStopping(
#     monitor="val_loss",
#     patience=20,
#     restore_best_weights=True
# )


history = rModel.fit(
    X_train, Y_train,
    epochs = 5,
    validation_data = (X_test, Y_test),
    verbose = 2
    # callbacks=[early_stop]
)

predS = rModel.predict(X_scaled)
pred = np.clip(scaler_Y.inverse_transform(predS), 0, None)

results = pd.DataFrame({
    "Race": merge["key"],
    "Offender Count": merge["value_offender"],
    "Actual Victim Count": merge["value_victim"],
    "Predicted Victim Count": pred.flatten()
})

print(results)
plt.figure(figsize=(12,6))
results_melted = results.melt(id_vars=["Race"], 
                              value_vars=["Actual Victim Count","Predicted Victim Count"],
                              var_name="Type", value_name="Victim Count")

sns.barplot(data=results_melted, x="Race", y="Victim Count", hue="Type")
plt.xticks(rotation=45, ha="right")
plt.title("Actual vs Predicted Victim Counts by Race")
plt.tight_layout()
plt.show()

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()