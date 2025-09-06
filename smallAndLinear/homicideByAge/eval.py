import tensorflow as tf
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


offenderStats = pd.read_csv('stats/age_offender_stats.csv')
victimStats = pd.read_csv('stats/age_victim_stats.csv')

merge = pd.merge(offenderStats, victimStats, on="key", suffixes=("_offender", "_victim"))
merge["value_offender_log"] = np.log1p(merge["value_offender"])
merge["value_victim_log"] = np.log1p(merge["value_victim"])

age_dummies = pd.get_dummies(merge["key"], prefix="age")
features = pd.concat([merge[["value_offender_log"]], age_dummies], axis=1)

scaler_X = StandardScaler()
scaler_Y = StandardScaler()

age = pd.get_dummies(merge["key"], prefix="age").values

X_scaled = scaler_X.fit_transform(features)
y_scaled = scaler_Y.fit_transform(merge[["value_victim_log"]])
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y_scaled, test_size=.25, random_state=42)

tf.random.set_seed(42)
np.random.seed(42)
#Tweak
rModel = tf.keras.Sequential([
    # tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    # # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(32),
    # tf.keras.layers.Dense(16, activation='relu'),
    # tf.keras.layers.Dense(1)
])

rModel.compile(
    loss = "mse",
    optimizer = tf.keras.optimizers.Adam(.001),
    metrics =["mse"]
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20
    # restore_best_weights=True
)

# lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
#     monitor = 'val_loss',
#     factor = 0.5,
#     patience=5,  
#     min_lr=1e-6,
#     verbose=1
# )


history = rModel.fit(
    X_train, Y_train,
    epochs = 500,
    validation_data = (X_test, Y_test),
    verbose = 2,
    callbacks=[early_stop]
    # callbacks = [lr_schedule]
 )

predS = rModel.predict(X_scaled)
pred_log = scaler_Y.inverse_transform(predS)
pred = np.expm1(pred_log)  # Reverse np.log1p
pred = np.clip(pred, 0, None)

results = pd.DataFrame({
        "Age": merge["key"],
        "Offender Count": merge["value_offender"],
        "Actual Victim Count": merge["value_victim"],
        "Predicted Victim Count": pred.flatten()
})

filtered_results = results[results["Age"].str.strip().str.lower() != "unknown"]

plt.figure(figsize=(12,6))

results_melted = filtered_results.melt(id_vars=["Age"], 
                              value_vars=["Actual Victim Count","Predicted Victim Count"],
                              var_name="Type", value_name="Victim Count")

print(filtered_results)
sns.barplot(data=results_melted, x="Age", y="Victim Count", hue="Type")
plt.xticks(rotation=45, ha="right")
plt.title("Actual vs Predicted Victim Counts by Age")
plt.tight_layout()
plt.show()

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()
