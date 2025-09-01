import tensorflow as tf
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

'''
Predicting victim count by offenders based on race

'''

offenderStats = pd.read_csv('stats/race_offender_stats.csv')
victimStats = pd.read_csv('stats/race_victim_stats.csv')

merge = pd.merge(offenderStats, victimStats, on="key", suffixes=("_offender", "_victim"))


X = pd.concat([
     merge[["value_offender"]],
     pd.get_dummies(merge["key"], prefix="race")

], axis=1)


scaler_X = StandardScaler()
scaler_Y = StandardScaler()

raceE = pd.get_dummies(merge["key"], prefix="race").values
X_scaledN = scaler_X.fit_transform(merge[["value_offender"]])
X_scaled = np.concatenate([X_scaledN, raceE], 1)
y_scaled = scaler_Y.fit_transform(merge[["value_victim"]])
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y_scaled, test_size=.25, random_state=42)

rModel = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='softplus', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='softplus'),
    tf.keras.layers.Dense(16, activation='softplus'),
    tf.keras.layers.Dense(8, activation='softplus'),
    tf.keras.layers.Dense(1)

])

rModel.compile(
    loss = "mse",
    optimizer = tf.keras.optimizers.Adam(.001),
    metrics =["mse"]
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)


history = rModel.fit(
    X_train, Y_train,
    epochs = 200,
    validation_data = (X_test, Y_test),
    verbose = 2,
    callbacks=[early_stop]
)

predS = rModel.predict(X_scaled)
pred = scaler_Y.inverse_transform(predS)

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