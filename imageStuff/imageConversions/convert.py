import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib

iPath = pathlib.Path("C:/Users/markdk01/machineL/machine-learning/imageStuff/pics/alien.png")
raw = tf.io.read_file(str(iPath))

iTensor = tf.image.decode_png(raw, channels=3)

print(f"Original shape:: {iTensor.shape}")
plt.imshow(iTensor.numpy())
plt.axis("off")
plt.show()