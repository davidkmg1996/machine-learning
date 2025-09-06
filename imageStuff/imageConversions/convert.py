import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib

iPath = pathlib.Path("pics/alien.png")
raw = tf.io.read_file(str(iPath))

iTensor = tf.image.decode_png(raw, channels=3)

print(f"Original shape:: {iTensor.shape}")
plt.imshow(iTensor.numpy())
plt.axis("off")
plt.show()