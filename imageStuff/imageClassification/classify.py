import tensorflow as tf
from tensorflow.python.client import device_lib
import kagglehub

#Print device type
#device_type: "CPU"
print(device_lib.list_local_devices())
path = kagglehub.dataset_download("mahmoudreda55/satellite-image-classification")
print(path)

bSize = 16
iHeight = 128
iWidth = 128

data = tf.keras.utils.image_dataset_from_directory(
    path + "/data/",
    image_size = (iHeight, iWidth),
    batch_size = bSize
)

train = tf.keras.utils.image_dataset_from_directory(
    path + "/data",
    validation_split = .2,
    seed = 42,
    subset = "training",
    image_size = (iHeight, iWidth),
    batch_size = bSize
)

validate = tf.keras.utils.image_dataset_from_directory(
    path + "/data",
    validation_split = .2,
    seed = 42,
    subset = "validation",
    image_size = (iHeight, iWidth),
    batch_size = bSize
)

classes = data.class_names

#Crate, compile, train

print(f"Classes: {classes}")
